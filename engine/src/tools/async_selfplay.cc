// Out-of-step bughouse self-play: C++ port of arena/sim.py + arena/state.py.
// See async_selfplay.h. Built as its own executable (async_selfplay) in the
// clock build; it does not touch the hivemind library's behaviour.
//
// Usage:
//   async_selfplay stub < games.tsv > games.jsonl
// One game per input line, tab separated:
//   game_id  fenA  fenB  base_ms  inc_ms  max_plies  seed
//   think_AW,think_AB,think_BW,think_BB  wait_AW,wait_AB,wait_BW,wait_BB  wait_pct
//   [min_move_ms  lag_max_ms]   (optional; 0 = off, as sim.py's defaults)
// One output line per game: {"game_id":..,"events":[..],"trace":[..]}.
//
// TODO(custom-engine): 'async_selfplay engine ...' driving Hivemind searches
// (one Agent per team, FreePass + TreeMoveMs + sample recording). Not built yet.

#include "tools/async_selfplay.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "common/globals.h"
#include "Fairy-Stockfish/src/apiutil.h"
#include "Fairy-Stockfish/src/movegen.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/uci.h"
#include "Fairy-Stockfish/src/variant.h"

namespace async_selfplay {

namespace {

const char* const SEAT_NAMES[NUM_SEATS] = {"A_white", "A_black", "B_white", "B_black"};
const char* const TEAMS[NUM_SEATS] = {"T1", "T2", "T2", "T1"};

std::string jstr(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string jopt(const std::string& s) { return s.empty() ? "null" : jstr(s); }

const char* other_team(const char* team) { return std::string(team) == "T1" ? "T2" : "T1"; }

int seat_of(int board, Stockfish::Color c) { return board * 2 + (c == Stockfish::WHITE ? 0 : 1); }

// state.py repetition_key: FEN placement without the pocket, side to move, castling, en passant.
std::string repetition_key(const std::string& fen) {
    std::istringstream in(fen);
    std::vector<std::string> parts;
    for (std::string w; in >> w;) parts.push_back(w);
    if (parts.empty()) return fen;
    std::string key = parts[0].substr(0, parts[0].find('['));
    for (size_t i = 1; i < 4 && i < parts.size(); ++i) key += " " + parts[i];
    return key;
}

constexpr uint64_t LAG_STREAM = 0x6C61675FULL;  // sim.py _LAG_STREAM ("lag_")

}  // namespace

const char* seat_name(int seat) { return SEAT_NAMES[seat]; }
const char* team_of(int seat) { return TEAMS[seat]; }
int partner_of(int seat) { return 3 - seat; }
int opponent_of(int seat) { return seat ^ 1; }

uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ------------------------------------------------------------------ state

BughouseState::BughouseState(int64_t base, int64_t inc, const std::string& fenA, const std::string& fenB)
    : baseMs(base), incrementMs(inc) {
    const Stockfish::Variant* v = Stockfish::variants.find("bughouse")->second;
    const std::string fens[2] = {fenA, fenB};
    for (int b = 0; b < 2; ++b) {
        pos_[b] = std::make_unique<Stockfish::Position>();
        states_[b] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        pos_[b]->set(v, fens[b], false, &states_[b]->back(), Stockfish::Threads.main());
        halfmove_[b] = pos_[b]->rule50_count();
    }
    for (int s = 0; s < NUM_SEATS; ++s) clocks[s] = base;
    for (int b = 0; b < 2; ++b) seen_[b][repetition_key(fen(b))] = 1;
}

std::string BughouseState::fen(int board) const {
    // Written as python-chess writes it: its own halfmove clock (drops never
    // reset it) and the en passant square only when the capture is legal.
    std::string f = pos_[board]->fen(false, true);
    std::istringstream in(f);
    std::vector<std::string> parts;
    for (std::string w; in >> w;) parts.push_back(w);
    if (parts.size() == 6) {
        parts[4] = std::to_string(halfmove_[board]);
        // python-chess shows the en passant square only when an en passant capture is legal.
        if (parts[3] != "-") {
            bool legalEp = false;
            for (const auto& m : Stockfish::MoveList<Stockfish::LEGAL>(*pos_[board]))
                legalEp = legalEp || Stockfish::type_of(m) == Stockfish::EN_PASSANT;
            if (!legalEp) parts[3] = "-";
        }
        f = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) f += " " + parts[i];
    }
    return f;
}

int BughouseState::to_move(int board) const { return seat_of(board, pos_[board]->side_to_move()); }

std::vector<std::string> BughouseState::legal_moves(int seat) const {
    std::vector<std::string> out;
    const int board = board_of(seat);
    if (result_ || to_move(board) != seat) return out;
    for (const auto& m : Stockfish::MoveList<Stockfish::LEGAL>(*pos_[board]))
        out.push_back(Stockfish::UCI::move(*pos_[board], m));
    return out;
}

std::optional<GameResult> BughouseState::advance_clocks(int64_t nowMs) {
    if (result_) return std::nullopt;
    const int64_t elapsed = std::max<int64_t>(0, nowMs - lastMs);
    lastMs = std::max(lastMs, nowMs);
    // The clock with less time left before the charge ran out first and loses;
    // an exact tie is a double flag, a draw (state.py, spec v3 D4).
    int flagged = 0;
    int64_t bestBefore = 0;
    int loser = -1;
    bool tie = false;
    for (int b = 0; b < 2; ++b) {
        const int seat = to_move(b);
        const int64_t before = clocks[seat];
        clocks[seat] = std::max<int64_t>(0, before - elapsed);
        if (clocks[seat] == 0) {
            if (flagged && before == bestBefore) tie = true;
            if (!flagged || before < bestBefore) {
                bestBefore = before;
                loser = seat;
            }
            ++flagged;
        }
    }
    if (!flagged) return std::nullopt;
    if (tie) {
        result_ = GameResult{"draw", "double_flag", -1, -1};
        return result_;
    }
    result_ = GameResult{other_team(TEAMS[loser]), "flag", board_of(loser), loser};
    return result_;
}

bool is_bughouse_mate(const Stockfish::Position& pos) {
    if (!pos.checkers()) return false;
    if (Stockfish::MoveList<Stockfish::LEGAL>(pos).size() > 0) return false;
    Stockfish::Bitboard checkers = pos.checkers();
    if (Stockfish::popcount(checkers) > 1) return true;
    const Stockfish::Square checker = Stockfish::lsb(checkers);
    const Stockfish::Square king = pos.square<Stockfish::KING>(pos.side_to_move());
    // python-chess chess.between: squares strictly between two aligned squares, else none.
    const int df = int(Stockfish::file_of(king)) - int(Stockfish::file_of(checker));
    const int dr = int(Stockfish::rank_of(king)) - int(Stockfish::rank_of(checker));
    const bool aligned = df == 0 || dr == 0 || std::abs(df) == std::abs(dr);
    const bool hasBetween = aligned && std::max(std::abs(df), std::abs(dr)) > 1;
    return !hasBetween;
}

MoveResult BughouseState::apply_move(int seat, const std::string& uci, int64_t nowMs) {
    if (result_) throw std::invalid_argument("game is over");
    const int b = board_of(seat);
    if (to_move(b) != seat) throw std::invalid_argument("not this seat's turn");
    Stockfish::Position& pos = *pos_[b];
    Stockfish::Move move = Stockfish::MOVE_NONE;
    for (const auto& m : Stockfish::MoveList<Stockfish::LEGAL>(pos)) {
        if (Stockfish::UCI::move(pos, m) == uci) {
            move = m;
            break;
        }
    }
    if (move == Stockfish::MOVE_NONE) throw std::invalid_argument("illegal move " + uci);
    if (advance_clocks(nowMs)) throw std::invalid_argument("flagged before the move landed");

    MoveResult mr;
    mr.san = Stockfish::SAN::move_to_san(pos, move, Stockfish::NOTATION_SAN);
    if (Stockfish::type_of(move) == Stockfish::DROP && mr.san.rfind("P@", 0) == 0) {
        mr.san.erase(0, 1);  // python-chess writes pawn drops as "@e5"
    }
    const bool isDrop = Stockfish::type_of(move) == Stockfish::DROP;
    // python-chess halfmove rule: a drop always counts up; otherwise a capture or pawn move resets.
    if (isDrop) {
        ++halfmove_[b];
    } else if (pos.capture(move) || Stockfish::type_of(pos.moved_piece(move)) == Stockfish::PAWN) {
        halfmove_[b] = 0;
    } else {
        ++halfmove_[b];
    }
    states_[b]->emplace_back();
    pos.do_move(move, states_[b]->back());
    // Fairy-Stockfish's bughouse variant hands the captured piece (a pawn if it
    // was promoted) to the caller; it goes to the partner's pocket on the other board.
    const Stockfish::Piece p = states_[b]->back().pieceToHand;
    if (p != Stockfish::NO_PIECE) {
        pos_[1 - b]->add_to_hand_with_key(p);
        mr.capture = std::string(1, pos.piece_to_char()[p]);
    }
    clocks[seat] += incrementMs;
    mr.fen = fen(b);

    if (is_bughouse_mate(pos)) {
        const int loser = seat_of(b, pos.side_to_move());
        result_ = GameResult{TEAMS[seat], "checkmate", b, loser};
    } else {
        // state.py FICS repetition: board-only, any drop resets the count, threefold is a draw.
        if (isDrop) seen_[b].clear();
        if (++seen_[b][repetition_key(mr.fen)] >= 3) result_ = GameResult{"draw", "repetition", b, -1};
    }
    return mr;
}

// ------------------------------------------------------------------ stub player

Decision StubPlayer::decide(const Observation& obs) {
    Decision d;
    const uint64_t r = rng_.next();
    if (obs.legalMoves.empty() || static_cast<int>(r % 100) < waitPct_) {
        d.isMove = false;
        d.waitMs = waitMs_;
        d.stallReason = obs.legalMoves.empty() ? "no_moves" : "stub_wait";
        return d;
    }
    std::vector<std::string> sorted = obs.legalMoves;
    std::sort(sorted.begin(), sorted.end());
    d.isMove = true;
    d.uci = sorted[rng_.next() % sorted.size()];
    d.thinkMs = thinkMs_;
    return d;
}

// ------------------------------------------------------------------ simulator

Simulator::Simulator(BughouseState& state, std::vector<Player*> players, const GameConfig& cfg)
    : st_(state), players_(std::move(players)), cfg_(cfg), lagRng_(cfg.seed ^ LAG_STREAM) {}

void Simulator::push(int64_t t, Kind kind, int seat, int64_t gen, Decision dec, int64_t started) {
    heap_.push_back(Item{t, counter_++, kind, seat, gen, std::move(dec), started});
    std::push_heap(heap_.begin(), heap_.end(), Later());
}

void Simulator::emit(const std::string& type, const std::string& fields) {
    std::ostringstream ss;
    ss << "{\"type\":" << jstr(type) << ",\"seq\":" << events.size() << ",\"ts\":" << now_;
    if (!fields.empty()) ss << "," << fields;
    ss << "}";
    events.push_back(ss.str());
}

std::string Simulator::clocks_json() const {
    std::ostringstream ss;
    ss << "{";
    for (int s = 0; s < NUM_SEATS; ++s) ss << (s ? "," : "") << jstr(SEAT_NAMES[s]) << ":" << st_.clocks[s];
    ss << "}";
    return ss.str();
}

bool Simulator::advance() {
    std::optional<GameResult> res = st_.advance_clocks(now_);
    if (!res) {
        for (int b = 0; b < 2; ++b) {
            const int seat = st_.to_move(b);
            if (st_.clocks[seat] <= 0) {
                res = res ? GameResult{"draw", "double_flag", -1, -1} : GameResult{TEAMS[opponent_of(seat)], "flag", b, seat};
            }
        }
    }
    if (res) {
        end(*res);
        return true;
    }
    return false;
}

void Simulator::end(const GameResult& res) {
    if (result_) return;
    result_ = res;
    for (int s = 0; s < NUM_SEATS; ++s) status_[s] = IDLE;
    std::ostringstream ss;
    ss << "\"result\":" << jstr(res.result) << ",\"reason\":" << jstr(res.reason)
       << ",\"board\":" << (res.board < 0 ? "null" : jstr(std::string(1, board_name(res.board))))
       << ",\"loserSeat\":" << (res.loserSeat < 0 ? "null" : jstr(SEAT_NAMES[res.loserSeat]));
    emit("game_end", ss.str());
}

void Simulator::schedule_flag_check(int board) {
    const int seat = st_.to_move(board);
    push(now_ + std::max<int64_t>(0, st_.clocks[seat]), FLAG, board, 0);
}

void Simulator::start_turn(int seat) {
    ++gen_[seat];
    status_[seat] = PENDING;
    push(now_, DECIDE, seat, gen_[seat]);
}

void Simulator::wake(int seat) {
    if (status_[seat] == WAITING) start_turn(seat);
}

void Simulator::stop_sitting(int seat) {
    if (sitting_[seat]) {
        sitting_[seat] = false;
        emit("stall", "\"seat\":" + jstr(SEAT_NAMES[seat]) + ",\"active\":false,\"reason\":null");
    }
}

void Simulator::start() {
    now_ = st_.lastMs;
    st_.advance_clocks(now_);
    std::ostringstream ss;
    ss << "\"game_id\":" << jstr(cfg_.gameId) << ",\"setup\":" << jstr(cfg_.setup) << ",\"seats\":{";
    for (int s = 0; s < NUM_SEATS; ++s) {
        ss << (s ? "," : "") << jstr(SEAT_NAMES[s]) << ":{\"name\":" << jstr(players_[s]->name())
           << ",\"team\":" << jstr(TEAMS[s]) << ",\"kind\":" << jstr(players_[s]->kind()) << "}";
    }
    ss << "},\"fens\":{\"A\":" << jstr(st_.fen(0)) << ",\"B\":" << jstr(st_.fen(1)) << "}"
       << ",\"clocks\":" << clocks_json() << ",\"base_ms\":" << st_.baseMs
       << ",\"increment_ms\":" << st_.incrementMs << ",\"channel_delay_ms\":0";
    emit("game_start", ss.str());
    if (st_.result()) {
        end(*st_.result());
        return;
    }
    for (int b = 0; b < 2; ++b) schedule_flag_check(b);
    for (int b = 0; b < 2; ++b) start_turn(st_.to_move(b));
}

GameResult Simulator::run() {
    start();
    while (!result_ && !heap_.empty()) {
        std::pop_heap(heap_.begin(), heap_.end(), Later());
        Item it = std::move(heap_.back());
        heap_.pop_back();
        now_ = std::max(now_, it.t);
        if (advance()) break;
        switch (it.kind) {
            case FLAG: break;  // advance() already ran for this timestamp
            case DECIDE: on_decide(it.seat, it.gen); break;
            case LAND: on_land(it.seat, it.gen, it.dec, it.started); break;
            case WAKE: on_wake(it.seat, it.gen); break;
        }
    }
    if (!result_) end(GameResult{"draw", "aborted", -1, -1});
    return *result_;
}

void Simulator::on_wake(int seat, int64_t gen) {
    if (gen != gen_[seat] || status_[seat] != WAITING) return;
    start_turn(seat);
}

void Simulator::on_decide(int seat, int64_t gen) {
    if (gen != gen_[seat] || status_[seat] != PENDING) return;
    if (st_.to_move(board_of(seat)) != seat) {
        status_[seat] = IDLE;
        return;
    }
    Observation obs;
    obs.nowMs = now_;
    obs.seat = seat;
    obs.state = &st_;
    obs.legalMoves = st_.legal_moves(seat);
    obs.ply = plies;
    {
        std::vector<std::string> sorted = obs.legalMoves;
        std::sort(sorted.begin(), sorted.end());
        std::string joined;
        for (size_t i = 0; i < sorted.size(); ++i) joined += (i ? " " : "") + sorted[i];
        std::ostringstream ts;
        ts << "{\"ts\":" << now_ << ",\"seat\":" << jstr(SEAT_NAMES[seat]) << ",\"n\":" << sorted.size()
           << ",\"h\":\"" << std::hex << fnv1a64(joined) << "\"}";
        trace.push_back(ts.str());
    }
    Decision dec = players_[seat]->decide(obs);
    if (dec.isMove && std::find(obs.legalMoves.begin(), obs.legalMoves.end(), dec.uci) == obs.legalMoves.end()) {
        Decision w;
        w.isMove = false;
        w.waitMs = 200;  // sim.py FALLBACK_WAIT_MS
        w.stallReason = obs.legalMoves.empty() ? "no_moves" : "invalid_move";
        dec = w;
    }
    const int64_t g = ++gen_[seat];
    if (dec.isMove) {
        int64_t cost = std::max<int64_t>(cfg_.minMoveMs, std::max<int64_t>(0, dec.thinkMs));
        if (cfg_.lagMaxMs > 0) cost += static_cast<int64_t>(lagRng_.next() % static_cast<uint64_t>(cfg_.lagMaxMs + 1));
        status_[seat] = THINKING;
        std::ostringstream ss;
        ss << "\"seat\":" << jstr(SEAT_NAMES[seat]) << ",\"phase\":\"start\",\"think_ms\":" << cost
           << ",\"nodes\":" << dec.nodes << ",\"eval\":null,\"depth\":null,\"borrowed\":false";
        emit("thinking", ss.str());
        push(now_ + cost, LAND, seat, g, dec, now_);
    } else {
        status_[seat] = WAITING;
        if (!sitting_[seat]) {
            sitting_[seat] = true;
            emit("stall", "\"seat\":" + jstr(SEAT_NAMES[seat]) + ",\"active\":true,\"reason\":" + jopt(dec.stallReason));
        }
        push(now_ + std::max<int64_t>(1, dec.waitMs), WAKE, seat, g);
    }
}

void Simulator::on_land(int seat, int64_t gen, const Decision& dec, int64_t started) {
    if (gen != gen_[seat] || status_[seat] != THINKING) return;
    const int board = board_of(seat);
    std::ostringstream endFields;
    endFields << "\"seat\":" << jstr(SEAT_NAMES[seat]) << ",\"phase\":\"end\",\"think_ms\":" << (now_ - started)
              << ",\"nodes\":" << dec.nodes << ",\"eval\":null,\"depth\":null,\"borrowed\":false";
    const std::vector<std::string> legal = st_.legal_moves(seat);
    if (std::find(legal.begin(), legal.end(), dec.uci) == legal.end()) {
        emit("thinking", endFields.str());
        start_turn(seat);
        return;
    }
    MoveResult mr;
    try {
        mr = st_.apply_move(seat, dec.uci, now_);
    } catch (const std::invalid_argument&) {
        if (st_.result()) {
            end(*st_.result());
            return;
        }
        emit("thinking", endFields.str());
        start_turn(seat);
        return;
    }
    stop_sitting(seat);
    status_[seat] = IDLE;
    ++plies;
    emit("thinking", endFields.str());
    {
        std::ostringstream ss;
        ss << "\"board\":\"" << board_name(board) << "\",\"seat\":" << jstr(SEAT_NAMES[seat])
           << ",\"uci\":" << jstr(dec.uci) << ",\"san\":" << jstr(mr.san)
           << ",\"isDrop\":" << (dec.uci.find('@') != std::string::npos ? "true" : "false")
           << ",\"capture\":" << jopt(mr.capture) << ",\"fen\":" << jstr(mr.fen) << ",\"clocks\":" << clocks_json();
        emit("move", ss.str());
    }
    if (!mr.capture.empty()) {
        std::ostringstream ss;
        ss << "\"fromBoard\":\"" << board_name(board) << "\",\"fromSeat\":" << jstr(SEAT_NAMES[seat])
           << ",\"toSeat\":" << jstr(SEAT_NAMES[partner_of(seat)]) << ",\"piece\":" << jstr(mr.capture);
        emit("pass", ss.str());
    }
    if (st_.result()) {
        end(*st_.result());
        return;
    }
    if (plies >= cfg_.maxPlies) {
        end(GameResult{"draw", "max_plies", -1, -1});
        return;
    }
    schedule_flag_check(board);
    for (int s = 0; s < NUM_SEATS; ++s) wake(s);
    const int nxt = st_.to_move(board);
    if (status_[nxt] == IDLE) start_turn(nxt);
}

std::string play_stub_game(const GameConfig& cfg) {
    BughouseState state(cfg.baseMs, cfg.incrementMs, cfg.fenA, cfg.fenB);
    std::vector<std::unique_ptr<StubPlayer>> owned;
    std::vector<Player*> players;
    for (int s = 0; s < NUM_SEATS; ++s) {
        owned.push_back(std::make_unique<StubPlayer>(cfg.seed, s, cfg.thinkMs[s], cfg.waitMs[s], cfg.waitPct));
        players.push_back(owned.back().get());
    }
    Simulator sim(state, players, cfg);
    sim.run();
    std::ostringstream out;
    out << "{\"game_id\":" << jstr(cfg.gameId) << ",\"events\":[";
    for (size_t i = 0; i < sim.events.size(); ++i) out << (i ? "," : "") << sim.events[i];
    out << "],\"trace\":[";
    for (size_t i = 0; i < sim.trace.size(); ++i) out << (i ? "," : "") << sim.trace[i];
    out << "]}";
    return out.str();
}

}  // namespace async_selfplay

// ------------------------------------------------------------------ CLI

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, sep)) out.push_back(cur);
    return out;
}

bool parse_line(const std::string& line, async_selfplay::GameConfig& cfg, std::string& err) {
    const auto f = split(line, '\t');
    if (f.size() != 10 && f.size() != 12) {
        err = "expected 10 or 12 tab-separated fields, got " + std::to_string(f.size());
        return false;
    }
    try {
        cfg.gameId = f[0];
        cfg.fenA = f[1];
        cfg.fenB = f[2];
        cfg.baseMs = std::stoll(f[3]);
        cfg.incrementMs = std::stoll(f[4]);
        cfg.maxPlies = std::stoi(f[5]);
        cfg.seed = std::stoull(f[6]);
        const auto think = split(f[7], ','), wait = split(f[8], ',');
        if (think.size() != 4 || wait.size() != 4) {
            err = "think and wait need 4 comma-separated values";
            return false;
        }
        for (int s = 0; s < 4; ++s) {
            cfg.thinkMs[s] = std::stoll(think[s]);
            cfg.waitMs[s] = std::stoll(wait[s]);
        }
        cfg.waitPct = std::stoi(f[9]);
        if (f.size() == 12) {
            cfg.minMoveMs = std::stoll(f[10]);
            cfg.lagMaxMs = std::stoll(f[11]);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode != "stub") {
        std::cerr << "usage: async_selfplay stub < games.tsv > games.jsonl\n"
                     "  (engine-driven self-play is a TODO; only the parity stub runs)\n";
        return 2;
    }
    init_fairy_stockfish();
    std::string line;
    int lineNo = 0;
    while (std::getline(std::cin, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') continue;
        async_selfplay::GameConfig cfg;
        std::string err;
        if (!parse_line(line, cfg, err)) {
            std::cerr << "line " << lineNo << ": " << err << "\n";
            return 1;
        }
        std::cout << async_selfplay::play_stub_game(cfg) << "\n";
    }
    std::cout.flush();
    return 0;
}
