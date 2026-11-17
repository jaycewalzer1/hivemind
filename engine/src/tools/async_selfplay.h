#pragma once

// Out-of-step bughouse self-play (custom engine spec, section 4).
//
// A C++ copy of the arena's event-driven game loop: arena/sim.py (Simulator) and
// arena/state.py (BughouseState). The two boards run independently on four
// running clocks; every timed thing goes through one heap ordered by
// (time ms, insertion counter), exactly as sim.py does, so a game is fully
// deterministic for deterministic players. The emitted events are the ones
// docs/events.md defines and sim.py emits, with the same fields.
//
// Only what the arena's own players can reach without messages or machine
// lending is ported: no message channel, no borrow/lend, charge_wall_time off.
// tools/async_parity.py (repo root) checks this file against the Python arena
// event for event with a deterministic stub player.
//
// TODO(custom-engine): an engine-driven Player (Hivemind search on the clock
// network, FreePass, TreeMoveMs = think time, 'go nodes N sample' recording).
// Only the stub player exists so far.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Fairy-Stockfish/src/position.h"

namespace async_selfplay {

// Seat indices: board = seat / 2, colour = seat % 2 (Stockfish WHITE = 0, BLACK = 1).
enum Seat : int { A_WHITE = 0, A_BLACK = 1, B_WHITE = 2, B_BLACK = 3 };
constexpr int NUM_SEATS = 4;

const char* seat_name(int seat);
const char* team_of(int seat);       // "T1" (A_white, B_black) or "T2"
int partner_of(int seat);            // the other board, other colour
int opponent_of(int seat);           // same board, other colour
inline int board_of(int seat) { return seat / 2; }
inline char board_name(int board) { return board == 0 ? 'A' : 'B'; }

struct GameResult {
    std::string result;  // "T1" | "T2" | "draw"
    std::string reason;  // "checkmate" | "flag" | "max_plies" | "aborted"
    int board = -1;      // -1 = none
    int loserSeat = -1;  // -1 = none
};

struct MoveResult {
    std::string san;
    std::string capture;  // piece letter as it enters the partner's pocket, "" = none
    std::string fen;
};

/// arena/state.py BughouseState: two crazyhouse boards, partner pocket transfer
/// (a captured promoted piece transfers as a pawn), four clocks, FICS mate rule.
class BughouseState {
  public:
    BughouseState(int64_t baseMs, int64_t incrementMs, const std::string& fenA, const std::string& fenB);

    std::string fen(int board) const;
    int to_move(int board) const;
    /// UCI strings in Fairy-Stockfish generation order; empty when the game is
    /// over or it is not this seat's turn.
    std::vector<std::string> legal_moves(int seat) const;
    const std::optional<GameResult>& result() const { return result_; }

    /// Charge elapsed time to the side to move on each board. Returns the flag
    /// result the first time a clock reaches 0.
    std::optional<GameResult> advance_clocks(int64_t nowMs);
    /// Throws std::invalid_argument on an illegal move or a flag before landing.
    MoveResult apply_move(int seat, const std::string& uci, int64_t nowMs);

    int64_t clocks[NUM_SEATS];
    int64_t baseMs;
    int64_t incrementMs;
    int64_t lastMs = 0;

  private:
    std::unique_ptr<Stockfish::Position> pos_[2];
    Stockfish::StateListPtr states_[2];
    std::optional<GameResult> result_;
    int halfmove_[2] = {0, 0};  ///< python-chess halfmove clock per board
};

/// Fairy-Stockfish has no notion of "a drop could block": the arena's geometric rule.
bool is_bughouse_mate(const Stockfish::Position& pos);

struct Observation {
    int64_t nowMs = 0;
    int seat = 0;
    const BughouseState* state = nullptr;
    std::vector<std::string> legalMoves;
    int ply = 0;
};

struct Decision {
    bool isMove = false;
    std::string uci;
    int64_t thinkMs = 0;
    int64_t waitMs = 0;
    int64_t nodes = 0;
    std::string stallReason;  // for waits; "" = null
};

class Player {
  public:
    virtual ~Player() = default;
    virtual Decision decide(const Observation& obs) = 0;
    virtual std::string name() const = 0;
    virtual std::string kind() const = 0;
};

/// SplitMix64. The Python stub (tools/async_parity.py) uses the same generator.
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

/// Deterministic stub. Per decision: r = rng.next(); no legal move -> wait
/// waitMs ("no_moves"); r % 100 < waitPct -> wait waitMs ("stub_wait");
/// otherwise move sorted(legal)[rng.next() % n] with a fixed think time.
/// Seeded with seed * 4 + seat.
class StubPlayer : public Player {
  public:
    StubPlayer(uint64_t seed, int seat, int64_t thinkMs, int64_t waitMs, int waitPct)
        : rng_(seed * 4 + static_cast<uint64_t>(seat)), thinkMs_(thinkMs), waitMs_(waitMs), waitPct_(waitPct) {}
    Decision decide(const Observation& obs) override;
    std::string name() const override { return "stub"; }
    std::string kind() const override { return "stub"; }

  private:
    SplitMix64 rng_;
    int64_t thinkMs_, waitMs_;
    int waitPct_;
};

struct GameConfig {
    std::string gameId;
    std::string setup = "stub";
    std::string fenA, fenB;
    int64_t baseMs = 60000;
    int64_t incrementMs = 0;
    int maxPlies = 400;
    uint64_t seed = 0;
    int64_t thinkMs[NUM_SEATS] = {100, 100, 100, 100};
    int64_t waitMs[NUM_SEATS] = {200, 200, 200, 200};
    int waitPct = 0;
};

/// arena/sim.py Simulator restricted to message-free, lend-free players.
class Simulator {
  public:
    Simulator(BughouseState& state, std::vector<Player*> players, const GameConfig& cfg);
    GameResult run();

    std::vector<std::string> events;  // one JSON object per event, docs/events.md
    std::vector<std::string> trace;   // one JSON object per decide: ts, seat, n legal, fnv1a64 of sorted moves
    int plies = 0;

  private:
    enum Kind { FLAG = 0, DECIDE, LAND, WAKE };
    enum Status { IDLE = 0, PENDING, THINKING, WAITING };
    struct Item {
        int64_t t;
        int64_t counter;
        Kind kind;
        int seat;  // or board for FLAG
        int64_t gen;
        Decision dec;
        int64_t started;
    };
    struct Later {
        bool operator()(const Item& a, const Item& b) const {
            return a.t != b.t ? a.t > b.t : a.counter > b.counter;
        }
    };

    void push(int64_t t, Kind kind, int seat, int64_t gen, Decision dec = {}, int64_t started = 0);
    void emit(const std::string& type, const std::string& fields);
    std::string clocks_json() const;
    bool advance();
    void end(const GameResult& res);
    void schedule_flag_check(int board);
    void start_turn(int seat);
    void wake(int seat);
    void stop_sitting(int seat);
    void start();
    void on_decide(int seat, int64_t gen);
    void on_land(int seat, int64_t gen, const Decision& dec, int64_t started);
    void on_wake(int seat, int64_t gen);

    BughouseState& st_;
    std::vector<Player*> players_;
    GameConfig cfg_;
    int64_t now_ = 0;
    std::optional<GameResult> result_;
    std::vector<Item> heap_;
    int64_t counter_ = 0;
    Status status_[NUM_SEATS] = {IDLE, IDLE, IDLE, IDLE};
    int64_t gen_[NUM_SEATS] = {0, 0, 0, 0};
    bool sitting_[NUM_SEATS] = {false, false, false, false};
};

uint64_t fnv1a64(const std::string& s);

/// Plays one game with stub players and returns
/// {"game_id": ..., "events": [...], "trace": [...]} on one line.
std::string play_stub_game(const GameConfig& cfg);

}  // namespace async_selfplay
