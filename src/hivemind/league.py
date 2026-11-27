"""League self-play and pace randomization for `hivemind rl-loop` (spec v7 section 5,
docs/custom_engine_spec_v7_league.md in the talisman repo).

Two switches, both off by default:

- `--pace`: every non-validation segment plays in the engine's pace mode. Each team draws
  its pace k per game (log-uniform in `--pace-k-range`); a decision costs clock / k and
  searches `--nodes x (--pace-ref-k / k) x clock / base` nodes, clamped to `--pace-nodes`.
- `--league` (needs `--pace`): part of the segments play against frozen opponents instead
  of the generator itself: past promoted generators and the fixed baseline (it13) at fixed
  paces. Only the learner's decisions are written, so a league segment yields about half
  the records of a self segment.

rl-loop (WP1) calls `add_arguments`, `validate`, `active`, `plan` and `segment_summary`;
`plan` gives the iteration's segment table, frozen in stages/league_plan.done. Standard
library only.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

DEFAULT_K_RANGE = (80, 400)
DEFAULT_REF_K = 110.0
DEFAULT_NODES = (200, 3200)
DEFAULT_SHARES = (0.5, 0.25, 0.25)  # self, past, fixed
DEFAULT_PAST = 3
DEFAULT_FIXED_K = (80, 110, 160, 250, 400)
# PFSP ("hard", AlphaStar): an opponent's weight is max(PFSP_FLOOR, 1 - w)^2, w the
# learner's score against it, once it has PFSP_MIN_GAMES learner games; else 1.
PFSP_FLOOR = 0.05
PFSP_MIN_GAMES = 200
# Placeholder rl-loop replaces with a per-process copy of the segment's opponent model.
OPPONENT_PLACEHOLDER = "{opponent_model}"


def _pair(kind):
    def parse(text):
        try:
            low, high = (kind(part) for part in text.split(":"))
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected LOW:HIGH, got {text!r}")
        return low, high
    return parse


def _floats(text):
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")


def _ints(text):
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")


def add_arguments(parser):
    group = parser.add_argument_group("league self-play and pace (spec v7 section 5; off by default)")
    group.add_argument("--pace", action="store_true",
                       help="Pace mode on every non-validation segment: each team draws its pace k per game; "
                            "a decision costs clock / k and its search follows")
    group.add_argument("--pace-k-range", type=_pair(int), default=DEFAULT_K_RANGE, metavar="MIN:MAX",
                       help="Log-uniform range of k (default 80:400)")
    group.add_argument("--pace-ref-k", type=float, default=DEFAULT_REF_K,
                       help="The pace at which a search gets --nodes at a full clock (default 110)")
    group.add_argument("--pace-nodes", type=_pair(int), default=DEFAULT_NODES, metavar="MIN:MAX",
                       help="Clamp of a paced search's budget (default 200:3200)")
    group.add_argument("--league", action="store_true",
                       help="League segments against past generators and the fixed baseline (needs --pace)")
    group.add_argument("--league-shares", type=_floats, default=DEFAULT_SHARES, metavar="SELF,PAST,FIXED",
                       help="Segment shares (default 0.5,0.25,0.25); the past share goes to fixed while "
                            "there is no past generator")
    group.add_argument("--league-past", type=int, default=DEFAULT_PAST,
                       help="Past opponents: the newest promoted generators but the current (default 3)")
    group.add_argument("--league-fixed-k", type=_ints, default=DEFAULT_FIXED_K, metavar="K,...",
                       help="Paces of the fixed opponent, cycled (default 80,110,160,250,400)")
    group.add_argument("--league-fixed-model", type=Path, default=None,
                       help="The fixed opponent (default: the lineage's fixed baseline, <lineage>-it0.onnx)")
    group.add_argument("--league-pfsp", choices=("on", "off"), default="on",
                       help="Weight opponents by the learner's last results against them (PFSP hard; default on)")


def _get(args, name, default):
    """Old namespaces (workers, earlier markers) may lack the spec v7 options."""
    value = getattr(args, name, default)
    return default if value is None and default is not None else value


def active(args):
    """Whether rl-loop plans its segments here (--league or --pace)."""
    return bool(_get(args, "league", False) or _get(args, "pace", False))


def validate(args, parser):
    low, high = _get(args, "pace_k_range", DEFAULT_K_RANGE)
    if not 1 <= low <= high:
        parser.error("--pace-k-range needs 1 <= MIN <= MAX")
    if not _get(args, "pace_ref_k", DEFAULT_REF_K) > 0:
        parser.error("--pace-ref-k must be positive")
    low, high = _get(args, "pace_nodes", DEFAULT_NODES)
    if not 1 <= low <= high:
        parser.error("--pace-nodes needs 1 <= MIN <= MAX")
    shares = _get(args, "league_shares", DEFAULT_SHARES)
    if len(shares) != 3 or any(share < 0 for share in shares) or shares[0] <= 0:
        parser.error("--league-shares needs three shares SELF,PAST,FIXED >= 0 with SELF > 0")
    if _get(args, "league_past", DEFAULT_PAST) < 0:
        parser.error("--league-past must be >= 0")
    fixed_k = _get(args, "league_fixed_k", DEFAULT_FIXED_K)
    if not fixed_k or any(k < 1 for k in fixed_k):
        parser.error("--league-fixed-k needs one or more paces >= 1")
    fixed_model = _get(args, "league_fixed_model", None)
    if fixed_model is not None and not Path(fixed_model).is_file():
        parser.error(f"--league-fixed-model {fixed_model} is not a file")
    pace_changed = (tuple(_get(args, "pace_k_range", DEFAULT_K_RANGE)) != DEFAULT_K_RANGE
                    or _get(args, "pace_ref_k", DEFAULT_REF_K) != DEFAULT_REF_K
                    or tuple(_get(args, "pace_nodes", DEFAULT_NODES)) != DEFAULT_NODES)
    league_changed = (tuple(shares) != DEFAULT_SHARES
                      or _get(args, "league_past", DEFAULT_PAST) != DEFAULT_PAST
                      or tuple(fixed_k) != DEFAULT_FIXED_K or fixed_model is not None
                      or _get(args, "league_pfsp", "on") != "on")
    # A knob without its switch would do nothing: refuse it rather than run a setting the
    # caller believes is on.
    if pace_changed and not _get(args, "pace", False):
        parser.error("--pace-k-range, --pace-ref-k and --pace-nodes need --pace")
    if league_changed and not _get(args, "league", False):
        parser.error("--league-shares, --league-past, --league-fixed-k, --league-fixed-model and "
                     "--league-pfsp need --league")
    if _get(args, "league", False) and not _get(args, "pace", False):
        # The fixed opponents are defined by their pace (--opponent-k is a pace-mode option).
        parser.error("--league needs --pace")


def pace_args(args):
    """The engine's pace-mode options for a non-validation segment ([] without --pace)."""
    if not _get(args, "pace", False):
        return []
    k_low, k_high = _get(args, "pace_k_range", DEFAULT_K_RANGE)
    n_low, n_high = _get(args, "pace_nodes", DEFAULT_NODES)
    return ["--pace-k-min", str(k_low), "--pace-k-max", str(k_high),
            "--pace-ref-k", f"{_get(args, 'pace_ref_k', DEFAULT_REF_K):g}",
            "--pace-min-nodes", str(n_low), "--pace-max-nodes", str(n_high)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def past_generators(runs, n, generator, limit):
    """[(M, rl-itM/models/C.onnx)] of the iterations before n that were promoted, newest
    first, at most `limit`, without the current generator (by sha256) or repeats."""
    if limit <= 0:
        return []
    current = sha256(generator) if Path(generator).is_file() else None
    seen, found = {current}, []
    for m in range(n - 1, 0, -1):
        it = Path(runs) / f"rl-it{m}"
        model = it / "models" / "C.onnx"
        try:
            promoted = json.loads((it / "stages" / "promote.done").read_text()).get("promoted")
        except (OSError, ValueError, AttributeError):
            continue
        if promoted is not True or not model.is_file():
            continue
        digest = sha256(model)
        if digest in seen:
            continue
        seen.add(digest)
        found.append((m, model))
        if len(found) >= limit:
            break
    return found


def previous_results(runs, n):
    """{theta: {"games", "score"}}: the learner's results per opponent over iteration
    n - 1's league segments (their markers' "league" summaries)."""
    stages = Path(runs) / f"rl-it{n - 1}" / "stages"
    totals = {}
    for marker in sorted(stages.glob("seg*.done")) if stages.is_dir() else []:
        try:
            summary = json.loads(marker.read_text()).get("league") or {}
        except (OSError, ValueError, AttributeError):
            continue
        theta = summary.get("theta")
        games = summary.get("learner_games") or 0
        if not theta or theta == "self" or not games:
            continue
        entry = totals.setdefault(theta, {"games": 0, "points": 0.0})
        entry["games"] += games
        entry["points"] += summary.get("wins", 0) + 0.5 * summary.get("draws", 0)
    return {theta: {"games": entry["games"], "score": entry["points"] / entry["games"]}
            for theta, entry in totals.items()}


def pfsp_weight(result):
    """PFSP hard: max(0.05, 1 - w)^2 with >= 200 learner games against the opponent, else 1."""
    if not result or result["games"] < PFSP_MIN_GAMES:
        return 1.0
    return max(PFSP_FLOOR, 1.0 - result["score"]) ** 2


def apportion(total, weights):
    """`total` whole items split in proportion to `weights` (largest remainder, ties to the
    earlier entry)."""
    if total <= 0 or not weights:
        return [0] * len(weights)
    norm = sum(weights)
    exact = [total * weight / norm for weight in weights]
    counts = [math.floor(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def spread(groups):
    """Interleave lists so each is spread evenly over the result (item j of a list of c at
    position (j + 0.5) / c; ties keep the group order)."""
    keyed = [((j + 0.5) / len(group), g, j, item)
             for g, group in enumerate(groups) for j, item in enumerate(group)]
    return [item for *_, item in sorted(keyed, key=lambda entry: entry[:3])]


def _share_counts(segments, shares, have_past):
    self_share, past_share, fixed_share = shares
    if not have_past:
        fixed_share, past_share = fixed_share + past_share, 0.0
    total = self_share + past_share + fixed_share
    n_past = round(segments * past_share / total)
    n_fixed = round(segments * fixed_share / total)
    while segments - n_past - n_fixed < 1:
        if n_fixed >= n_past:
            n_fixed -= 1
        else:
            n_past -= 1
    return segments - n_past - n_fixed, n_past, n_fixed


def plan(args, runs, n, generator, fixed, segment_games):
    """Iteration n's segment table: {"segments": {segNN: {"theta", "seed", "games",
    "engine_args", "opponent_model", "league"}}, "summary": str, "opponents": {...}}.

    S = ceil(--games / segment_games) segments with seeds 1000 n + k and games as without a
    plan. With --league the shares are rounded onto whole segments (self keeps the rest,
    at least one); past opponents are the newest promoted generators (none: their share
    goes to the fixed baseline, which cycles its paces); within a share the segments are
    dealt in proportion to the PFSP weights. `opponent_model` is the opponent's source
    path; its engine_args name it OPPONENT_PLACEHOLDER for rl-loop to substitute."""
    games = args.games
    segments = -(-games // segment_games)
    pace = pace_args(args)
    league = bool(_get(args, "league", False))
    pfsp = _get(args, "league_pfsp", "on") == "on"
    fixed_model = Path(_get(args, "league_fixed_model", None) or fixed)
    opponents = {}
    items = [{"theta": "self"}] * segments
    counts = (segments, 0, 0)
    if league:
        past = past_generators(runs, n, generator, _get(args, "league_past", DEFAULT_PAST))
        results = previous_results(runs, n) if pfsp else {}
        counts = _share_counts(segments, _get(args, "league_shares", DEFAULT_SHARES), bool(past))
        fixed_sha = sha256(fixed_model) if fixed_model.is_file() else None
        entries = {"past": [{"theta": f"past:it{m}", "kind": "past", "iteration": m, "model": str(model),
                             "sha256": sha256(model)} for m, model in past],
                   "fixed": [{"theta": f"fixed:k{k}", "kind": "fixed", "k": k, "model": str(fixed_model),
                              "sha256": fixed_sha}
                             for k in dict.fromkeys(_get(args, "league_fixed_k", DEFAULT_FIXED_K))]}
        dealt = {}
        for kind, total in (("past", counts[1]), ("fixed", counts[2])):
            for entry in entries[kind]:
                entry["weight"] = round(pfsp_weight(results.get(entry["theta"])), 6)
                if entry["theta"] in results:
                    entry["previous"] = results[entry["theta"]]
            shares = apportion(total, [entry["weight"] for entry in entries[kind]])
            for entry, count in zip(entries[kind], shares):
                entry["segments"] = count
                opponents[entry["theta"]] = {key: value for key, value in entry.items() if key != "theta"}
            dealt[kind] = spread([[entry] * count for entry, count in zip(entries[kind], shares) if count])
        items = spread([g for g in ([{"theta": "self"}] * counts[0], dealt["past"], dealt["fixed"]) if g])
    table = {}
    for k, item in enumerate(items):
        name = f"seg{k:02d}"
        spec = {"theta": item["theta"], "seed": 1000 * n + k,
                "games": min(segment_games, games - k * segment_games)}
        if item["theta"] == "self":
            spec.update(engine_args=list(pace), opponent_model=None, league={"kind": "self"})
        else:
            engine_args = [*pace, "--opponent-model", OPPONENT_PLACEHOLDER]
            if item["kind"] == "fixed":
                engine_args += ["--opponent-k", str(item["k"])]
            spec.update(engine_args=engine_args, opponent_model=item["model"],
                        league={key: item[key] for key in ("kind", "iteration", "k", "sha256", "weight")
                                if key in item})
        table[name] = spec
    summary = f"it{n}: {segments} segments"
    if league:
        summary += (f", self {counts[0]}, past {counts[1]}, fixed {counts[2]}; "
                    + (", ".join(f"{theta} x{o['segments']} (w {o['weight']:g})"
                                 for theta, o in opponents.items() if o["segments"]) or "no opponents")
                    + f"; PFSP {'on' if pfsp else 'off'}")
    if pace:
        k_low, k_high = _get(args, "pace_k_range", DEFAULT_K_RANGE)
        summary += f"; pace k {k_low}:{k_high}"
    return {"segments": table, "summary": summary, "opponents": opponents}


GAME_LINE = re.compile(r"^selfplay game (\d+)/\d+ ")


def _field(line, key):
    match = re.search(rf" {re.escape(key)} (\S+)", line)
    return match[1] if match else None


def segment_summary(log_path, spec):
    """The learner's results and the paces of a segment from its engine log's game lines
    (`selfplay game ...`; a resumed run's repeated game counts once, its last line):
    {"theta", "games", "learner_games", "wins", "draws", "losses", "score", "mean_k",
    "mean_k_learner", "mean_k_opponent", "mean_s"}. Means are None without data."""
    games = {}
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        match = GAME_LINE.match(line)
        if match:
            games[int(match[1])] = line
    results = {"win": 0, "draw": 0, "loss": 0}
    k_all, k_learner, k_opponent, scales = [], [], [], []
    for line in games.values():
        team = _field(line, "learner_team")
        result = _field(line, "learner_result")
        if result in results:
            results[result] += 1
        paces = {colour: _field(line, f"k_{colour}_team") for colour in ("white", "black")}
        if all(paces.values()):
            paces = {colour: float(value) for colour, value in paces.items()}
            k_all += paces.values()
            if team in paces:
                k_learner.append(paces[team])
                k_opponent.append(paces["black" if team == "white" else "white"])
        if (scale := _field(line, "mean_s")) is not None:
            scales.append(float(scale))

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None

    learner_games = sum(results.values())
    return {"theta": (spec or {}).get("theta"), "games": len(games), "learner_games": learner_games,
            "wins": results["win"], "draws": results["draw"], "losses": results["loss"],
            "score": (round((results["win"] + 0.5 * results["draw"]) / learner_games, 4)
                      if learner_games else None),
            "mean_k": mean(k_all), "mean_k_learner": mean(k_learner), "mean_k_opponent": mean(k_opponent),
            "mean_s": mean(scales)}
