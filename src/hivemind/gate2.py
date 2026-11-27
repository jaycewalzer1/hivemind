"""The real-clock two-score gate (talisman docs/custom_engine_spec_v7_league.md section 3).

rl-loop --gate2 promotes a candidate only if neither of two real-clock scores drops against the incumbent's. Both
networks play two arms of arena games against stock upstream-it13 on PACE (talisman cluster/gate2_match.sh, one net
dir RUN_ROOT/gate2/nets/<sha16>/ per network): `equal` (our wave 2 contender at clock_k 110, no think head) and
`free` (clock_k 160 with the think head). Every network plays the same openings in the same colours, pair by pair, so
the two networks are compared paired: per arm, over the colour-swapped pairs both finished, d = mean(x - y) of the
candidate's and the incumbent's pair scores, SE = sd(x - y) / sqrt(n).

  drop(arm)      d < -margin and d + 1.645 SE < 0 (one-sided 95%: clearly worse by more than noise-level margin)
  complete(arm)  at least min_pairs shared pairs
  promote        both arms complete, neither drops, and d_equal + d_free >= 0

Standard library only. Command line:
  python -m hivemind.gate2 wave2-json [rl-loop wave 2 flags]   the contender's engine options (wave2.json)
  python -m hivemind.gate2 decide CAND_DIR INC_DIR              the decision as JSON"""

import argparse
import json
import math
from pathlib import Path
import sys

ARMS = ("equal", "free")
DROP_MARGIN = 0.02
MIN_PAIRS = 150
Z_ONE_SIDED = 1.645


def read_pairs(net_dir, arm):
    """{(part, pair_index): pair score} of one arm of a net dir, from its merged <arm>/pairs.tsv (columns part,
    pair_index, opening_id, contender_team, score_pair, game_ids); {} before the arm is merged. A pair score is the
    mean of the pair's two game scores: 0, 0.5 or 1 (0.25, 0.75 with a draw)."""
    path = Path(net_dir) / arm / "pairs.tsv"
    if not path.is_file():
        return {}
    pairs = {}
    lines = path.read_text().splitlines()
    header = lines[0].split("\t") if lines else []
    for line in lines[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t")))
        pairs[(int(row["part"]), int(row["pair_index"]))] = float(row["score_pair"])
    return pairs


def compare(cand, inc, margin=DROP_MARGIN, min_pairs=MIN_PAIRS):
    """One arm: the candidate's pair scores `cand` against the incumbent's `inc`, over the pairs both have."""
    shared = sorted(set(cand) & set(inc))
    n = len(shared)
    diffs = [cand[k] - inc[k] for k in shared]
    d = sum(diffs) / n if n else None
    se = (math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n) if n >= 2 else None)
    drop = d is not None and d < -margin and (se is None or d + Z_ONE_SIDED * se < 0)
    return {"pairs": n, "d": d, "se": se, "drop": drop, "complete": n >= min_pairs}


def summary(net_dir, arms=ARMS):
    """{"sha16", arm: the arm's merged summary.json (or None)} of a net dir."""
    net_dir = Path(net_dir)
    out = {"sha16": net_dir.name}
    try:
        config = json.loads((net_dir / "config.json").read_text())
        out["label"] = config.get("label")
        out["sha16"] = str(config.get("sha256") or net_dir.name)[:16]
    except (OSError, ValueError):
        pass
    for arm in arms:
        try:
            out[arm] = json.loads((net_dir / arm / "summary.json").read_text())
        except (OSError, ValueError):
            out[arm] = None
    return out


def _fmt(x, spec="+.3f"):
    return "n/a" if x is None else format(x, spec)


def decide(cand_dir, inc_dir, margin=DROP_MARGIN, min_pairs=MIN_PAIRS, arms=ARMS):
    """The gate's decision for the candidate's net dir against the incumbent's (spec v7 3.2):
    {"candidate", "incumbent" (summaries), "arms": {arm: {"pairs", "d", "se", "drop", "complete"}}, "promoted",
    "reason", "margin", "min_pairs"}."""
    result = {arm: compare(read_pairs(cand_dir, arm), read_pairs(inc_dir, arm), margin, min_pairs) for arm in arms}
    incomplete = [arm for arm in arms if not result[arm]["complete"]]
    drops = [arm for arm in arms if result[arm]["drop"]]
    each = ", ".join(f"{arm} d {_fmt(r['d'])} SE {_fmt(r['se'], '.3f')} over {r['pairs']} pairs"
                     for arm, r in result.items())
    if incomplete:
        promoted = False
        reason = ("incomplete: " + ", ".join(f"{arm} {result[arm]['pairs']} of {min_pairs} pairs"
                                             for arm in incomplete))
    elif drops:
        promoted = False
        reason = f"drop in {' and '.join(drops)}: {each}"
    else:
        total = sum(result[arm]["d"] for arm in arms)
        promoted = total >= 0
        reason = (f"{'promoted' if promoted else 'sum below 0'}: d sum {total:+.3f}; {each}")
    return {"candidate": summary(cand_dir, arms), "incumbent": summary(inc_dir, arms), "arms": result,
            "promoted": promoted, "reason": reason, "margin": margin, "min_pairs": min_pairs}


def wave2_json(argv):
    """Options.as_dict() of rl-loop's wave 2 flags in `argv`; other rl-loop flags are ignored (named on stderr),
    so a launcher can pass its whole WAVE2_EXTRA."""
    from hivemind import wave2

    parser = argparse.ArgumentParser(prog="python -m hivemind.gate2 wave2-json")
    wave2.add_arguments(parser)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"wave2-json: ignoring {' '.join(unknown)}", file=sys.stderr)
    try:
        return wave2.from_args(args).as_dict()
    except ValueError as error:
        parser.error(str(error))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["wave2-json"]:
        print(json.dumps(wave2_json(argv[1:]), indent=1, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(prog="python -m hivemind.gate2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("wave2-json", help="print the contender's engine options for rl-loop's wave 2 flags")
    d = sub.add_parser("decide", help="print the gate's decision for two net dirs as JSON")
    d.add_argument("candidate", type=Path)
    d.add_argument("incumbent", type=Path)
    d.add_argument("--margin", type=float, default=DROP_MARGIN)
    d.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    args = parser.parse_args(argv)
    print(json.dumps(decide(args.candidate, args.incumbent, args.margin, args.min_pairs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
