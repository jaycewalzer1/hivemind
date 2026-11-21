"""Wave 2 (talisman fork): self-play and matches on four real clocks, with
the time model in the tree and no hard-coded sit rule.

The engine's wave 2 options (docs/wave2_training.md has the exact contract):

  --durative-tree      per-seat ready times in the tree (UCI DurativeTree)
  --sit-gate           upstream's hard can_sit = lead > 0 (UCI SitGate);
                       wave 2 plays with it off, so a pass is always legal
  --flag-loss          a seat whose clock reaches 0 loses for its team, in
                       the tree and in the game (UCI FlagLoss)
  --instant-transfer   captured pieces usable by the partner at once
  --mate-race-by-landing
  --pass-cap-k         a pass lasts at most k x the root move time
  --base-mix           per-game base clock, MS:WEIGHT,...; every seat starts
                       at the drawn base (then the usual lead odds apply)
  --seat-move-times    each seat draws its own move time, not each team

Wave 2c (talisman docs/custom_engine_spec_v5_continuity.md), off by default:

  --team-tree-reuse    each team keeps its search tree between its decisions
                       and adopts the node the game reached, re-timed to the
                       real clocks (UCI TeamTreeReuse); needs the durative
                       tree and a base mix
  --team-tree-reuse-tolerance-ms
                       largest clock difference a node may have (0..5000)
  --joint-plans        below the root a team that acts plans its other
                       on-turn seat's move in the same joint action (UCI
                       JointPlans); needs the durative tree
  --joint-plan-free-permille
                       prior share of "decide later" (0..1000)
  --plan-carry         the half of a team's chosen action for its on-turn
                       seat that is not deciding is mixed into that seat's
                       next search (UCI PlanCarry); needs the durative tree
                       and a base mix
  --plan-carry-permille
                       prior weight of the carried move (0..1000)
  --continuity         rl-loop only: the three switches above at once, with
                       --own-pace-below-root and --blockable-check-not-mate
                       below (the engine gets the five flags)

Fixes from the durative tree diagnosis (talisman docs/durative_diagnosis.md,
wave2-hivemind docs/wave2_engine.md "Diagnosis fixes"), off by default:

  --own-pace-below-root
                       D1: below the root our seats act at their own observed
                       pace, not the root move time (UCI OwnPaceBelowRoot);
                       needs the durative tree
  --wait-node-acting-eval
                       D3: forced-wait nodes are evaluated from the team that
                       acts next, value negated (UCI WaitNodeActingEval);
                       needs the durative tree; not in --continuity
  --blockable-check-not-mate
                       D5: a check a dropped piece could block is never mate,
                       as in the arena (UCI BlockableCheckNotMate); any tree

Every option's stock value reproduces v3.0.3, so `STOCK` is the engine with
none of them. `WAVE2` is the setting the arena and self-play send.
"""

from dataclasses import dataclass, fields, replace

# The spec v3 D2 base-time mix (cluster/iterate.sh BASE_MIX): 60/120/180/300 s.
DEFAULT_BASE_MIX = ((60_000, 0.25), (120_000, 0.25), (180_000, 0.35), (300_000, 0.15))


def parse_base_mix(text):
    """'60000:0.25,120000:0.75' -> ((60000, 0.25), (120000, 0.75)); 'off' or
    '' -> (). Weights must be positive; they need not sum to one."""
    if text is None or text.strip() in ("", "off"):
        return ()
    mix = []
    for item in text.split(","):
        base, sep, weight = item.strip().partition(":")
        if not sep:
            raise ValueError(f"base mix entry {item!r} is not MS:WEIGHT")
        base_ms, share = int(base), float(weight)
        if base_ms < 1 or not share > 0:
            raise ValueError(f"base mix entry {item!r}: base must be >= 1 ms and weight > 0")
        mix.append((base_ms, share))
    return tuple(mix)


def format_base_mix(mix):
    return ",".join(f"{base}:{weight:g}" for base, weight in mix)


@dataclass(frozen=True)
class Options:
    durative_tree: bool = False
    sit_gate: bool = True
    flag_loss: bool = False
    instant_transfer: bool = True
    mate_race_by_landing: bool = False
    pass_cap_k: int = 8
    base_mix: tuple = ()
    seat_move_times: bool = False
    team_tree_reuse: bool = False
    team_tree_reuse_tolerance_ms: int = 250
    joint_plans: bool = False
    joint_plan_free_permille: int = 500
    plan_carry: bool = False
    plan_carry_permille: int = 300
    own_pace_below_root: bool = False
    wait_node_acting_eval: bool = False
    blockable_check_not_mate: bool = False

    def __post_init__(self):
        if not 1 <= self.pass_cap_k <= 64:
            raise ValueError("pass_cap_k must be in 1..64")
        if not 0 <= self.team_tree_reuse_tolerance_ms <= 5000:
            raise ValueError("team_tree_reuse_tolerance_ms must be in 0..5000")
        if not 0 <= self.joint_plan_free_permille <= 1000:
            raise ValueError("joint_plan_free_permille must be in 0..1000")
        if not 0 <= self.plan_carry_permille <= 1000:
            raise ValueError("plan_carry_permille must be in 0..1000")
        # A knob without its switch does nothing; the engine refuses it too, so it never
        # silently moves a stock run onto the clock time model.
        defaults = {field.name: field.default for field in fields(self)}
        for switch, knob in (("team_tree_reuse", "team_tree_reuse_tolerance_ms"),
                             ("joint_plans", "joint_plan_free_permille"),
                             ("plan_carry", "plan_carry_permille")):
            if not getattr(self, switch) and getattr(self, knob) != defaults[knob]:
                raise ValueError(f"{knob} needs {switch}")
        if self.plan_carry and not (self.durative_tree and self.base_mix):
            # The engine refuses it too: only seat-clock games play one seat's
            # half and keep the agent until the partner decides.
            raise ValueError("plan_carry needs durative_tree and a base_mix")
        if self.joint_plans and not self.durative_tree:
            # Fix 2 acts inside each durative search; the engine refuses it otherwise.
            raise ValueError("joint_plans needs durative_tree")
        for name in ("own_pace_below_root", "wait_node_acting_eval"):
            if getattr(self, name) and not self.durative_tree:
                # Diagnosis fixes D1 and D3 change the durative tree only; the engine refuses them otherwise.
                raise ValueError(f"{name} needs durative_tree")
        if self.team_tree_reuse and not (self.durative_tree and self.base_mix):
            # The engine refuses it too: it acts on durative seat-clock games only.
            raise ValueError("team_tree_reuse needs durative_tree and a base_mix")
        if self.flag_loss and not self.base_mix:
            # Without seat clocks nothing can flag: the option would do nothing.
            raise ValueError("flag_loss needs a base_mix (real seat clocks)")

    @property
    def is_stock(self):
        return self == STOCK

    def engine_args(self):
        """Command-line options for the engine's selfplay and tournament
        commands: only those that differ from stock, so the stock settings
        give the stock command line and run on an engine without them."""
        args = []
        for field in fields(self):
            value, default = getattr(self, field.name), getattr(STOCK, field.name)
            if value == default:
                continue
            flag = "--" + field.name.replace("_", "-")
            if isinstance(value, bool):
                args += [flag, "true" if value else "false"]
            elif field.name == "base_mix":
                args += [flag, format_base_mix(value)]
            else:
                args += [flag, str(value)]
        return args

    def as_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self)}
        d["base_mix"] = format_base_mix(self.base_mix)
        return d


STOCK = Options()
# MateRaceByLanding stays off: it switches off the engine's root mate provers,
# and with the gate off the tree then passed instead of mating in won positions
# (talisman docs/wave2_local_check.md). --mate-race-by-landing true ablates it.
WAVE2 = Options(durative_tree=True, sit_gate=False, flag_loss=True, instant_transfer=False,
                mate_race_by_landing=False, pass_cap_k=8, base_mix=DEFAULT_BASE_MIX,
                seat_move_times=True)

# The engine flags rl-loop overrides, by Options field.
BOOLEAN_FIELDS = ("durative_tree", "sit_gate", "flag_loss", "instant_transfer", "mate_race_by_landing",
                  "seat_move_times", "team_tree_reuse", "joint_plans", "plan_carry",
                  "own_pace_below_root", "wait_node_acting_eval", "blockable_check_not_mate")
# Wave 2c: what --continuity switches on (an explicit switch still overrides it): the three
# fixes and the diagnosis fixes D1 and D5 (D3 showed no gain and stays out).
CONTINUITY_FIELDS = ("team_tree_reuse", "joint_plans", "plan_carry", "own_pace_below_root",
                     "blockable_check_not_mate")


def from_args(args):
    """The Options an argparse namespace asks for (see add_arguments): the
    wave 2 preset with --wave2, else stock, then each explicit override."""
    base = WAVE2 if getattr(args, "wave2", False) else STOCK
    changes = {}
    if getattr(args, "continuity", False):
        changes.update({name: True for name in CONTINUITY_FIELDS})
    for name in BOOLEAN_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value == "true"
    if getattr(args, "pass_cap_k", None) is not None:
        changes["pass_cap_k"] = args.pass_cap_k
    if getattr(args, "base_mix", None) is not None:
        changes["base_mix"] = parse_base_mix(args.base_mix)
    if getattr(args, "team_tree_reuse_tolerance_ms", None) is not None:
        changes["team_tree_reuse_tolerance_ms"] = args.team_tree_reuse_tolerance_ms
    if getattr(args, "joint_plan_free_permille", None) is not None:
        changes["joint_plan_free_permille"] = args.joint_plan_free_permille
    if getattr(args, "plan_carry_permille", None) is not None:
        changes["plan_carry_permille"] = args.plan_carry_permille
    return replace(base, **changes)


def add_arguments(parser):
    group = parser.add_argument_group(
        "wave 2", "Real seat clocks, the durative tree and no hard sit gate (docs/wave2_training.md). "
                  "Each option overrides the --wave2 preset; without --wave2 they override stock v3.")
    group.add_argument("--wave2", action="store_true",
                       help="Self-play and matches with the wave 2 preset: durative tree, sit gate off, "
                            "flag loss, delayed transfer, pass cap 8, base mix "
                            f"{format_base_mix(DEFAULT_BASE_MIX)}, per-seat move times")
    for name in BOOLEAN_FIELDS:
        group.add_argument("--" + name.replace("_", "-"), choices=("true", "false"), default=None)
    group.add_argument("--pass-cap-k", type=int, default=None)
    group.add_argument("--base-mix", default=None, help="MS:WEIGHT,... or off")
    group.add_argument("--team-tree-reuse-tolerance-ms", type=int, default=None,
                       help="Wave 2c: largest clock difference --team-tree-reuse adopts (default 250)")
    group.add_argument("--joint-plan-free-permille", type=int, default=None,
                       help="Wave 2c: prior share of decide-later under --joint-plans (default 500)")
    group.add_argument("--plan-carry-permille", type=int, default=None,
                       help="Wave 2c: prior weight of the carried move under --plan-carry (default 300)")
    group.add_argument("--continuity", action="store_true",
                       help="Wave 2c: --team-tree-reuse, --joint-plans, --plan-carry, "
                            "--own-pace-below-root and --blockable-check-not-mate true at once "
                            "(an explicit switch overrides it)")


def missing_engine_options(options, engine_help):
    """The options the engine's --help does not list: an engine without the
    wave 2 options would reject them, or (worse) ignore them."""
    return [arg for arg in options.engine_args()[::2] if arg not in engine_help]


def init_lineage(name, models, artifacts, fetch=None, wide=False, widen=None):
    """Start a wave 2 lineage from twin-s-clock iteration 13: the generator
    (MODELS/NAME.onnx), the fixed baseline every iteration is matched against
    (MODELS/NAME-it0.onnx, rl-loop's convention) and the generator's training
    checkpoint (ARTIFACTS/NAME/best.pt). The downloads are SHA-256 checked;
    files already there are kept, so a lineage that has promoted keeps its
    generator. Returns {role: path}.

    wide=True (wave 2b): the generator and its checkpoint are it13 widened
    (hivemind.wide, made first if missing); the fixed baseline stays it13
    itself, so the gate is against the same network as the narrow lineage's."""
    import shutil

    if fetch is None:
        from hivemind.cli.fetch_network import fetch_network as fetch
    from hivemind.network import ARTIFACTS

    onnx = fetch("clock-it13", models)
    checkpoint = fetch("clock-it13-checkpoint", artifacts / "twin-s-clock-it13")
    paths = {"generator": models / f"{name}.onnx", "fixed": models / f"{name}-it0.onnx",
             "checkpoint": artifacts / name / "best.pt"}
    generator, start = onnx, checkpoint
    if wide:
        from hivemind import wide as wide_net

        generator = models / f"{wide_net.WIDE_NAME}.onnx"
        start = artifacts / wide_net.WIDE_NAME / f"{wide_net.WIDE_NAME}.pt"
        if not generator.is_file() or not start.is_file():
            (widen or wide_net.widen)(checkpoint, start.parent, models)
    for role, source in (("generator", generator), ("fixed", onnx), ("checkpoint", start)):
        if not paths[role].is_file():
            paths[role].parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, paths[role])
    return {**paths, "it13_onnx_sha256": ARTIFACTS["clock-it13"].sha256,
            "it13_checkpoint_sha256": ARTIFACTS["clock-it13-checkpoint"].sha256}


def main():
    import argparse
    from pathlib import Path

    from hivemind.paths import MODEL_DIRECTORY, PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Wave 2 helpers")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help=init_lineage.__doc__.split(".")[0])
    init.add_argument("--name", default="twin-s-clock-w2")
    init.add_argument("--models", type=Path, default=MODEL_DIRECTORY)
    init.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "distill")
    init.add_argument("--wide", action="store_true",
                      help="Wave 2b: start from it13 widened (twin-s-clock-wide), fixed baseline it13")
    sub.add_parser("options", help="Print the wave 2 preset's engine options")
    widen = sub.add_parser("widen", help="Wave 2b: widen it13 to twin-s-clock-wide (.pt, .onnx, "
                                         "-fp32.onnx), check parity, record SHA-256s")
    widen.add_argument("--models", type=Path, default=MODEL_DIRECTORY)
    widen.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "distill")
    widen.add_argument("--positions", type=int, default=300)
    widen.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "options":
        print(" ".join(WAVE2.engine_args()))
        return
    if args.command == "widen":
        import json

        from hivemind import wide
        from hivemind.cli.fetch_network import fetch_network

        checkpoint = fetch_network("clock-it13-checkpoint", args.artifacts.resolve() / "twin-s-clock-it13")
        it13 = fetch_network("clock-it13", args.models.resolve())
        # The published FP32 copy is it13's FP16 file converted, as ours is.
        it13_fp32 = it13.with_name(f"{it13.stem}-fp32{it13.suffix}")
        if not it13_fp32.is_file():
            wide._convert_fp32(it13, it13_fp32)
        record = wide.widen(checkpoint, args.artifacts.resolve() / wide.WIDE_NAME, args.models.resolve(),
                            positions=args.positions, seed=args.seed,
                            published={"it13_fp16": (it13, "fp16"), "it13_fp32": (it13_fp32, "fp32")})
        print(json.dumps(record, indent=1))
        return
    for role, value in init_lineage(args.name, args.models.resolve(), args.artifacts.resolve(),
                                    wide=args.wide).items():
        print(f"{role} {value}")


if __name__ == "__main__":
    main()
