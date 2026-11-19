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

    def __post_init__(self):
        if not 1 <= self.pass_cap_k <= 64:
            raise ValueError("pass_cap_k must be in 1..64")
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
WAVE2 = Options(durative_tree=True, sit_gate=False, flag_loss=True, instant_transfer=False,
                mate_race_by_landing=True, pass_cap_k=8, base_mix=DEFAULT_BASE_MIX,
                seat_move_times=True)

# The engine flags rl-loop overrides, by Options field.
BOOLEAN_FIELDS = ("durative_tree", "sit_gate", "flag_loss", "instant_transfer", "mate_race_by_landing",
                  "seat_move_times")


def from_args(args):
    """The Options an argparse namespace asks for (see add_arguments): the
    wave 2 preset with --wave2, else stock, then each explicit override."""
    base = WAVE2 if getattr(args, "wave2", False) else STOCK
    changes = {}
    for name in BOOLEAN_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value == "true"
    if getattr(args, "pass_cap_k", None) is not None:
        changes["pass_cap_k"] = args.pass_cap_k
    if getattr(args, "base_mix", None) is not None:
        changes["base_mix"] = parse_base_mix(args.base_mix)
    return replace(base, **changes)


def add_arguments(parser):
    group = parser.add_argument_group(
        "wave 2", "Real seat clocks, the durative tree and no hard sit gate (docs/wave2_training.md). "
                  "Each option overrides the --wave2 preset; without --wave2 they override stock v3.")
    group.add_argument("--wave2", action="store_true",
                       help="Self-play and matches with the wave 2 preset: durative tree, sit gate off, "
                            "flag loss, delayed transfer, mate races by landing, pass cap 8, base mix "
                            f"{format_base_mix(DEFAULT_BASE_MIX)}, per-seat move times")
    for name in BOOLEAN_FIELDS:
        group.add_argument("--" + name.replace("_", "-"), choices=("true", "false"), default=None)
    group.add_argument("--pass-cap-k", type=int, default=None)
    group.add_argument("--base-mix", default=None, help="MS:WEIGHT,... or off")


def missing_engine_options(options, engine_help):
    """The options the engine's --help does not list: an engine without the
    wave 2 options would reject them, or (worse) ignore them."""
    return [arg for arg in options.engine_args()[::2] if arg not in engine_help]


def init_lineage(name, models, artifacts, fetch=None):
    """Start a wave 2 lineage from twin-s-clock iteration 13: the generator
    (MODELS/NAME.onnx), the fixed baseline every iteration is matched against
    (MODELS/NAME-it0.onnx, rl-loop's convention) and the generator's training
    checkpoint (ARTIFACTS/NAME/best.pt). The downloads are SHA-256 checked;
    files already there are kept, so a lineage that has promoted keeps its
    generator. Returns {role: path}."""
    import shutil

    if fetch is None:
        from hivemind.cli.fetch_network import fetch_network as fetch
    from hivemind.network import ARTIFACTS

    onnx = fetch("clock-it13", models)
    checkpoint = fetch("clock-it13-checkpoint", artifacts / "twin-s-clock-it13")
    paths = {"generator": models / f"{name}.onnx", "fixed": models / f"{name}-it0.onnx",
             "checkpoint": artifacts / name / "best.pt"}
    for role, source in (("generator", onnx), ("fixed", onnx), ("checkpoint", checkpoint)):
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
    sub.add_parser("options", help="Print the wave 2 preset's engine options")
    args = parser.parse_args()
    if args.command == "options":
        print(" ".join(WAVE2.engine_args()))
        return
    for role, value in init_lineage(args.name, args.models.resolve(), args.artifacts.resolve()).items():
        print(f"{role} {value}")


if __name__ == "__main__":
    main()
