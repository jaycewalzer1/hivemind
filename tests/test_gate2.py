"""hivemind.gate2: the real-clock two-score gate's decision (talisman docs/custom_engine_spec_v7_league.md 3.2)."""

import json
import subprocess
import sys

import pytest

from hivemind import gate2, wave2


def write_net(net_dir, arms, sha="ab" * 32):
    """A merged net dir: {arm: {(part, pair_index): score}} as pairs.tsv, with a summary.json and config.json."""
    net_dir.mkdir(parents=True, exist_ok=True)
    (net_dir / "config.json").write_text(json.dumps({"label": net_dir.name, "sha256": sha}))
    for arm, pairs in arms.items():
        (net_dir / arm).mkdir(exist_ok=True)
        rows = ["part\tpair_index\topening_id\tcontender_team\tscore_pair\tgame_ids"]
        rows += [f"{part}\t{index}\t{index}\tT1,T2\t{score}\tp{2 * index + 1},p{2 * index + 2}"
                 for (part, index), score in sorted(pairs.items())]
        (net_dir / arm / "pairs.tsv").write_text("\n".join(rows) + "\n")
        (net_dir / arm / "summary.json").write_text(json.dumps({"score": sum(pairs.values()) / max(1, len(pairs)),
                                                                "pairs": len(pairs)}))
    return net_dir


def table(scores, part=1):
    return {(part, i): s for i, s in enumerate(scores)}


def test_read_pairs(tmp_path):
    net = write_net(tmp_path / "n", {"equal": {(1, 0): 1.0, (2, 3): 0.25}})
    assert gate2.read_pairs(net, "equal") == {(1, 0): 1.0, (2, 3): 0.25}
    assert gate2.read_pairs(net, "free") == {}


def test_equal_networks_promote(tmp_path):
    pairs = table([1.0, 0.5, 0.0, 1.0] * 50)
    cand = write_net(tmp_path / "c", {"equal": pairs, "free": pairs})
    inc = write_net(tmp_path / "i", {"equal": pairs, "free": pairs})
    result = gate2.decide(cand, inc)
    assert result["promoted"] is True
    assert result["arms"]["equal"] == {"pairs": 200, "d": 0.0, "se": 0.0, "drop": False, "complete": True}
    assert result["reason"].startswith("promoted: d sum +0.000")
    assert result["candidate"]["label"] == "c" and result["candidate"]["equal"]["pairs"] == 200


def test_a_clear_drop_in_one_arm_blocks(tmp_path):
    inc_pairs = table([1.0] * 200)
    worse = table([0.0] * 30 + [1.0] * 170)  # d = -0.15, SE about 0.025
    better = table([1.0] * 200)
    cand = write_net(tmp_path / "c", {"equal": worse, "free": better})
    inc = write_net(tmp_path / "i", {"equal": inc_pairs, "free": table([0.0] * 100 + [1.0] * 100)})
    result = gate2.decide(cand, inc)
    assert result["arms"]["equal"]["drop"] is True and result["arms"]["free"]["drop"] is False
    assert result["promoted"] is False and result["reason"].startswith("drop in equal")


def test_a_noisy_small_drop_is_not_a_drop_but_the_sum_rule_still_applies(tmp_path):
    base = [1.0, 0.0] * 100
    inc = table(base)
    # Candidate a little worse in equal: 36 pairs lost, 30 won -> d = -0.03, SE about 0.04: not a drop.
    worse = [0.0 if i % 2 == 0 and i < 72 else 1.0 if i % 2 and i < 60 else s for i, s in enumerate(base)]
    cand = write_net(tmp_path / "c", {"equal": table(worse), "free": inc})
    incd = write_net(tmp_path / "i", {"equal": inc, "free": inc})
    result = gate2.decide(cand, incd)
    equal = result["arms"]["equal"]
    assert equal["d"] == pytest.approx(-0.03) and not equal["drop"]
    assert result["promoted"] is False and result["reason"].startswith("sum below 0")
    # The free arm making up for it promotes (d_free = +0.05).
    better = [1.0 if i % 2 and i < 20 else s for i, s in enumerate(base)]
    cand2 = write_net(tmp_path / "c2", {"equal": table(worse), "free": table(better)})
    assert gate2.decide(cand2, incd)["promoted"] is True


def test_incomplete_arms_do_not_promote(tmp_path):
    pairs = table([1.0] * 200)
    cand = write_net(tmp_path / "c", {"equal": pairs})  # free not merged
    inc = write_net(tmp_path / "i", {"equal": pairs, "free": pairs})
    result = gate2.decide(cand, inc)
    assert result["promoted"] is False
    assert result["reason"] == "incomplete: free 0 of 150 pairs"
    assert result["arms"]["free"] == {"pairs": 0, "d": None, "se": None, "drop": False, "complete": False}
    assert result["candidate"]["free"] is None


def test_only_shared_pairs_count(tmp_path):
    # Candidate: parts 1 and 2; incumbent: parts 2 and 3. Only part 2's pairs are compared.
    cand_pairs = {**table([0.0] * 100, part=1), **table([1.0] * 100, part=2)}
    inc_pairs = {**table([0.5] * 100, part=2), **table([1.0] * 100, part=3)}
    cand = write_net(tmp_path / "c", {"equal": cand_pairs, "free": cand_pairs})
    inc = write_net(tmp_path / "i", {"equal": inc_pairs, "free": inc_pairs})
    result = gate2.decide(cand, inc, min_pairs=100)
    assert result["arms"]["equal"]["pairs"] == 100 and result["arms"]["equal"]["d"] == 0.5
    assert result["promoted"] is True
    assert gate2.decide(cand, inc)["reason"] == "incomplete: equal 100 of 150 pairs, free 100 of 150 pairs"


def test_drop_needs_the_margin(tmp_path):
    inc = table([1.0] * 200)
    cand = table([0.0] * 2 + [1.0] * 198)  # d = -0.01: inside the margin, however small the SE
    result = gate2.compare(cand, inc)
    assert result["d"] == pytest.approx(-0.01) and result["drop"] is False


def test_decide_cli(tmp_path):
    pairs = table([1.0] * 160)
    cand = write_net(tmp_path / "c", {"equal": pairs, "free": pairs})
    inc = write_net(tmp_path / "i", {"equal": pairs, "free": pairs})
    out = subprocess.run([sys.executable, "-m", "hivemind.gate2", "decide", str(cand), str(inc)],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["promoted"] is True


@pytest.mark.parametrize("argv", [[], ["--wave2"], ["--wave2", "--durative-tree", "false", "--sit-gate", "true",
                                                    "--hybrid"]])
def test_wave2_json_round_trip(argv, capsys):
    assert gate2.main(["wave2-json", *argv, "--gate2", "--gate2-games", "400"]) == 0
    out = capsys.readouterr()
    printed = json.loads(out.out)
    parser = __import__("argparse").ArgumentParser()
    wave2.add_arguments(parser)
    assert printed == wave2.from_args(parser.parse_args(argv)).as_dict()
    assert "ignoring --gate2 --gate2-games 400" in out.err


def test_wave2_json_refuses_bad_options():
    with pytest.raises(SystemExit):
        gate2.main(["wave2-json", "--hybrid"])  # hybrid needs a base mix
