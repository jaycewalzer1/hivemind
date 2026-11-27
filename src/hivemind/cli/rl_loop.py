#!/usr/bin/env python3
"""Improve twin-s by self-play, forever. Each iteration N:
  1. plays --games self-play games with the generator, in segments
  2. plays --val-games validation games
  3. trains from iteration N-1's last weights on the new games plus a replay
     buffer of the previous --window iterations
  4. matches the result against the generator; it becomes the generator unless
     it scores below 50%
  5. uploads every iteration's network and training checkpoint, and the
     generator with its checkpoint, to Hugging Face
     (skipped when not logged in to an account that can write to --repo)
  6. matches the result against a fixed baseline: the pre-RL network (it0), or with
     --anchor-after N the generator as it stood after iteration N

Self-play and training use the settings of `hivemind selfplay` and `hivemind
train`. With --time-model clock, self-play and matches run on a simulated clock
(the engine's --time-model clock) and networks read the clock input planes; run
such a lineage with its own --generator, --generator-checkpoint and --runs. Every finished stage leaves a marker in RUNS/rl-itN/stages, so a rerun
resumes at the first unfinished one (an interrupted segment or match is moved
to rl-itN/aborted and redone). Training and matches wait while a game is running
(see --pause-for), and a match restarts if one starts mid-way.

With --distributed, other machines share step 1: the loop publishes a work
order on --exchange, `hivemind contribute` plays games for it on any number of
machines, and the loop collects their batches (and keeps playing its own
segments, unless --no-local-selfplay) until the iteration has --games games.
Batches that arrive later still count, as replay data for later iterations.

Each finished self-play segment moves to the private --archive dataset on
Hugging Face as a zstd tarball (about 13x smaller than the chunks), and is
deleted from disk once the Hub's copy checks out; rl-itN/search/archive.json
lists what went where. Training downloads the games it needs, with the teacher
validation set, and deletes them again once it has finished. --keep-selfplay
keeps everything on disk; --archive-only archives and exits.

With --wave2 (talisman fork), self-play and matches run on four real seat
clocks with the durative tree, flag loss and no hard sit gate (hivemind.wave2;
docs/wave2_training.md), each segment's value labels are checked
(hivemind.distill.labels), training continues the generator's checkpoint with
a slow trunk (--train-from generator, --trunk-lr), and a candidate is
promoted only when it beats the generator (--promote-above 0.5).

With --shared-selfplay (talisman fork), any number of `rl-loop --selfplay-worker`
processes on other GPUs of the same cluster share step 1 through the shared
filesystem: every local segment is claimed first (rl-itN/stages/segNN.claim,
created exclusively, its mtime a heartbeat while the segment plays), so no two
processes play one segment, and a claim whose heartbeat stopped (a job killed at
its time limit) is taken over and its segment resumed. A worker plays segments
of the iteration in self-play until the claimed and finished games reach
--games, then exits; the main loop plays validation while workers finish, then
trains and gates as before. --selfplay-hook runs a shell command at the start of
each iteration's self-play (the cluster uses it to submit the worker jobs).
An iteration plays its self-play the way it began, so a run can switch to or
from --shared-selfplay at any point: an iteration begun shared leaves
stages/selfplay_mode.done ({"shared": true, "segment_games": S}, written
before its generator marker) and is finished shared, with segments of S games,
by any later process, with or without the flag (without it, no hook runs and
the main loop claims what is left alone). An iteration begun without the flag
(no such marker) is finished one segment after another by the main loop, with
the segment size it began with, even under --shared-selfplay: the marker then
records {"shared": false}, no hook runs and workers leave it alone.

With --gate-sprt ELO0,ELO1 (talisman fork) the match against the generator is
sequential: it plays --gate-chunk games at a time and stops as soon as a
generalized SPRT (fishtest's trinomial normal approximation, logistic Elo;
--gate-sprt-alpha, --gate-sprt-beta) accepts H0 (Elo <= ELO0) or H1 (Elo >=
ELO1), or at --gate-max-games (default --match-games; larger extends a close
gate). H1 promotes, H0 does not, and an undecided gate falls back to the score
rule. The engine restarts a match whose --games changed, so each chunk is its
own tournament in arms/match_generator/chunkNN (seed 1000 N + chunk, the
openings rotated so the chunks play them in the order one long match would);
their games and summary are merged into arms/match_generator itself. The match
against the fixed baseline keeps its fixed size, so that curve stays comparable.

With --search-balance (talisman fork; docs/search_balance.md) the search balance
controller (talisman cluster/search_balance.py, --search-balance-controller)
picks each iteration's self-play recipe. --runs is then RUN/runs and the
controller's files live in RUN/balance. Before an iteration's self-play the
loop runs `search_balance.py plan RUN N` once and freezes its segment table
(games, seed and engine flags of each segment) into rl-itN/stages/
balance_plan.done, before the iteration's generator marker, so every
--selfplay-worker reads the same table; the iteration plays exactly the
planned segments (a challenger segment carries its arm's playout-cap flags
and its twin's seed; validation always plays stock). Training counts steps by
the plan's steps_rule and passes its trainer_args (--search-q-blend). After
training: `hivemind balance probe` (when the plan refreshes the probe),
`minitrain` and `score` (when it has a challenger); after the gate: the
mini-gate (when `decide --dry` says it can promote), `balance accounting`,
`search_balance.py realise` and `decide` (RUN/balance/itN.json). Each step
leaves a stages/balance_*.done marker; a failed balance step is logged and
skipped, never fatal (the controller then plans failsafe). Without a prior
(RUN/balance/prior.json or --search-balance-prior) the controller holds
today's recipe and plans no challenger, so self-play and training run exactly
as without the flag.

With --gate2 (talisman fork; talisman docs/custom_engine_spec_v7_league.md section 3)
the candidate is promoted by the real-clock two-score gate instead of the 100 ms
match (which still plays, unchanged): after match_generator, the stage gate2
runs talisman cluster/gate2_match.sh (--gate2-script) every --gate2-poll-s for
the candidate and the incumbent; it submits each network's arena matches against
upstream-it13 to PACE (arms equal and free, cached per network in
RUN/gate2/nets/<sha16>, RUN = the parent of --runs) and merges them. Once both
networks are done, hivemind.gate2 decides (promote iff both arms are complete
and neither drops, and d_equal + d_free >= 0); not done by --gate2-timeout-h
(from stages/gate2.started, so a resumed job keeps the clock) is not promoted.
A failing script is logged and called again at the next poll, never fatal.

With --league/--pace (hivemind.league) and --think-* (hivemind.think) the loop
calls those modules' hooks (spec v7 3.5): a league plan frozen in
stages/league_plan.done like the search balance plan, the plan's and the think
policy's engine flags on every self-play segment but validation, a
think_labels stage before training (non-fatal) and the think trainer flags.
The modules are imported only when their flags are given (or for --help).

Stop between stages with: touch RUNS/rl-loop.stop"""

import argparse
import contextlib
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

from hivemind import gate2 as two_score
from hivemind import wave2
from hivemind.cli import selfplay, train
from hivemind.constants import NUM_WIDE_CHANNELS
from hivemind.data import archive
from hivemind.distill.data import chunk_problem
from hivemind.distill import timing
from hivemind.distill.labels import label_report
from hivemind.distill.exchange import PROTOCOL, HubExchange, install, open_exchange, sha256
from hivemind.network import SELFPLAY_ARCHIVE, SELFPLAY_EXCHANGE, TWIN_ONNX_PATH, TWIN_REPOSITORY
from hivemind.paths import PROJECT_ROOT

RUNS = PROJECT_ROOT / "data" / "distill" / "runs"
# The 100 positions every RL match since iteration 2 opened from.
OPENINGS = PROJECT_ROOT / "engine" / "openings.tsv"
ENGINE_DIR = PROJECT_ROOT / "engine"
MATCH_ENGINE = ENGINE_DIR / "build-ninja" / "hivemind"
QUIET_SECONDS = 300
POLL_SECONDS = 20
COLLECT_SECONDS = 60
RETRY_SECONDS = 60
ATTEMPTS = 3
# --shared-selfplay: a claim's mtime is touched this often while its segment
# plays, and taken over once it is this old.
HEARTBEAT_SECONDS = 60
STALE_CLAIM_SECONDS = 15 * 60


class StageFailed(Exception):
    pass


def next_iteration(runs):
    """The first iteration without stages/iteration.done (the pilot, rl-it1,
    has no stages and counts as finished once it has trained). An rl-it1 with
    stages is a staged iteration like any other: its arms/C/last.pt appears at
    training, before the matches, so a job cut off in the matches must resume it."""
    n = 1
    while ((runs / f"rl-it{n}" / "stages" / "iteration.done").is_file()
           or (n == 1 and not (runs / "rl-it1" / "stages").exists()
               and (runs / "rl-it1" / "arms" / "C" / "last.pt").is_file())):
        n += 1
    return n


def train_dirs(runs, n):
    """Self-play training data directories of iteration n, oldest layout first
    (the pilot wrote its chunks straight into search/train)."""
    it = runs / f"rl-it{n}"
    data = it / "search" / "train"
    dirs = [data] if any(data.glob("*.dst")) else []
    return dirs + [data / marker.stem for marker in sorted((it / "stages").glob("seg*.done"))]


def replay_iterations(n, window):
    return list(range(n - 1, max(1, n - window) - 1, -1))


def engine_help(engine):
    try:
        result = subprocess.run([str(engine), "--help"], cwd=ENGINE_DIR, capture_output=True,
                                text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout + result.stderr


def engine_supports(engine, text):
    """Whether the engine's --help mentions `text`: older builds lack options
    the loop now passes when it can."""
    return text in engine_help(engine)


def checkpoint_games(progress):
    """The game count an interrupted self-play run was started with."""
    for line in progress.read_text().splitlines():
        key, _, value = line.partition(" ")
        if key == "games":
            return int(value)
    raise StageFailed(f"{progress} has no game count")


def match_summary(path, games):
    d = json.loads(Path(path).read_text())
    wins, losses, draws = d["contender_wins"], d["baseline_wins"], d["draws"]
    if not (all(type(k) is int and k >= 0 for k in (wins, losses, draws))
            and wins + losses + draws == games):
        raise StageFailed(f"incomplete match in {path}")
    elo = d["contender_elo"]
    if elo is not None and not math.isfinite(elo):
        raise StageFailed(f"invalid Elo in {path}")
    return dict(wins=wins, losses=losses, draws=draws, score=(wins + draws / 2) / games, elo=elo,
                ci=d["elo_confidence_95"],
                nps=[d["performance"]["contender"]["nps"], d["performance"]["baseline"]["nps"]])


def archive_iteration(path):
    return int(path.stem.rsplit("-it", 1)[1])


def iteration_checkpoints(runs):
    """(n, last.pt) of every iteration whose training finished."""
    found = []
    for it in runs.glob("rl-it*"):
        if not re.fullmatch(r"rl-it\d+", it.name):
            continue  # an aborted run moved aside, such as rl-it2-unsegmented-aborted
        n, last = int(it.name[len("rl-it"):]), it / "arms" / "C" / "last.pt"
        trained = (it / "stages" / "train.done").is_file() or (n == 1 and not (it / "stages").exists())
        if trained and last.is_file():
            found.append((n, last))
    return sorted(found)


def sync_hub(api, repo, generator, checkpoint, log, runs):
    """Upload, in one commit, what the Hub lacks: every iteration's network
    (NAME-itN.onnx) and training checkpoint (NAME-itN.pt), and the generator
    (NAME.onnx) and its checkpoint (NAME.pt) when they differ from the Hub's."""
    from huggingface_hub import CommitOperationAdd

    stem = generator.stem
    present = set(api.list_repo_files(repo))
    archives = sorted(generator.parent.glob(f"{stem}-it*.onnx"), key=archive_iteration)
    uploads = [(path, path.name) for path in archives if path.name not in present]
    uploads += [(path, f"{stem}-it{n}.pt") for n, path in iteration_checkpoints(runs)
                if f"{stem}-it{n}.pt" not in present]
    remote = {info.path: info.lfs.sha256 for info in api.get_paths_info(repo, [generator.name, f"{stem}.pt"])
              if info.lfs}
    local = sha256(generator)
    promoted = local != remote.get(generator.name)
    if promoted:
        uploads.append((generator, generator.name))
    if checkpoint.is_file() and sha256(checkpoint) != remote.get(f"{stem}.pt"):
        uploads.append((checkpoint, f"{stem}.pt"))
    if not uploads:
        return
    names = [name for _, name in uploads]
    message = f"Upload {', '.join(names)}"
    n = next((archive_iteration(p) for p in archives if sha256(p) == local), None)
    if promoted and n is not None:
        message = f"Upload {stem} from RL iteration {n}"
        marker = runs / f"rl-it{n}" / "stages" / "match_generator.done"
        if marker.is_file():
            m = json.loads(marker.read_text())
            elo = "" if m["elo"] is None else f" ({m['elo']:+.0f} Elo)"
            at = f"{m.get('movetime', 100)} ms" + (", clock" if m.get("time_model") == "clock" else "")
            message += f": {m['wins']}-{m['losses']}-{m['draws']} vs the previous generator at {at}{elo}"
    commit = api.create_commit(
        repo_id=repo, commit_message=message,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for path, name in uploads])
    log(f"uploaded {', '.join(names)} to {repo}: {message} ({commit.oid})")


def upload_blocker(api, repo, token):
    """Why this machine cannot upload to `repo`, or None if it can."""
    if token is None:
        return "not logged in to Hugging Face (run `hf auth login` to upload)"
    try:
        account = api.whoami()
    except Exception:
        return None  # offline for now: each upload retries
    owners = {account["name"], *(org["name"] for org in account.get("orgs", []))}
    if repo.split("/")[0] not in owners:
        return f"Hugging Face account {account['name']} cannot upload to {repo}"
    return None


def finished_selfplay(it):
    """The paths under rl-itN/search that hold finished self-play: the games,
    records and log of every finished segment and collected batch, and the
    validation games once played. The pilot, rl-it1, left no stage markers and
    is finished as a whole. An interrupted segment's files stay for its rerun
    to move aside."""
    stages = it / "stages"
    if not stages.is_dir():
        return None
    names = sorted(marker.stem for marker in stages.glob("seg*.done"))
    paths = [path for name in names for path in (f"train/{name}", f"train_run/{name}", f"logs/{name}.log")]
    if (stages / "val.done").is_file():
        paths += ["val", "val_run", "logs/val.log"]
    return paths


# Settings added for wave 2, at their upstream behaviour: what the loop uses
# when its arguments do not name them.
WAVE2_DEFAULTS = {"wave2_options": wave2.STOCK, "label_check": False, "no_gate": False,
                  "train_from": "previous", "lr": "1e-4", "trunk_lr": None, "promote_above": None,
                  "wide": False, "arrivals_weight": None, "shared_selfplay": False, "selfplay_hook": None,
                  "gate_sprt": None, "gate_sprt_alpha": 0.05, "gate_sprt_beta": 0.05, "gate_chunk": 100,
                  "gate_max_games": None, "search_balance": False, "search_balance_controller": None,
                  "search_balance_prior": None, "search_balance_mini_steps": 200, "search_balance_probe_games": 150,
                  "search_balance_minigate_games": 100, "search_balance_gpus": 1.0, "search_balance_device": "auto"}
# Spec v7 (league, think policy, real-clock gate), kept apart so WAVE2_DEFAULTS stays comparable with earlier
# revisions; Loop.option reads both.
V7_DEFAULTS = {"gate2": False, "gate2_script": None, "gate2_games": 400, "gate2_parts": 4, "gate2_base_ms": 60000,
               "gate2_free_k": 160, "gate2_equal_k": 110, "gate2_seed_base": 7000, "gate2_opp_sit": "observed",
               "gate2_drop_margin": two_score.DROP_MARGIN, "gate2_min_pairs": two_score.MIN_PAIRS,
               "gate2_poll_s": 300.0, "gate2_timeout_h": 10.0, "league_on": False, "think_on": False}
GATE2_ARMS = "equal free"


# --search-balance: engine flags of the playout cap and the state dump, which validation never gets.
CAP_FLAGS = ("--deep-fraction", "--deep-multiplier", "--fast-nodes", "--fast-policy-mask", "--fast-dirichlet",
             "--deep-dirichlet", "--cap-seed-salt", "--record-state-output")
STEPS_RULES = ("positions", "policy_positions")  # functions of the same name in hivemind.cli.train
# Spec v7: league, pace and think policy flags, which validation never gets either (its losses stay comparable).
VAL_STOCK_FLAGS = ("--pace-k-min", "--opponent-model", "--think-policy", "--record-state-output")


def hook(name):
    """hivemind.league (WP3) or hivemind.think (WP4), imported on first use so a missing module only matters when
    its flags are on."""
    return importlib.import_module(f"hivemind.{name}")


def gate2_script(explicit=None):
    """--gate2: the matches script, `explicit` (None if it does not exist), else $HIVEMIND_GATE2_SCRIPT, else
    cluster/gate2_match.sh in the first directory above the fork that has one (as search_balance_controller)."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    if os.environ.get("HIVEMIND_GATE2_SCRIPT"):
        path = Path(os.environ["HIVEMIND_GATE2_SCRIPT"]).expanduser().resolve()
        return path if path.is_file() else None
    return next((parent / "cluster" / "gate2_match.sh" for parent in PROJECT_ROOT.parents
                 if (parent / "cluster" / "gate2_match.sh").is_file()), None)


def search_balance_controller(explicit=None):
    """The controller script: `explicit` (None if it does not exist), else
    $HIVEMIND_SEARCH_BALANCE, else cluster/search_balance.py in the first
    directory above the fork that has one (the talisman repo the fork sits in
    as third_party/wave2-hivemind)."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    if os.environ.get("HIVEMIND_SEARCH_BALANCE"):
        path = Path(os.environ["HIVEMIND_SEARCH_BALANCE"]).expanduser().resolve()
        return path if path.is_file() else None
    return next((parent / "cluster" / "search_balance.py" for parent in PROJECT_ROOT.parents
                 if (parent / "cluster" / "search_balance.py").is_file()), None)


def last_json_line(text):
    """The last line of `text` that parses as a JSON object, or None."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


# --gate-sprt: the sequential gate against the generator.
def expected_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def score_elo(score):
    return None if not 0 < score < 1 else 400 * math.log10(score / (1 - score))


def sprt_bounds(alpha, beta):
    """Wald's bounds on the log-likelihood ratio: below the first accepts H0,
    above the second H1 (-2.94 and 2.94 at fishtest's alpha = beta = 0.05)."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def sprt_llr(wins, losses, draws, elo0, elo1):
    """The GSPRT log-likelihood ratio of H1 (Elo elo1) against H0 (Elo elo0),
    as fishtest's LLRcalc.LLR_logistic computes it on a trinomial: the score's
    mean and variance per game from the counts, then N (s1 - s0)(2 mean - s0 -
    s1) / (2 var). Logistic Elo, the Elo the matches report, rather than
    BayesElo, whose draw Elo needs draws and bughouse games are seldom drawn. A
    zero count is taken as 1e-3, as fishtest regularizes, so a one-sided start
    still moves the ratio."""
    if wins + losses + draws == 0:
        return 0.0
    counts = [max(count, 1e-3) for count in (wins, draws, losses)]
    n = sum(counts)
    w, d = counts[0] / n, counts[1] / n
    mean = w + d / 2
    var = w + d / 4 - mean ** 2
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


# The trinomial's variance is meaningless on a handful of games (2-0 gives an LLR
# past 30), so no bound counts before this many games or the cap.
SPRT_MIN_GAMES = 100


def sprt_decision(llr, bounds, games=SPRT_MIN_GAMES, cap=SPRT_MIN_GAMES):
    if games < min(SPRT_MIN_GAMES, cap):
        return None
    return "H0" if llr <= bounds[0] else "H1" if llr >= bounds[1] else None


def rotated_openings(path, skip):
    """The opening lines of `path` starting at line `skip` (mod their count):
    the engine opens pair p with line p, so a chunk after `skip` pairs plays the
    openings one long match would."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    skip %= len(lines)
    return "\n".join(lines[skip:] + lines[:skip]) + "\n"


def pair_confidence(outcomes):
    """95% score interval over colour-swapped pairs, as the engine reports it
    (paired-opening normal approximation)."""
    pairs = [(outcomes[i] + outcomes[i + 1] + 2) / 4 for i in range(0, len(outcomes) - 1, 2)]
    if len(pairs) < 2:
        return None
    mean = sum(pairs) / len(pairs)
    margin = 1.959963984540054 * math.sqrt(sum((p - mean) ** 2 for p in pairs) / (len(pairs) - 1) / len(pairs))
    return [max(0.0, mean - margin), min(1.0, mean + margin)]


def add_counts(total, part):
    """Sum the integer leaves of two engine summary dicts (breakdowns, terminations)."""
    for key, value in part.items():
        if isinstance(value, dict):
            add_counts(total.setdefault(key, {}), value)
        elif isinstance(value, int) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    return total


def merge_chunks(out, chunks, sprt):
    """Write out/summary.json, games.tsv and games.pgn for the chunked match
    `chunks` [(directory, seed, games)], in the engine's format where the
    readers of a match look (match_summary, training curves): counts summed,
    games renumbered, score, Elo and the paired interval recomputed over all."""
    summaries = [json.loads((directory / "summary.json").read_text()) for directory, _, _ in chunks]
    rows, outcomes, pgn = [], [], b""
    perf = {"contender": [0, 0, 0], "baseline": [0, 0, 0]}  # searches, nodes, nanoseconds
    for directory, _, _ in chunks:
        lines = (directory / "games.tsv").read_text().splitlines()[1:]
        for line in lines:
            fields = line.split("\t")
            outcomes.append(int(fields[1]))
            for side, at in (("contender", 5), ("baseline", 8)):
                for i in range(3):
                    perf[side][i] += int(fields[at + i])
            fields[0] = str(len(rows))
            fields[11] = str(len(pgn) + int(fields[11]))
            rows.append("\t".join(fields))
        if (directory / "games.pgn").is_file():
            pgn += (directory / "games.pgn").read_bytes()
    wins, losses, draws = (sum(s[key] for s in summaries) for key in ("contender_wins", "baseline_wins", "draws"))
    games = wins + losses + draws
    if games != len(outcomes):
        raise StageFailed(f"{out}: chunk summaries count {games} games, their games.tsv {len(outcomes)}")
    score = (wins + draws / 2) / games
    interval = pair_confidence(outcomes)
    elo_interval = (None if interval is None or interval[0] <= 0 or interval[1] >= 1
                    else [score_elo(interval[0]), score_elo(interval[1])])
    first = summaries[0]
    merged = {key: first[key] for key in first
              if key not in ("games", "seed", "positions_file", "contender_wins", "baseline_wins", "draws",
                             "contender_score", "contender_elo", "confidence_method", "score_confidence_95",
                             "elo_confidence_95", "sprt", "performance", "contender_breakdown", "terminations")}
    merged.update({
        "games": games, "seeds": [seed for _, seed, _ in chunks],
        "chunks": [{"directory": directory.name, "seed": seed, "games": n} for directory, seed, n in chunks],
        "contender_wins": wins, "baseline_wins": losses, "draws": draws, "contender_score": score,
        "contender_elo": score_elo(score), "confidence_method": "paired-opening normal approximation",
        "score_confidence_95": interval, "elo_confidence_95": elo_interval, "sprt": sprt,
        "performance": {side: {"searches": s, "nodes": nodes, "nps": 1e9 * nodes / ns if ns else 0.0}
                        for side, (s, nodes, ns) in perf.items()},
        "contender_breakdown": {}, "terminations": {}})
    for summary in summaries:
        add_counts(merged["contender_breakdown"], summary.get("contender_breakdown", {}))
        add_counts(merged["terminations"], summary.get("terminations", {}))
    (out / "games.pgn").write_bytes(pgn)
    seeds = " ".join(f"{directory.name}:{seed}" for directory, seed, _ in chunks)
    (out / "games.tsv.tmp").write_text("\n".join([f"config\tsequential gate, chunks {seeds}", *rows]) + "\n")
    (out / "games.tsv.tmp").replace(out / "games.tsv")
    (out / "summary.json.tmp").write_text(json.dumps(merged, indent=2) + "\n")
    (out / "summary.json.tmp").replace(out / "summary.json")


class Claims:
    """--shared-selfplay: which process plays which local segment of an
    iteration. A claim is rl-itN/stages/segNN.claim, created with O_EXCL (atomic
    on the cluster's shared filesystem) and holding the segment's game count; its
    mtime is the owner's heartbeat. A finished segment's .done marker outlives
    its claim, which the owner then removes."""

    def __init__(self, stages, owner, stale_seconds=STALE_CLAIM_SECONDS):
        self.stages, self.owner, self.stale_seconds = stages, owner, stale_seconds

    def path(self, k):
        return self.stages / f"seg{k:02d}.claim"

    def live(self):
        """{k: games} of the claims whose heartbeat is fresh and whose segment is not finished."""
        found, now = {}, time.time()
        for claim in self.stages.glob("seg*.claim"):
            k = claim.stem[3:]
            if not k.isdigit() or (self.stages / f"seg{k}.done").is_file():
                continue
            try:
                if now - claim.stat().st_mtime < self.stale_seconds:
                    found[int(k)] = json.loads(claim.read_text())["games"]
            except (OSError, ValueError, KeyError):  # removed or half written: a claim being taken over
                continue
        return found

    def take(self, k, games, balance=None):
        """Claim segment k for `games` games: True if it is now ours. A stale
        claim is renamed aside first; of two processes taking it over, only the
        one whose rename succeeds goes on to create the new claim. `balance`
        (--search-balance) records the segment's arm and engine flags in it."""
        claim = self.path(k)
        try:
            if time.time() - claim.stat().st_mtime >= self.stale_seconds:
                aside = claim.with_name(f"{claim.name}.stale-{self.owner}-{time.time_ns()}")
                claim.rename(aside)
                if time.time() - aside.stat().st_mtime < self.stale_seconds:
                    # Another process took it over between our stat and our rename: give its claim
                    # back (link fails if a third one claimed meanwhile) and leave the segment to it.
                    try:
                        os.link(aside, claim)
                    except FileExistsError:
                        pass
                    aside.unlink()
                    return False
        except FileNotFoundError:
            pass
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps({"games": games, "owner": self.owner, "at": time.time(),
                                     **({"balance": balance} if balance is not None else {})}) + "\n")
        return True

    def release(self, k):
        try:
            self.path(k).unlink()
        except FileNotFoundError:
            pass

    def beat(self, k):
        try:
            os.utime(self.path(k))
        except FileNotFoundError:
            pass


class Loop:
    def __init__(self, args, generator=TWIN_ONNX_PATH, checkpoint=train.GENERATOR_CHECKPOINT,
                 teacher_val=train.TEACHER_VAL):
        self.args = args
        self.runs = args.runs
        self.generator = generator
        self.generator_sp1 = selfplay.one_worker_copy(generator)
        self.checkpoint = checkpoint
        # The fixed baseline: it0, or with --anchor-after N the best network
        # after iteration N, matched against from iteration N + 1 on.
        anchor = args.anchor_after
        self.fixed_name = "it0" if anchor is None else f"it{anchor}"
        self.fixed = generator.with_name(f"{generator.stem}-{'it0' if anchor is None else f'anchor-it{anchor}'}"
                                         f"{generator.suffix}")
        self.fixed_time_model = args.fixed_time_model if anchor is None else args.time_model
        self.teacher_val = [teacher_val] if teacher_val.is_dir() else []
        # Interrupted self-play segments and matches pick up where they
        # stopped, on engines that can.
        self.checkpoint_selfplay = (args.checkpoint_games > 0
                                    and engine_supports(args.selfplay_engine, "--checkpoint-games"))
        self.resume_matches = engine_supports(args.match_engine, "games.tsv")
        self.api = None
        self.exchange = None  # set for --distributed
        # League segments: this process's copies of opponent networks (a worker uses its own directory).
        self.opponent_dir = self.runs / "league_opponents"

    def option(self, name):
        return getattr(self.args, name, WAVE2_DEFAULTS[name] if name in WAVE2_DEFAULTS else V7_DEFAULTS[name])

    def log(self, message):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)
        with (self.runs / "rl-loop.log").open("a") as stream:
            stream.write(line + "\n")

    def check_stop(self):
        if (self.runs / "rl-loop.stop").exists():
            self.log(f"stop requested ({self.runs / 'rl-loop.stop'}): exiting before the next stage")
            sys.exit(0)

    def gaming(self):
        return bool(self.args.pause_for) and subprocess.run(
            ["pgrep", "-x", self.args.pause_for], stdout=subprocess.DEVNULL).returncode == 0

    def wait_for_quiet(self):
        if not self.gaming():
            return
        self.log(f"waiting: a game is running ({self.args.pause_for})")
        quiet = 0
        while quiet < QUIET_SECONDS:
            quiet = 0 if self.gaming() else quiet + POLL_SECONDS
            time.sleep(POLL_SECONDS)
        self.log(f"no game for {QUIET_SECONDS}s: continuing")

    # Stage bookkeeping for the current iteration.
    def done(self, stage):
        return (self.it / "stages" / f"{stage}.done").is_file()

    def mark(self, stage, value):
        temporary = self.it / "stages" / f"{stage}.tmp"
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(self.it / "stages" / f"{stage}.done")

    def aside(self, *paths):
        """Move leftovers of an unfinished stage out of the way."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in paths:
            if path.exists():
                (self.it / "aborted").mkdir(exist_ok=True)
                rel = "_".join(path.relative_to(self.it).parts)
                path.rename(self.it / "aborted" / f"{rel}-{stamp}")

    def attempt(self, stage, *args):
        """Retry a stage (a crash, an out-of-memory error) before giving up."""
        for count in range(1, ATTEMPTS + 1):
            try:
                return stage(*args)
            except Exception as error:
                self.log(f"it{self.n}: {stage.__name__} failed: {error} (attempt {count} of {ATTEMPTS})")
                if count < ATTEMPTS:
                    time.sleep(RETRY_SECONDS)
        sys.exit(1)

    def selfplay(self, name, games, seed, data, records):
        if self.done(name):
            return
        self.check_stop()
        generator = (self.it / "stages" / "generator").read_text().strip()
        selfplay.refresh_copy(self.generator, self.generator_sp1)
        if sha256(self.generator_sp1) != generator:
            raise StageFailed(f"the generator changed during it{self.n}")
        log = self.it / "search" / "logs" / f"{name}.log"
        progress = records / "checkpoint" / "progress"
        resuming = self.checkpoint_selfplay and progress.is_file()
        if resuming:
            # The count it started with, whatever contributors sent since.
            games = checkpoint_games(progress)
        else:
            self.aside(data, records, log)
        log.parent.mkdir(parents=True, exist_ok=True)
        # --search-balance or --league: the plan's segment (None for validation and without a plan).
        planned = self.planned_segments()
        spec = None if planned is None else planned.get(name)
        league = planned is not None and self.balance_segments() is None
        cap = [] if spec is None else self.segment_args(spec, records)
        # --think-*: the think policy's engine flags (and the state dump for labels), never for validation.
        think = ([str(a).replace("{segment_dir}", str(records)) for a in hook("think").selfplay_args(self.args)]
                 if self.option("think_on") and name != "val" else [])
        self.log(f"it{self.n} self-play {name}: {'resuming ' if resuming else ''}{games} games, "
                 f"seed {seed}" + (f" ({self.owner})" if self.option("shared_selfplay") else "")
                 + ("" if spec is None else f", {'league' if league else 'search balance arm'} {spec['theta']}"
                    + (f" {' '.join(cap)}" if cap else " (stock)"))
                 + (f", think {' '.join(think)}" if think else ""))
        extra = list(selfplay.time_model_options(self.args.time_model))
        # Validation keeps the settings every earlier iteration's was played
        # with, so its losses stay comparable across iterations.
        if name != "val":
            if self.args.resign_threshold < 0:
                extra += ["--resign-threshold", str(self.args.resign_threshold)]
            if self.args.tree_reuse:
                extra += ["--tree-reuse", "true"]
        if self.checkpoint_selfplay:
            extra += ["--checkpoint-games", str(self.args.checkpoint_games), "--resume", "true"]
        # The wave 2 rules hold for validation too: its games are this iteration's game.
        extra += self.option("wave2_options").engine_args()
        extra += cap
        extra += think
        command = selfplay.engine_command(self.args.selfplay_engine, self.generator_sp1, games, seed,
                                          data, records, self.args.parallel_games, extra)
        if planned is not None:
            if spec is None and any(flag in command for flag in CAP_FLAGS):
                raise StageFailed(f"{name} must play stock under search balance: {command}")
            if cap and not league and not engine_supports(self.args.selfplay_engine, "--deep-fraction"):
                raise StageFailed(f"{name}: {self.args.selfplay_engine} lacks the playout-cap flags {cap}")
        if name == "val" and any(flag in command for flag in VAL_STOCK_FLAGS):
            raise StageFailed(f"validation must play without league, pace or think flags: {command}")
        started = time.monotonic()
        with log.open("a" if resuming else "w") as stream, self.heartbeat():
            if subprocess.run(command, cwd=ENGINE_DIR, stdout=stream, stderr=subprocess.STDOUT).returncode:
                raise StageFailed(f"self-play {name} failed (see {log})")
        wall = time.monotonic() - started
        lines = log.read_text(errors="replace").splitlines()
        if self.checkpoint_selfplay:
            # A resumed run prints game lines only for the games it played.
            if f"selfplay complete {games} games" not in lines:
                raise StageFailed(f"{name}: no completion line in {log}")
        elif (count := sum(line.startswith("selfplay game ") for line in lines)) != games:
            raise StageFailed(f"{name}: expected {games} games, got {count}")
        positions = train.positions([data])
        if not positions:
            raise StageFailed(f"{name}: empty corpus")
        marker = {"games": games, "seed": seed, "positions": positions, "generator": generator}
        if not self.option("wave2_options").is_stock:
            marker["wave2"] = self.option("wave2_options").as_dict()
        if self.option("label_check"):
            labels = label_report(log, data)
            self.log(f"it{self.n} {name} labels: {labels['games']} games, {labels['records']} records, "
                     f"{labels['flag_games']} flag games, {labels['repeat_records']} repeated positions, "
                     f"terminations {json.dumps(labels['terminations'])}")
            if labels["problems"]:
                raise StageFailed(f"{name}: value labels unfit to train on: {'; '.join(labels['problems'][:5])}")
            marker["labels"] = {k: v for k, v in labels.items() if k != "problems"}
        seat_rows = records / timing.SEAT_ROWS
        if self.option("wide") and seat_rows.is_file():
            # Wave 2b: the wide planes' timing and the arrivals labels go into
            # the chunks (v3), so training reads them with the positions.
            try:
                attached = timing.attach_timing(log, seat_rows, data)
            except ValueError as error:
                raise StageFailed(f"{name}: seat timing does not fit the chunks: {error}")
            self.log(f"it{self.n} {name} timing: {attached['rewritten']} of {attached['chunks']} chunks "
                     f"to v3, {attached['labelled']} of {attached['records']} records with arrivals labels")
            marker["timing"] = attached
        if league:
            if spec is not None and not self.league_failsafe():
                try:
                    marker["league"] = hook("league").segment_summary(log, spec)
                except Exception as error:  # a summary is telemetry: the segment's games still count
                    self.log(f"it{self.n} {name}: league segment summary failed: {error!r}")
                    marker["league"] = {"theta": spec["theta"], "error": repr(error)}
        elif planned is not None:
            if spec is not None:
                marker["balance"] = {"theta": spec["theta"], "engine_args": cap,
                                     **({"twin": spec["twin"]} if spec.get("twin") else {}),
                                     "plan_sha256": (self.balance_plan_marker() or {}).get("plan_sha256")}
            # The engine's own time, for the controller's realised cost (balance accounting). A resumed
            # segment's earlier part is unknown: no number rather than a partial one.
            if not resuming:
                gpus = (self.balance_plan_marker() or {}).get("gpus_per_job", 1.0)
                marker["wall_s"], marker["gpu_s"] = round(wall, 3), round(wall * gpus, 3)
        self.mark(name, marker)
        self.archive(self.n)

    def games_played(self):
        return sum(json.loads(marker.read_text())["games"] for marker in (self.it / "stages").glob("seg*.done"))

    # --search-balance: the iteration's plan, frozen in stages/balance_plan.done.
    def balance_plan_marker(self):
        try:
            return json.loads((self.it / "stages" / "balance_plan.done").read_text())
        except (OSError, ValueError):
            return None

    def balance_segments(self):
        return (self.balance_plan_marker() or {}).get("segments") or None

    def planned_segments(self):
        """{segNN: {"theta", "seed", "games", "engine_args"}} when the iteration
        has a search balance plan (workers read it whatever their own flags),
        else the league plan's table (--league/--pace, the same rules), else
        None: self-play then plays --games in --segment-games segments."""
        return self.balance_segments() or (self.league_plan_marker() or {}).get("segments") or None

    # --league/--pace: the iteration's league plan, frozen in stages/league_plan.done.
    def league_plan_marker(self):
        try:
            return json.loads((self.it / "stages" / "league_plan.done").read_text())
        except (OSError, ValueError):
            return None

    def league_failsafe(self):
        return "failsafe" in (self.league_plan_marker() or {})

    def segment_args(self, spec, records):
        """A planned segment's engine flags: {segment_dir} is the segment's records directory, {opponent_model}
        this process's copy of the segment's opponent network."""
        args = [str(a).replace("{segment_dir}", str(records)) for a in spec["engine_args"]]
        if any("{opponent_model}" in a for a in args):
            copy = self.opponent_copy(spec.get("opponent_model"))
            args = [a.replace("{opponent_model}", str(copy)) for a in args]
        return args

    def opponent_copy(self, source):
        """A league opponent's network as this process's own file (TensorRT caches are per file), named by its
        sha256 so two opponents never share a cache."""
        path = Path(source) if source else None
        if path is None or not path.is_file():
            raise StageFailed(f"league opponent network {source} not found")
        copy = selfplay.one_worker_copy(self.opponent_dir / f"opponent-{sha256(path)[:16]}{path.suffix}")
        copy.parent.mkdir(parents=True, exist_ok=True)
        selfplay.refresh_copy(path, copy)
        return copy

    def league_plan(self):
        """`league.plan(...)` frozen in stages/league_plan.done (before the iteration's generator marker, so a
        worker that sees the iteration sees the plan). If the plan fails, a failsafe table: plain segments, no
        league or pace flags."""
        if self.done("league_plan"):
            return
        args, n = self.args, self.n
        try:
            plan = hook("league").plan(args, self.runs, n, self.generator, self.fixed, args.segment_games)
            table = {}
            for name, s in sorted(plan["segments"].items()):
                if not re.fullmatch(r"seg\d{2}", name):
                    raise ValueError(f"segment name {name!r}")
                engine_args = [str(a) for a in s.get("engine_args") or []]
                opponent = s.get("opponent_model")
                if any("{opponent_model}" in a for a in engine_args) and not opponent:
                    raise ValueError(f"{name} plays {{opponent_model}} without an opponent_model")
                table[name] = {"theta": str(s["theta"]), "seed": int(s["seed"]), "games": int(s["games"]),
                               "engine_args": engine_args,
                               "opponent_model": None if opponent is None else str(opponent),
                               **({"league": s["league"]} if s.get("league") else {})}
            if not table:
                raise ValueError("no segments")
            marker = {"segments": table, "summary": plan.get("summary")}
            self.log(f"it{n} league plan: {plan.get('summary') or ''}; " + ", ".join(
                f"{name} {s['theta']} {s['games']} games" for name, s in table.items()))
        except Exception as error:
            segments = -(-args.games // args.segment_games)
            table = {f"seg{k:02d}": {"theta": "self", "seed": 1000 * n + k, "engine_args": [], "opponent_model": None,
                                     "games": min(args.segment_games, args.games - k * args.segment_games)}
                     for k in range(segments)}
            marker = {"failsafe": f"{type(error).__name__}: {error}", "segments": table}
            self.log(f"it{n} league plan failed ({marker['failsafe']}): failsafe, plain self-play segments without "
                     f"league or pace flags this iteration")
        self.mark("league_plan", marker)

    def selfplay_left(self):
        planned = self.planned_segments()
        if planned is None:
            return self.games_played() < self.args.games
        return any(not self.done(name) for name in planned)

    # --shared-selfplay: segments claimed through the shared filesystem.
    def selfplay_mode(self):
        """stages/selfplay_mode.done: how the iteration began its self-play, or None (begun without
        --shared-selfplay by a loop that never ran with it, or not begun)."""
        try:
            return json.loads((self.it / "stages" / "selfplay_mode.done").read_text())
        except (OSError, ValueError):
            return None

    def shared_iteration(self):
        """Whether this iteration's self-play is shared: only when it began shared."""
        return bool((self.selfplay_mode() or {}).get("shared"))

    def legacy_segment_games(self):
        """The segment size an iteration begun without --shared-selfplay was playing: its seg00's game
        count (finished or checkpointed), else this loop's --segment-games."""
        try:
            return int(json.loads((self.it / "stages" / "seg00.done").read_text())["games"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        progress = self.it / "search" / "train_run" / "seg00" / "checkpoint" / "progress"
        if progress.is_file():
            try:
                return checkpoint_games(progress)
            except (OSError, ValueError, StageFailed):
                pass
        return self.args.segment_games

    def iteration_segment_games(self):
        """Games per unplanned segment: the size the iteration recorded when it began (selfplay_mode),
        else --segment-games (a loop that never ran with --shared-selfplay: exactly as before)."""
        recorded = (self.selfplay_mode() or {}).get("segment_games")
        return int(recorded) if recorded else self.args.segment_games

    @property
    def owner(self):
        return f"{socket.gethostname()}:{os.getpid()}:job{os.environ.get('SLURM_JOB_ID', '-')}"

    def heartbeat(self):
        """While the claimed segment plays, keep its claim fresh."""
        k = getattr(self, "claimed", None)
        if k is None:
            return contextlib.nullcontext()
        claims, stop = Claims(self.it / "stages", self.owner), threading.Event()

        def beat():
            while not stop.wait(HEARTBEAT_SECONDS):
                claims.beat(k)

        @contextlib.contextmanager
        def running():
            thread = threading.Thread(target=beat, daemon=True)
            thread.start()
            try:
                yield
            finally:
                stop.set()
                thread.join()
        return running()

    def next_claim(self):
        """Claim the next local segment still needed, as (k, games), or None once
        the finished and the live-claimed games reach --games."""
        claims = Claims(self.it / "stages", self.owner)
        planned = self.planned_segments()
        while True:
            live = claims.live()
            balance = None
            if planned is not None:
                # --search-balance: exactly the plan's segments, each with its own game count.
                left = [name for name in planned if not self.done(name) and int(name[3:]) not in live]
                if not left:
                    return None
                k, spec = int(left[0][3:]), planned[left[0]]
                games = spec["games"]
                balance = {"theta": spec["theta"], "engine_args": spec["engine_args"]}
            else:
                committed = self.games_played() + sum(live.values())
                if committed >= self.args.games:
                    return None
                k = 0
                while self.done(f"seg{k:02d}") or k in live:
                    k += 1
                if k >= 999:
                    raise StageFailed("out of local self-play seeds")
                games = min(self.iteration_segment_games(), self.args.games - committed)
            # A segment taken over from a dead claim resumes with the count its checkpoint holds.
            progress = self.it / "search" / "train_run" / f"seg{k:02d}" / "checkpoint" / "progress"
            if self.checkpoint_selfplay and progress.is_file():
                games = checkpoint_games(progress)
            if claims.take(k, games, balance):
                return k, games
            time.sleep(1)  # another process claimed k first, or its claim is still being written

    def shared_segment(self):
        """Claim and play one segment: False when none is needed any more."""
        claim = self.next_claim()
        if claim is None:
            return False
        k, games = claim
        name = f"seg{k:02d}"
        # A search balance challenger segment plays its twin's seed.
        seed = (self.planned_segments() or {}).get(name, {}).get("seed", 1000 * self.n + k)
        self.claimed = k
        try:
            self.attempt(self.selfplay, name, games, seed, self.it / "search" / "train" / name,
                         self.it / "search" / "train_run" / name)
        finally:
            self.claimed = None
        Claims(self.it / "stages", self.owner).release(k)
        return True

    def run_hook(self):
        hook = self.option("selfplay_hook")
        if not hook:
            return
        self.log(f"it{self.n}: self-play hook: {hook}")
        result = subprocess.run(hook, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                env={**os.environ, "RL_ITERATION": str(self.n), "RL_RUNS": str(self.runs)})
        for line in (result.stdout + result.stderr).splitlines():
            self.log(f"it{self.n}: hook: {line}")
        if result.returncode:
            self.log(f"it{self.n}: the self-play hook failed ({result.returncode}): playing on without it")

    def work(self):
        """--selfplay-worker: play segments of the iteration now in self-play, then exit."""
        n = next_iteration(self.runs)
        self.n, self.it = n, self.runs / f"rl-it{n}"
        self.arms = self.it / "arms"
        if not (self.it / "stages" / "generator").is_file():
            self.log(f"worker {self.owner}: it{n} has not started; exiting")
            return 0
        if not self.shared_iteration():
            self.log(f"worker {self.owner}: it{n} began without --shared-selfplay (no shared "
                     f"stages/selfplay_mode.done): its main loop plays it alone; exiting")
            return 0
        self.log(f"worker {self.owner}: joining it{n} self-play")
        played = 0
        while True:
            self.check_stop()
            if not self.shared_segment():
                break
            played += 1
        self.log(f"worker {self.owner}: it{n} needs no more segments; played {played}, exiting")
        return played

    def publish(self, accepting):
        order = {"protocol": PROTOCOL, "iteration": self.n, "accepting": accepting, "games": self.args.games,
                 "time_model": self.args.time_model}
        try:
            self.exchange.publish(order, self.generator)
        except Exception as error:
            self.log(f"it{self.n}: publishing the work order on {self.exchange} failed: {error!r}")
            return False
        self.log(f"it{self.n}: work order on {self.exchange} {'open' if accepting else 'closed'}")
        return True

    def collect(self):
        """Take in contributors' batches. A batch played with the generator of
        its iteration joins that iteration's games, even once it has trained;
        any other batch is closed unused."""
        for submission in self.exchange.submissions():
            manifest = submission.manifest
            n, tail = manifest.get("iteration"), submission.folder.rsplit("/", 1)[-1]
            stages = self.runs / f"rl-it{n}" / "stages"
            name = f"seg-{tail}"
            if (stages / f"{name}.done").is_file():  # collected before the batch could be closed
                self.exchange.finish(submission, True, "Collected.")
                continue
            if (type(n) is not int or not re.fullmatch(r"[A-Za-z0-9._-]+", tail)
                    or manifest.get("protocol") != PROTOCOL or not (stages / "generator").is_file()
                    or manifest.get("generator") != (stages / "generator").read_text().strip()):
                self.exchange.finish(submission, False, "Not played with the generator of its iteration.")
                self.log(f"closed {submission.folder} from {manifest.get('contributor')} unused: "
                         f"not played with the generator of its iteration")
                continue
            data = self.runs / f"rl-it{n}" / "search" / "train" / name
            shutil.rmtree(data, ignore_errors=True)
            self.exchange.fetch(submission, data)
            problem = next(filter(None, map(chunk_problem, sorted(data.glob("*.dst")))), None)
            if problem:
                shutil.rmtree(data)
                self.exchange.finish(submission, False, f"Unusable games: {problem}.")
                self.log(f"closed {submission.folder} from {manifest.get('contributor')} unused: {problem}")
                continue
            positions = train.positions([data])
            if not positions:
                shutil.rmtree(data)
                self.exchange.finish(submission, False, "No positions.")
                continue
            marker = {"games": manifest["games"], "positions": positions, "generator": manifest["generator"],
                      "contributor": manifest.get("contributor"), "folder": submission.folder}
            (stages / f"{name}.tmp").write_text(json.dumps(marker) + "\n")
            (stages / f"{name}.tmp").replace(stages / f"{name}.done")
            self.exchange.finish(submission, True, f"Collected for iteration {n}. Thank you!")
            self.log(f"it{n}: collected {manifest['games']} games ({positions} positions) from "
                     f"{manifest.get('contributor')}")

    def gather(self):
        try:
            self.collect()
        except Exception as error:  # the network: the next poll retries
            self.log(f"collecting games from {self.exchange} failed: {error!r}")

    def distributed_selfplay(self):
        """Self-play shared with contributors until --games games are in."""
        if self.done("selfplay"):
            return
        published, reported, k = False, None, 0
        while True:
            self.check_stop()
            self.gather()
            played = self.games_played()
            if played >= self.args.games:
                break
            if not published:
                published = self.publish(True)
            if self.args.local_selfplay:
                while self.done(f"seg{k:02d}"):
                    k += 1
                if k >= 999:
                    raise StageFailed("out of local self-play seeds")
                name = f"seg{k:02d}"
                self.attempt(self.selfplay, name, min(self.args.segment_games, self.args.games - played),
                             1000 * self.n + k, self.it / "search" / "train" / name,
                             self.it / "search" / "train_run" / name)
            else:
                if played != reported:
                    self.log(f"it{self.n}: {played} of {self.args.games} games in")
                    reported = played
                time.sleep(COLLECT_SECONDS)
        self.publish(False)
        self.mark("selfplay", {"games": played})
        self.log(f"it{self.n}: {played} self-play games in")

    def train(self):
        if self.done("train"):
            return
        self.check_stop()
        previous = self.runs / f"rl-it{self.n - 1}"
        # --train-from generator: the better network (the promotion rule) is
        # where the next iteration starts, not an unpromoted candidate.
        if previous.exists() and self.option("train_from") == "previous":
            init = previous / "arms" / "C" / "last.pt"
        else:
            init = train.generator_checkpoint(self.generator, self.checkpoint)
        if not init.is_file():
            raise StageFailed(f"missing {init}")
        replays = replay_iterations(self.n, self.args.window)
        for m in [self.n, *replays]:
            archive.restore(self.runs / f"rl-it{m}" / "search", self.log)
        for directory in self.teacher_val:
            archive.restore(directory, self.log)
        new = train_dirs(self.runs, self.n)
        old = [d for m in replays for d in train_dirs(self.runs, m)]
        # --search-balance: the plan's step rule (policy_positions leaves the playout cap's masked records
        # out of the count) and trainer options (--search-q-blend); all segments train, the challenger's too.
        balance = self.balance_plan_marker() if self.planned_segments() is not None else None
        rule = (balance or {}).get("steps_rule", "positions")
        if rule not in STEPS_RULES:
            raise StageFailed(f"unknown search balance steps rule {rule!r}")
        count = getattr(train, rule)
        npos, opos = count(new), count(old)
        if not npos:
            raise StageFailed("no new self-play positions")
        if npos < self.args.batch_size:
            # distill-train draws whole batches from each window of chunks and
            # would wait forever for one.
            raise StageFailed(f"{npos} new positions, fewer than one batch of {self.args.batch_size}: "
                              f"play more games or lower --batch-size")
        steps, fraction = train.plan(npos, opos, self.args.new_passes, self.args.replay_passes,
                                     self.args.batch_size)
        out, log = self.arms / "C", self.arms / "C.log"
        self.aside(out, log)
        self.wait_for_quiet()
        self.log(f"it{self.n} training: {steps} steps from {init}, {npos} new positions "
                 f"x{self.args.new_passes}, {opos} replay positions x{self.args.replay_passes} "
                 f"(replay share {fraction}, {len(old)} dirs, batch {self.args.batch_size})"
                 + ("" if balance is None else f", search balance steps rule {rule}, trainer args "
                    f"{' '.join(balance.get('trainer_args') or []) or 'none'}"))
        command = train.distill_command(init, new, [self.it / "search" / "val"], old, steps, fraction,
                                        self.args.batch_size, self.n, out, self.teacher_val,
                                        lr=self.option("lr"), trunk_lr=self.option("trunk_lr"),
                                        extra=(["--arrivals-weight", str(self.option("arrivals_weight"))]
                                               if self.option("wide") else [])
                                        + [str(a) for a in (balance or {}).get("trainer_args") or []]
                                        + self.think_trainer_args())
        with log.open("w") as stream:
            if subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              env={**os.environ, **train.TRAINING_ENV}).returncode:
                raise StageFailed(f"training failed (see {log})")
        onnx = next(out.glob("distill-*-v3.0.onnx"), None)
        if not (out / "last.pt").is_file() or onnx is None:
            raise StageFailed(f"training left no last.pt and ONNX in {out}")
        install(onnx, self.it / "models" / "C.onnx")
        install(onnx, self.generator.with_name(f"{self.generator.stem}-it{self.n}.onnx"))
        marker = {"steps": steps, "replay_fraction": fraction, "new_positions": npos,
                  "replay_positions": opos, "init": sha256(init), "onnx": sha256(self.it / "models" / "C.onnx")}
        if balance is not None:
            marker["search_balance"] = {"steps_rule": rule, "trainer_args": balance.get("trainer_args") or [],
                                        "main_mix": self.main_mix()}
        if self.option("think_on"):
            marker["think_trainer_args"] = self.think_trainer_args()
        self.mark("train", marker)
        self.archive(self.n, *replays)
        if not self.args.keep_selfplay:
            for directory in self.teacher_val:
                archive.evict(directory)

    def think_trainer_args(self):
        return [str(a) for a in hook("think").trainer_args(self.args)] if self.option("think_on") else []

    def think_labels(self):
        """--think-labels: `think.label_command(...)` after self-play, before training. Retried like a stage, but
        never fatal: after the retries the stage is marked {"failed": reason} and training runs without labels.
        The marker is the JSON line the command printed last."""
        if self.done("think_labels"):
            return
        command = hook("think").label_command(self.args, self.n, self.it, self.generator, self.args.selfplay_engine)
        if command is None:
            return
        self.check_stop()
        command = [str(c) for c in command]
        log = self.it / "search" / "logs" / "think_labels.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.wait_for_quiet()
        self.log(f"it{self.n} think labels: {' '.join(command)}")
        reason = None
        for count in range(1, ATTEMPTS + 1):
            try:
                result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
                stdout, output, status = result.stdout, result.stdout + result.stderr, result.returncode
            except OSError as error:
                stdout, output, status = "", f"{error!r}\n", 127
            with log.open("a") as stream:
                stream.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} attempt {count}: exit {status}\n{output}")
            report = last_json_line(stdout)
            if not status and report is not None:
                self.log(f"it{self.n} think labels: {json.dumps(report)}")
                self.mark("think_labels", report)
                return
            reason = f"exit {status}" if status else "no JSON report"
            self.log(f"it{self.n}: think labels failed ({reason}; see {log}; attempt {count} of {ATTEMPTS})")
            if count < ATTEMPTS:
                time.sleep(RETRY_SECONDS)
        self.log(f"it{self.n}: think labels skipped after {ATTEMPTS} attempts: training without them")
        self.mark("think_labels", {"failed": reason})

    def match(self, name, baseline, baseline_time_model=None):
        """baseline_time_model "flag" makes a clocked match's baseline search on
        the binary time advantage, for a network that cannot read the clock."""
        if self.done(f"match_{name}"):
            return
        self.check_stop()
        if name == "generator" and self.option("gate_sprt"):
            return self.sequential_match(name, baseline, baseline_time_model)
        out, log = self.arms / f"match_{name}", self.arms / f"match_{name}.log"
        games, movetime, clock = self.args.match_games, self.args.match_movetime, self.args.time_model
        flag_baseline = clock == "clock" and baseline_time_model == "flag"
        self.tournament(name, baseline, out, log, games, self.n, self.args.openings, flag_baseline)
        summary = {**match_summary(out / "summary.json", games), "movetime": movetime, "time_model": clock}
        if not self.option("wave2_options").is_stock:
            summary["wave2"] = self.option("wave2_options").as_dict()
        if flag_baseline:
            summary["baseline_time_model"] = "flag"
        self.log(f"it{self.n} vs {name}: {json.dumps(summary)}")
        self.mark(f"match_{name}", summary)

    def tournament(self, name, baseline, out, log, games, seed, openings, flag_baseline, note="", contender=None):
        """Play one engine tournament into `out`, restarting it when a game starts
        (`contender` defaults to the iteration's candidate, models/C.onnx)."""
        movetime, clock = self.args.match_movetime, self.args.time_model
        while True:
            # A resumable match keeps its finished pairs; the engine starts over
            # by itself when they came from another match.
            if not self.resume_matches:
                self.aside(out, log)
            self.wait_for_quiet()
            self.log(f"it{self.n} match vs {name} ({baseline}, {games} games at {movetime} ms"
                     + (", on a simulated clock" if clock == "clock" else "") + note + ")")
            command = [str(self.args.match_engine), "tournament",
                       "--contender", str(contender or self.it / "models" / "C.onnx"), "--baseline", str(baseline),
                       "--movetime", str(movetime), "--games", str(games), "--seed", str(seed),
                       "--positions", str(openings), "--output", str(out),
                       *(("--time-model", "clock") if clock == "clock" else ()),
                       *(("--baseline-time-model", "flag") if flag_baseline else ()),
                       *(("--resume", "true") if self.resume_matches else ()),
                       *self.option("wave2_options").engine_args()]
            with log.open("a" if self.resume_matches else "w") as stream:
                process = subprocess.Popen(command, cwd=ENGINE_DIR, stdout=stream, stderr=subprocess.STDOUT)
                status = None
                while status is None:
                    try:
                        status = process.wait(timeout=POLL_SECONDS)
                    except subprocess.TimeoutExpired:
                        if self.gaming():
                            process.terminate()
                            process.wait()
                            break
            if status is None:
                self.log(f"it{self.n} match vs {name}: a game started, restarting the match")
                continue
            if status:
                raise StageFailed(f"match vs {name} failed (see {log})")
            return

    def sequential_match(self, name, baseline, baseline_time_model=None):
        """--gate-sprt: chunks of --gate-chunk games until the SPRT decides or
        --gate-max-games are played. Each finished chunk leaves
        match_NAME/chunkNN.done, so a rerun replays the same decisions and plays
        on from the first unfinished chunk (resumed by the engine where it can)."""
        out = self.arms / f"match_{name}"
        out.mkdir(parents=True, exist_ok=True)
        movetime, clock = self.args.match_movetime, self.args.time_model
        flag_baseline = clock == "clock" and baseline_time_model == "flag"
        elo0, elo1 = self.option("gate_sprt")
        alpha, beta = self.option("gate_sprt_alpha"), self.option("gate_sprt_beta")
        bounds = sprt_bounds(alpha, beta)
        cap = self.option("gate_max_games") or self.args.match_games
        chunks, wins, losses, draws, k = [], 0, 0, 0, 0
        llr, decision = 0.0, None
        while decision is None and wins + losses + draws < cap:
            played = wins + losses + draws
            directory, marker = out / f"chunk{k:02d}", out / f"chunk{k:02d}.done"
            seed = 1000 * self.n + k
            if marker.is_file():
                result = json.loads(marker.read_text())
                games = result["games"]
            else:
                games = min(self.option("gate_chunk"), cap - played)
                openings = out / f"chunk{k:02d}.openings.tsv"
                openings.write_text(rotated_openings(self.args.openings, played // 2))
                self.check_stop()
                self.tournament(name, baseline, directory, out / f"chunk{k:02d}.log", games, seed, openings,
                                flag_baseline, f", chunk {k} of up to {cap} games, seed {seed}")
                result = {**match_summary(directory / "summary.json", games), "games": games, "seed": seed}
                marker.with_suffix(".tmp").write_text(json.dumps(result) + "\n")
                marker.with_suffix(".tmp").replace(marker)
            chunks.append((directory, seed, games))
            wins, losses, draws = wins + result["wins"], losses + result["losses"], draws + result["draws"]
            llr = sprt_llr(wins, losses, draws, elo0, elo1)
            decision = sprt_decision(llr, bounds, wins + losses + draws, cap)
            self.log(f"it{self.n} vs {name} after {wins + losses + draws} games: {wins}-{losses}-{draws}, "
                     f"LLR {llr:.2f} in [{bounds[0]:.2f}, {bounds[1]:.2f}]"
                     + (f": accepts {decision}" if decision else ""))
            k += 1
        played = wins + losses + draws
        sprt = {"llr": llr, "bounds": list(bounds), "decision": decision or "inconclusive", "elo0": elo0,
                "elo1": elo1, "alpha": alpha, "beta": beta, "chunk": self.option("gate_chunk"),
                "max_games": cap, "chunks": len(chunks)}
        merge_chunks(out, chunks, sprt)
        summary = {**match_summary(out / "summary.json", played), "movetime": movetime, "time_model": clock}
        if not self.option("wave2_options").is_stock:
            summary["wave2"] = self.option("wave2_options").as_dict()
        if flag_baseline:
            summary["baseline_time_model"] = "flag"
        summary.update(games=played, sprt=sprt)
        self.log(f"it{self.n} vs {name}: {json.dumps(summary)}")
        self.mark(f"match_{name}", summary)

    # --gate2: the real-clock two-score gate (talisman cluster/gate2_match.sh, hivemind.gate2).
    @property
    def gate2_root(self):
        """RUN/gate2 of the launcher layout runs/full/RUN/runs."""
        return self.runs.parent / "gate2"

    def gate2_settings(self):
        """What the gate's networks play (the script's GATE2_* and every net dir's config.json settings)."""
        return {"games": self.option("gate2_games"), "parts": self.option("gate2_parts"),
                "base_ms": self.option("gate2_base_ms"), "free_k": self.option("gate2_free_k"),
                "equal_k": self.option("gate2_equal_k"), "seed_base": self.option("gate2_seed_base"),
                "opp_sit": self.option("gate2_opp_sit"), "arms": GATE2_ARMS.split()}

    def write_once(self, path, value):
        """Write `value` as JSON to `path` unless it is there; refuse other content already there."""
        if path.is_file():
            try:
                there = json.loads(path.read_text())
            except ValueError:
                there = None
            if there != value:
                raise StageFailed(f"{path} holds {there}, not this run's {value}: refused (never mixed)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        path.with_suffix(".tmp").replace(path)

    def gate2_call(self, script, label, onnx):
        """One call of the matches script for one network: it submits what is missing and merges what is finished.
        A failure is logged; the next poll calls it again."""
        settings = self.gate2_settings()
        env = {**os.environ, "GATE2_GAMES": str(settings["games"]), "GATE2_PARTS": str(settings["parts"]),
               "GATE2_BASE_MS": str(settings["base_ms"]), "GATE2_FREE_K": str(settings["free_k"]),
               "GATE2_EQUAL_K": str(settings["equal_k"]), "GATE2_SEED_BASE": str(settings["seed_base"]),
               "GATE2_OPP_SIT": settings["opp_sit"], "GATE2_ARMS": GATE2_ARMS}
        try:
            result = subprocess.run(["bash", str(script), str(self.runs.parent), label, str(onnx)],
                                    cwd=script.parent.parent, env=env, capture_output=True, text=True)
            output, status = result.stdout + result.stderr, result.returncode
        except OSError as error:
            output, status = repr(error), 127
        for line in output.splitlines():
            self.log(f"it{self.n} gate2: {line}")
        if status:
            self.log(f"it{self.n} gate2: {script.name} {label} failed (exit {status}): trying again at the next poll")

    def gate2(self):
        """Stage gate2 (after match_generator, before promote): poll the matches script for the candidate and the
        incumbent until both net dirs are done or --gate2-timeout-h has passed since stages/gate2.started, then
        decide (hivemind.gate2.decide; a timeout is never a promotion)."""
        if self.done("gate2"):
            return
        self.check_stop()
        stages = self.it / "stages"
        settings = self.gate2_settings()
        self.write_once(self.gate2_root / "wave2.json", self.option("wave2_options").as_dict())
        self.write_once(self.gate2_root / "settings.json", settings)
        script = gate2_script(self.option("gate2_script"))
        if script is None:
            raise StageFailed(f"no gate2 matches script ({self.option('gate2_script') or 'cluster/gate2_match.sh'})")
        if not (stages / "gate2.started").is_file():
            (stages / "gate2.started").write_text(f"{time.time()}\n")
        started = float((stages / "gate2.started").read_text())
        timeout = 3600 * self.option("gate2_timeout_h")
        candidate = self.it / "models" / "C.onnx"
        generator = (stages / "generator").read_text().strip()
        # The incumbent is the generator this iteration played; its file still is the generator before promotion.
        incumbent = next((path for path in (self.generator, self.generator_sp1)
                          if path.is_file() and sha256(path) == generator), None)
        nets = {"candidate": (f"it{self.n}-candidate", candidate, sha256(candidate)),
                "incumbent": (f"it{self.n}-incumbent", incumbent, generator)}
        dirs = {role: self.gate2_root / "nets" / sha[:16] for role, (_, _, sha) in nets.items()}
        self.log(f"it{self.n} gate2: candidate {dirs['candidate'].name}, incumbent {dirs['incumbent'].name}, "
                 f"{json.dumps(settings, sort_keys=True)}, timeout {self.option('gate2_timeout_h'):g} h")
        timed_out = False
        while True:
            for role, (label, onnx, _) in nets.items():
                if (dirs[role] / "done").is_file():
                    continue
                if onnx is None:
                    self.log(f"it{self.n} gate2: no file of the incumbent {generator[:16]} to submit")
                    continue
                self.gate2_call(script, label, onnx)
            if all((d / "done").is_file() for d in dirs.values()):
                break
            if time.time() - started >= timeout:
                timed_out = True
                break
            self.check_stop()
            time.sleep(self.option("gate2_poll_s"))
        decision = two_score.decide(dirs["candidate"], dirs["incumbent"], self.option("gate2_drop_margin"),
                                    self.option("gate2_min_pairs"))
        if timed_out:
            decision["promoted"] = False
            decision["reason"] = (f"incomplete: not done after {self.option('gate2_timeout_h'):g} h "
                                  f"({', '.join(r for r in dirs if not (dirs[r] / 'done').is_file())}); "
                                  f"{decision['reason']}")
        marker = {key: decision[key] for key in ("candidate", "incumbent", "arms", "promoted", "reason")}
        marker["settings"] = {**settings, "drop_margin": self.option("gate2_drop_margin"),
                              "min_pairs": self.option("gate2_min_pairs"),
                              "timeout_h": self.option("gate2_timeout_h")}
        marker["hours"] = round((time.time() - started) / 3600, 3)
        self.log(f"it{self.n} gate2: {'promote' if decision['promoted'] else 'no promotion'} ({decision['reason']})")
        self.mark("gate2", marker)

    def promote(self):
        if self.done("promote"):
            return
        result = json.loads((self.it / "stages" / "match_generator.done").read_text())
        score = result["score"]
        above = self.option("promote_above")
        promoted = score >= 0.5 if above is None else score > above
        # A sequential gate's decision overrides the score rule; undecided, the rule stands.
        decision = result.get("sprt", {}).get("decision")
        if decision in ("H0", "H1"):
            promoted = decision == "H1"
        why = f"score {score}" + (f", SPRT {decision}" if "sprt" in result else "")
        # --gate2: the real-clock two-score gate decides; the 100 ms score stays in the marker for old readers.
        gate = json.loads((self.it / "stages" / "gate2.done").read_text()) if self.option("gate2") else None
        if gate is not None:
            promoted = gate["promoted"]
            why += f", gate2 {gate['reason']}"
        if promoted:
            install(self.it / "models" / "C.onnx", self.generator)
            install(self.it / "models" / "C.onnx", self.generator_sp1)
            install(self.arms / "C" / "last.pt", self.checkpoint)
            self.log(f"it{self.n} promoted ({why}): it is the generator for it{self.n + 1}")
        else:
            self.log(f"it{self.n} not promoted ({why}): the generator stays")
        self.mark("promote", {"promoted": promoted, "score": score,
                              **({"sprt": decision} if "sprt" in result else {}),
                              **({"gate2": {"promoted": gate["promoted"], "arms": gate["arms"],
                                            "reason": gate["reason"]}} if gate is not None else {})})

    def upload(self):
        """Not fatal: the next iteration's upload retries whatever is missing."""
        if self.args.no_upload or self.done("upload"):
            return
        for count in range(1, ATTEMPTS + 1):
            try:
                sync_hub(self.api, self.args.repo, self.generator, self.checkpoint, self.log, self.runs)
                self.mark("upload", {"repo": self.args.repo, "generator": sha256(self.generator)})
                return
            except Exception as error:
                self.log(f"it{self.n}: upload to {self.args.repo} failed: {error!r} "
                         f"(attempt {count} of {ATTEMPTS})")
                if count < ATTEMPTS:
                    time.sleep(RETRY_SECONDS)
        self.log(f"it{self.n}: continuing without uploading")

    def archive(self, *iterations):
        """Move the finished self-play of these iterations, and what their
        interrupted stages left in rl-itN/aborted, to --archive. Not fatal:
        what is left stays on disk and the next call retries."""
        if self.args.keep_selfplay:
            return
        for m in iterations:
            it = self.runs / f"rl-it{m}"
            for directory, paths in ((it / "search", finished_selfplay(it)), (it / "aborted", None)):
                if not directory.is_dir():
                    continue
                try:
                    freed = archive.archive(self.api, self.args.archive, directory, self.log, paths)
                except Exception as error:
                    self.log(f"it{self.n}: archiving {directory} to {self.args.archive} failed: {error!r}; "
                             f"it stays on disk")
                    return
                if freed:
                    self.log(f"{it.name}: freed {freed / 1e9:.1f} GB on disk")

    # --search-balance: the controller (talisman cluster/search_balance.py) and `hivemind balance` stages.
    @property
    def balance_root(self):
        """RUN of runs/full/RUN: the loop's --runs is RUN/runs, the controller's files are in RUN/balance."""
        return self.runs.parent

    @property
    def balance_controller(self):
        return search_balance_controller(self.option("search_balance_controller"))

    def balance_file(self, *parts):
        return self.balance_root.joinpath("balance", *parts)

    def balance_on(self):
        """Whether this iteration runs the balance stages: the flag, and a real plan (not failsafe, not skipped)."""
        return bool(self.option("search_balance")) and bool((self.balance_plan_marker() or {}).get("plan"))

    def balance_call(self, what, command):
        """Run one balance step, its output appended to RUN/balance/itN/logs/WHAT.log: (exit code, log, output)."""
        log = self.balance_file(f"it{self.n}", "logs", f"{what}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} $ {' '.join(map(str, command))}\n")
            stream.flush()
            start = stream.tell()
            controller = self.balance_controller
            env = {**os.environ, **({"HIVEMIND_SEARCH_BALANCE": str(controller)} if controller else {})}
            try:
                status = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env=env).returncode
            except OSError as error:
                stream.write(f"{error!r}\n")
                status = 127
        with log.open("rb") as stream:
            stream.seek(start)
            output = stream.read().decode(errors="replace")
        return status, log, output

    def controller_command(self, *argv):
        return [sys.executable, str(self.balance_controller), *map(str, argv)]

    def balance_stage_command(self, stage, *argv):
        # The module itself, not `-m hivemind balance`: hivemind/__main__.py drops main()'s return code, so a stage's
        # exit 2 (BalanceError) would read as success.
        return [sys.executable, "-m", "hivemind.cli.balance", stage, str(self.balance_root), str(self.n),
                *map(str, argv)]

    def controller_prior_args(self):
        prior = self.option("search_balance_prior")
        return ["--prior", str(prior)] if prior else []

    def balance_plan(self):
        """`search_balance.py plan RUN N`, its segment table frozen in
        stages/balance_plan.done (written before the iteration's generator
        marker, so a worker that sees the iteration sees the plan). If the
        controller fails, a failsafe table: stock self-play, no balance stages."""
        if self.done("balance_plan"):
            return
        args, n = self.args, self.n
        segments = -(-args.games // args.segment_games)
        command = self.controller_command(
            "plan", self.balance_root, n, "--segments", segments, "--segment-games", args.segment_games,
            "--mini-steps", self.option("search_balance_mini_steps"), "--generator", self.generator)
        if self.option("search_balance_prior"):
            command += ["--prior", str(self.option("search_balance_prior"))]
        path, failure = self.balance_file(f"it{n}.plan.json"), None
        if self.balance_controller is None:
            failure, log = "no controller (--search-balance-controller)", None
        else:
            status, log, _ = self.balance_call("plan", command)
            if status:
                failure = f"exit {status}"
        if failure is None:
            try:
                plan = json.loads(path.read_text())
                if plan.get("iteration") != n or sorted(plan["segments"]) != [f"seg{k:02d}" for k in range(segments)]:
                    raise ValueError(f"segments {sorted(plan['segments'])}, iteration {plan.get('iteration')}")
                if plan.get("val", {}).get("engine_args"):
                    raise ValueError("validation is not stock")
                table = {name: {"theta": s["theta"], "seed": int(s["seed"]), "games": int(s["games"]),
                                "engine_args": [str(a) for a in s.get("engine_args") or []],
                                **({"twin": s["twin"]} if s.get("twin") else {})}
                         for name, s in sorted(plan["segments"].items())}
            except (OSError, ValueError, KeyError, TypeError) as error:
                failure = f"unreadable plan {path}: {error}"
        if failure is not None:
            table = {f"seg{k:02d}": {"theta": "stock", "seed": 1000 * n + k, "engine_args": [],
                                     "games": min(args.segment_games, args.games - k * args.segment_games)}
                     for k in range(segments)}
            marker = {"failsafe": failure, "segments": table, "trainer_args": [], "steps_rule": "positions",
                      "gpus_per_job": self.option("search_balance_gpus")}
            self.log(f"it{n} search balance plan failed ({failure}" + (f"; see {log}" if log else "")
                     + "): failsafe, stock self-play and no balance stages this iteration")
        else:
            marker = {"plan": str(path.relative_to(self.balance_root)), "plan_sha256": sha256(path),
                      "mode": plan.get("mode"), "reason": plan.get("reason"), "theta_star": plan.get("theta_star"),
                      "challenger": plan.get("challenger"), "challenger_segment": plan.get("challenger_segment"),
                      "twin_segment": plan.get("twin_segment"), "segments": table,
                      "trainer_args": [str(a) for a in plan.get("trainer_args") or []],
                      "steps_rule": plan.get("steps_rule", "positions"),
                      "probe_refresh": bool((plan.get("probe") or {}).get("refresh")),
                      "gpus_per_job": self.option("search_balance_gpus")}
            arms = ", ".join(f"{name} {s['theta']} {s['games']} games"
                             + (f" {' '.join(s['engine_args'])}" if s["engine_args"] else "")
                             for name, s in table.items())
            self.log(f"it{n} search balance plan: mode {marker['mode']}, challenger "
                     f"{json.dumps(marker['challenger'], sort_keys=True) if marker['challenger'] else 'none'} "
                     f"({marker['reason']}); {arms}")
        self.mark("balance_plan", marker)

    def balance_step(self, what, command, marker=None):
        """Run one balance step with the loop's retries. A step that keeps
        failing is logged and skipped, never fatal: the controller then has no
        evidence this iteration (or plans failsafe next). `marker`: the stage
        marker the step leaves (None: the loop marks balance_WHAT itself).
        Returns (ok, output of the last attempt)."""
        stage = marker or f"balance_{what}"
        if self.done(stage):
            return True, ""
        self.check_stop()
        output = ""
        for count in range(1, ATTEMPTS + 1):
            status, log, output = self.balance_call(what, command)
            if not status and (marker is None or self.done(stage)):
                return True, output
            tail = " | ".join(output.strip().splitlines()[-2:])
            self.log(f"it{self.n}: search balance {what} failed (exit {status}: {tail}; see {log}; "
                     f"attempt {count} of {ATTEMPTS})")
            if count < ATTEMPTS:
                time.sleep(RETRY_SECONDS)
        self.log(f"it{self.n}: search balance {what} skipped after {ATTEMPTS} attempts")
        return False, output

    def main_mix(self):
        """The share of this iteration's training positions each search balance arm played."""
        mix = {}
        for marker in sorted((self.it / "stages").glob("seg*.done")):
            done = json.loads(marker.read_text())
            arm = (done.get("balance") or {}).get("theta", "other")
            mix[arm] = mix.get(arm, 0) + (done.get("positions") or 0)
        total = sum(mix.values())
        return {arm: round(count / total, 6) for arm, count in mix.items()} if total else {}

    def balance_device(self):
        device = self.option("search_balance_device")
        if device != "auto":
            return device
        probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True)
        return "cuda" if probe.stdout.strip() == "True" else "cpu"

    def balance_after_train(self):
        """The probe refresh, the four mini-trains and their scores (docs/search_balance.md section 4)."""
        if not self.balance_on():
            return
        plan = json.loads(self.balance_file(f"it{self.n}.plan.json").read_text())
        gpus = self.option("search_balance_gpus")
        if (plan.get("probe") or {}).get("refresh") or plan.get("challenger"):
            # Training archived the segments again: the mini-trains (and probe wave 2 options) need them.
            for m in (self.n, self.n - 1):
                if (self.runs / f"rl-it{m}" / "search").is_dir():
                    archive.restore(self.runs / f"rl-it{m}" / "search", self.log)
        if (plan.get("probe") or {}).get("refresh"):
            self.wait_for_quiet()
            self.balance_step("probe", self.balance_stage_command(
                "probe", "--engine", self.args.selfplay_engine, "--model", self.generator_sp1,
                "--games", self.option("search_balance_probe_games"), "--parallel-games", self.args.parallel_games,
                "--time-model", self.args.time_model, "--gpus", gpus, *(["--wide"] if self.option("wide") else [])),
                "balance_probe")
        if not (plan.get("challenger") and plan.get("minitrain")):
            return
        self.wait_for_quiet()
        trunk = self.option("trunk_lr")
        ok, _ = self.balance_step("minitrain", self.balance_stage_command(
            "minitrain", "--batch-size", self.args.batch_size, "--lr", self.option("lr"),
            *(["--trunk-lr", trunk] if trunk is not None else []), "--gpus", gpus,
            *(["--extra", f"--arrivals-weight {self.option('arrivals_weight')}"] if self.option("wide") else [])),
            "balance_minitrain")
        if ok:
            self.wait_for_quiet()
            self.balance_step("score", self.balance_stage_command(
                "score", "--device", self.balance_device(), "--gpus", gpus), "balance_score")

    def balance_minigate(self, plan):
        """100 games (--search-balance-minigate-games) of the challenger's mini
        net (seed 0) against the incumbent's, into RUN/balance/itN/minigate.json."""
        if self.done("balance_minigate"):
            return
        out_file = self.balance_file(f"it{self.n}", "minigate.json")
        games = self.option("search_balance_minigate_games")
        mini = self.balance_file(f"it{self.n}", "mini")
        nets = {arm: Path(json.loads((mini / f"{arm}_s0.done").read_text())["onnx"]) for arm in ("inc", "chal")}
        out, log = self.arms / "balance_minigate", self.arms / "balance_minigate.log"
        started = time.monotonic()
        if not out_file.is_file():
            self.tournament("search balance mini-gate", nets["inc"], out, log, games, 1000 * self.n + 997,
                            self.args.openings, False, ", challenger mini net vs incumbent mini net",
                            contender=nets["chal"])
            summary = match_summary(out / "summary.json", games)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.with_suffix(".tmp").write_text(json.dumps(
                {"games": games, **{k: summary[k] for k in ("wins", "losses", "draws", "score", "elo")}},
                sort_keys=True) + "\n")
            out_file.with_suffix(".tmp").replace(out_file)
        wall = time.monotonic() - started
        self.log(f"it{self.n} search balance mini-gate: {out_file.read_text().strip()}")
        self.mark("balance_minigate", {"path": str(out_file), "sha256": sha256(out_file), "games": games,
                                       "wall_s": round(wall, 3),
                                       "gpu_s": round(wall * self.option("search_balance_gpus"), 3)})

    def balance_finish(self):
        """After the gate: the mini-gate when it can change the decision, the
        job accounting, the realised costs and the decision (RUN/balance/itN.json)."""
        if not self.balance_on() or self.done("balance_decide"):
            return
        plan = json.loads(self.balance_file(f"it{self.n}.plan.json").read_text())
        run, n = self.balance_root, self.n
        if (plan.get("challenger") and plan.get("minigate_allowed") and not plan.get("revert_test")
                and self.done("balance_score") and not self.option("no_gate") and not self.done("balance_minigate")):
            status, log, output = self.balance_call("decide_dry", self.controller_command(
                "decide", run, n, "--dry", *self.controller_prior_args()))
            dry = last_json_line(output) if not status else None
            if dry is None:
                self.log(f"it{n}: search balance decide --dry failed (exit {status}; see {log}): no mini-gate")
            elif dry.get("run_minigate"):
                for count in range(1, ATTEMPTS + 1):
                    try:
                        self.balance_minigate(plan)
                        break
                    except Exception as error:
                        self.log(f"it{n}: search balance mini-gate failed: {error} (attempt {count} of {ATTEMPTS})")
                        if count < ATTEMPTS:
                            time.sleep(RETRY_SECONDS)
        status, log, output = self.balance_call("accounting", self.balance_stage_command(
            "accounting", "--gpus", self.option("search_balance_gpus")))
        if status:
            tail = " | ".join(output.strip().splitlines()[-2:])
            self.log(f"it{n}: search balance accounting failed (exit {status}: {tail}; see {log}): "
                     f"no gamma sample and only planned overhead this iteration")
        # realise rewrites its file only before decide has read it (--force: a redone segment changed it).
        if self.balance_step("realise", self.controller_command("realise", run, n, "--force"))[0]:
            self.mark("balance_realise", {"path": str(self.balance_file(f"it{n}", "realised.json"))})
        # A challenger whose mini-trains or scores failed: decide still runs (action none, no evidence) so the
        # overhead spent, the realised costs and the probe version reach the ledger instead of a failsafe next plan.
        no_score = ["--no-score"] if plan.get("challenger") and not self.done("balance_score") else []
        if no_score:
            self.log(f"it{n}: search balance has no mini-train scores this iteration: decide records no evidence")
        ok, _ = self.balance_step("decide", self.controller_command("decide", run, n, *self.controller_prior_args(),
                                                                    *no_score))
        result_path = self.balance_file(f"it{n}.json")
        if ok and result_path.is_file():
            result = json.loads(result_path.read_text())
            decision = result.get("decision") or {}
            self.mark("balance_decide", {"path": str(result_path), "sha256": sha256(result_path),
                                         "action": decision.get("action"), "reason": decision.get("reason")})
            self.log(f"it{n} search balance decision: {decision.get('action')} ({decision.get('reason')}); "
                     f"theta* {json.dumps((result.get('state') or {}).get('theta_star'), sort_keys=True)}, "
                     f"overhead {((result.get('state') or {}).get('overhead') or {}).get('frac')}")

    def iteration(self, n):
        self.n, self.it = n, self.runs / f"rl-it{n}"
        self.arms = self.it / "arms"
        if self.done("iteration"):
            return
        if self.it.exists() and not (self.it / "stages").is_dir():
            self.log(f"{self.it} exists but was not made by this loop")
            sys.exit(1)
        for directory in ("stages", "search", "arms", "models"):
            (self.it / directory).mkdir(parents=True, exist_ok=True)
        # The generator is fixed for a whole iteration's self-play.
        if not (self.it / "stages" / "generator").is_file():
            if self.option("search_balance"):
                self.balance_plan()  # first: workers join once the generator marker exists
            if self.option("league_on"):
                self.league_plan()  # likewise
            if self.option("shared_selfplay") and not self.done("selfplay_mode"):
                # Also before the generator marker: a worker that sees the iteration sees how it plays.
                self.mark("selfplay_mode", {"shared": True, "segment_games": self.args.segment_games})
            (self.it / "stages" / "generator").write_text(sha256(self.generator) + "\n")
        elif self.option("search_balance") and not self.done("balance_plan"):
            self.mark("balance_plan", {"skipped": "the iteration started without --search-balance"})
            self.log(f"it{n}: started without --search-balance: it plays and trains as it began, "
                     f"without balance stages")
        if self.option("league_on") and not self.done("league_plan"):
            self.mark("league_plan", {"skipped": "the iteration started without --league/--pace"})
            self.log(f"it{n}: started without --league/--pace: it plays as it began, without league segments")
        if self.option("shared_selfplay") and not self.done("selfplay_mode"):
            began = self.legacy_segment_games()
            self.mark("selfplay_mode", {"shared": False, "segment_games": began,
                                        "skipped": "the iteration started without --shared-selfplay"})
            self.log(f"it{n}: started without --shared-selfplay: its self-play finishes in this job, one "
                     f"segment after another ({began} games each), with no workers")
        self.archive(*range(1, n + 1))
        args = self.args
        planned = self.planned_segments()
        if self.exchange:
            self.distributed_selfplay()
        elif self.shared_iteration():
            if self.selfplay_left() and self.option("shared_selfplay"):
                self.run_hook()
            while self.selfplay_left():
                if self.shared_segment():
                    continue
                # Workers hold every segment still needed: play validation meanwhile, then wait.
                if not self.done("val"):
                    self.attempt(self.selfplay, "val", args.val_games, 1000 * n + 999, self.it / "search" / "val",
                                 self.it / "search" / "val_run")
                else:
                    self.check_stop()
                    time.sleep(POLL_SECONDS)
        elif planned is not None:
            for name, spec in planned.items():
                self.attempt(self.selfplay, name, spec["games"], spec["seed"], self.it / "search" / "train" / name,
                             self.it / "search" / "train_run" / name)
        else:
            k = 0
            played = self.games_played()
            size = self.iteration_segment_games()
            while played < args.games:
                while self.done(f"seg{k:02d}"):
                    k += 1
                if k >= 999:
                    raise StageFailed("out of local self-play seeds")
                name = f"seg{k:02d}"
                self.attempt(self.selfplay, name, min(size, args.games - played),
                             1000 * n + k, self.it / "search" / "train" / name,
                             self.it / "search" / "train_run" / name)
                played = self.games_played()
        self.attempt(self.selfplay, "val", args.val_games, 1000 * n + 999, self.it / "search" / "val",
                     self.it / "search" / "val_run")
        if self.exchange:
            self.gather()
        if self.option("think_on"):
            self.think_labels()
        self.attempt(self.train)
        self.balance_after_train()
        if self.option("no_gate"):
            self.log(f"it{n}: no matches (--no-gate): the generator stays")
            self.balance_finish()
            self.mark("iteration", {"gate": False})
            return
        self.attempt(self.match, "generator", self.generator)
        if self.option("gate2"):
            self.attempt(self.gate2)
        self.promote()
        self.balance_finish()
        self.upload()
        anchor = self.args.anchor_after
        if anchor is not None and n <= anchor:
            if n == anchor:
                install(self.generator, self.fixed)
                self.log(f"it{n}: the generator is now the fixed baseline {self.fixed.name}")
            else:
                self.log(f"it{n}: no fixed-baseline match before it{anchor + 1}")
        elif (self.fixed.is_file() and not self.done(f"match_{self.fixed_name}")
              and sha256(self.fixed) == (self.it / "stages" / "generator").read_text().strip()
              and self.fixed_time_model == self.args.time_model
              and "sprt" not in json.loads((self.it / "stages" / "match_generator.done").read_text())):
            # The generator is still the fixed baseline: the same match again
            # (unless the gate was sequential: the fixed match keeps its size).
            self.log(f"it{n}: the generator is {self.fixed.name}; its match counts for both")
            self.mark(f"match_{self.fixed_name}", {
                **json.loads((self.it / "stages" / "match_generator.done").read_text()), "same_as": "generator"})
        elif self.fixed.is_file():
            self.attempt(self.match, self.fixed_name, self.fixed, self.fixed_time_model)
        else:
            self.log(f"it{n}: no {self.fixed.name}, skipping the match against it")
        self.mark("iteration", {})
        results = {name: (self.it / "stages" / f"match_{name}.done") for name in ("generator", self.fixed_name)}
        self.log(f"it{n} finished: " + ", ".join(
            f"vs {name} {path.read_text().strip()}" for name, path in results.items() if path.is_file()))

    def run(self, first, last):
        self.log(f"loop: iterations {first}..{last or 'forever'}, {self.args.games} games "
                 f"({self.args.segment_games} per segment), {self.args.time_model} time model, "
                 f"generator {self.generator.name}, matches {self.args.match_games} games at "
                 f"{self.args.match_movetime} ms, window {self.args.window}, "
                 f"passes new {self.args.new_passes} / replay {self.args.replay_passes}, "
                 f"upload {'off' if self.args.no_upload else self.args.repo}, "
                 f"self-play {'kept on disk' if self.args.keep_selfplay else 'moved to ' + self.args.archive}"
                 + (f", distributed via {self.exchange}" if self.exchange else "")
                 + (f", sequential gate SPRT [{self.option('gate_sprt')[0]:g}, {self.option('gate_sprt')[1]:g}] "
                    f"alpha {self.option('gate_sprt_alpha'):g} beta {self.option('gate_sprt_beta'):g} in chunks of "
                    f"{self.option('gate_chunk')} up to {self.option('gate_max_games') or self.args.match_games} games"
                    if self.option("gate_sprt") else "")
                 + (f", search balance (controller {self.balance_controller}, files in "
                    f"{self.balance_file()})" if self.option("search_balance") else "")
                 + (f", real-clock gate {self.option('gate2_games')} games a network and arm at "
                    f"{self.option('gate2_base_ms')} ms (k {self.option('gate2_equal_k')}/"
                    f"{self.option('gate2_free_k')}, "
                    f"{gate2_script(self.option('gate2_script'))}, files in {self.gate2_root})"
                    if self.option("gate2") else ""))
        n = first
        while last is None or n <= last:
            self.iteration(n)
            n += 1
        self.log(f"loop finished at it{last}")


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def even_int(value):
    parsed = positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("must be even")
    return parsed


def sprt_elos(value):
    """ELO0,ELO1 of --gate-sprt."""
    try:
        elo0, elo1 = (float(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("must be ELO0,ELO1, e.g. 0,15")
    if not elo1 > elo0:
        raise argparse.ArgumentTypeError("ELO1 must be above ELO0")
    return elo0, elo1


def onnx_input_planes(path):
    """The input planes an ONNX network reads (78 narrow, 130 wide), or None."""
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        return model.graph.input[0].type.tensor_type.shape.dim[1].dim_value or None
    except Exception:
        return None


def hook_modules(argv):
    """{"league": module or None, "think": ...}: a module is imported when one of its flags is given (it must then
    exist) or for --help (when it exists), so runs without the flags never import it."""
    wanted = {"league": any(a.startswith(("--league", "--pace")) for a in argv),
              "think": any(a.startswith("--think") for a in argv)}
    helping = any(a in ("-h", "--help") for a in argv)
    modules = {}
    for name, used in wanted.items():
        modules[name] = None
        if used or helping:
            try:
                modules[name] = hook(name)
            except ImportError as error:
                if used:
                    sys.exit(f"rl-loop: the {name} flags need hivemind.{name}, which cannot be imported: {error}")
    return modules


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=positive_int,
                        help="First iteration (default: the first unfinished one)")
    parser.add_argument("--last", type=positive_int, help="Last iteration (default: run forever)")
    parser.add_argument("--games", type=positive_int, default=100_000)
    parser.add_argument("--segment-games", type=positive_int,
                        help="Games per local self-play segment (default: 10000, or 1000 with --distributed)")
    parser.add_argument("--val-games", type=positive_int, default=1000)
    parser.add_argument("--window", type=positive_int, default=3,
                        help="Earlier iterations replayed in training")
    parser.add_argument("--new-passes", type=float, default=1.5)
    parser.add_argument("--replay-passes", type=float, default=0.5)
    parser.add_argument("--parallel-games", type=positive_int, default=6)
    parser.add_argument("--resign-threshold", type=float, default=0.0,
                        help="Local self-play resigns after 3 searches in a row at or below this root "
                             "value, except in 10%% of games; 0 never resigns (default: %(default)s)")
    parser.add_argument("--tree-reuse", action="store_true",
                        help="Local self-play keeps the played move's subtree for the next search")
    parser.add_argument("--checkpoint-games", type=int, default=50,
                        help="Local self-play saves its finished games this often, so an interrupted "
                             "segment resumes rather than starting over; 0 turns it off "
                             "(default: %(default)s)")
    parser.add_argument("--match-games", type=even_int, default=1000,
                        help="Games per match (vs the generator and vs it0), in colour-swapped "
                             "pairs (default: %(default)s)")
    parser.add_argument("--match-movetime", type=positive_int, default=100,
                        help="Milliseconds per move in matches (default: %(default)s)")
    parser.add_argument("--time-model", choices=("flag", "clock"), default="flag",
                        help="flag: one team may sit all game, as iterations 1-10 trained; clock: "
                             "self-play and matches on a simulated clock, for networks that read "
                             "the clock planes (default: %(default)s)")
    parser.add_argument("--fixed-time-model", choices=("flag", "clock"),
                        help="What the fixed it0 baseline searches on in clock matches; flag for an it0 "
                             "from before the clock planes, which misreads them (default: --time-model)")
    parser.add_argument("--anchor-after", type=positive_int, metavar="N",
                        help="Skip the fixed-baseline match up to iteration N, then match later "
                             "iterations against the generator as it stands after iteration N "
                             "rather than it0")
    parser.add_argument("--generator", type=Path, default=TWIN_ONNX_PATH,
                        help="The generator network, replaced on promotion; another lineage needs "
                             "its own (default: %(default)s)")
    parser.add_argument("--generator-checkpoint", type=Path, default=train.GENERATOR_CHECKPOINT,
                        help="The generator's training checkpoint, which the first iteration "
                             "trains from (default: %(default)s)")
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--openings", type=Path, default=OPENINGS, help="Match opening positions (TSV)")
    parser.add_argument("--runs", type=Path, default=RUNS, help="Directory of the rl-itN runs")
    parser.add_argument("--repo", default=TWIN_REPOSITORY,
                        help="Hugging Face model repository; uploads are skipped when this machine "
                             "is not logged in to an account that can write to it")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to Hugging Face")
    parser.add_argument("--archive", default=SELFPLAY_ARCHIVE,
                        help="Private Hugging Face dataset that finished self-play moves to, compressed, "
                             "freeing the disk")
    parser.add_argument("--keep-selfplay", action="store_true",
                        help="Keep all self-play on disk rather than moving it to --archive")
    parser.add_argument("--archive-only", action="store_true",
                        help="Move all finished self-play to --archive, then exit")
    parser.add_argument("--pause-for", default="reaper|wineserver",
                        help="pgrep -x pattern of processes (Steam and Proton games by default) that "
                             "pause training and matches; empty disables")
    parser.add_argument("--distributed", action="store_true",
                        help="Share self-play with machines running `hivemind contribute`")
    parser.add_argument("--exchange", default=SELFPLAY_EXCHANGE,
                        help="Where work orders and games are exchanged with --distributed: "
                             "hf://NAMESPACE/NAME (a Hugging Face dataset) or a shared directory")
    parser.add_argument("--no-local-selfplay", action="store_true",
                        help="With --distributed, leave self-play to contributors")
    parser.add_argument("--shared-selfplay", action="store_true",
                        help="Claim each local self-play segment on the shared filesystem first, so "
                             "--selfplay-worker processes on other GPUs can play segments of the same iteration")
    parser.add_argument("--selfplay-worker", action="store_true",
                        help="Only play claimed segments of the iteration now in self-play, then exit "
                             "(implies --shared-selfplay; give it the main loop's settings)")
    parser.add_argument("--selfplay-hook", default=None,
                        help="With --shared-selfplay, a shell command run at the start of each iteration's "
                             "self-play (RL_ITERATION and RL_RUNS in its environment), e.g. to submit workers")
    parser.add_argument("--selfplay-engine", type=Path, default=selfplay.DEFAULT_ENGINE)
    parser.add_argument("--match-engine", type=Path, default=MATCH_ENGINE)
    parser.add_argument("--lr", default=None,
                        help="Learning rate of the heads, and of the trunk without --trunk-lr "
                             "(default: 1e-4; 1e-4 with --wave2)")
    parser.add_argument("--trunk-lr", default=None,
                        help="The trunk's own learning rate, for a fine-tune (default: --lr; 5e-5 with --wave2)")
    parser.add_argument("--train-from", choices=("previous", "generator"), default=None,
                        help="Where each iteration's training starts: the previous iteration's weights "
                             "whether promoted or not (upstream), or the generator's checkpoint, the "
                             "better network by the promotion rule (default: previous; generator with --wave2)")
    parser.add_argument("--promote-above", type=float, default=None,
                        help="Promote when the score against the generator is above this (default: at or "
                             "above 0.5; above 0.5 with --wave2, so a tie keeps the generator)")
    parser.add_argument("--gate-sprt", type=sprt_elos, metavar="ELO0,ELO1",
                        help="Sequential gate against the generator: stop once an SPRT of H0 Elo <= ELO0 "
                             "against H1 Elo >= ELO1 decides (e.g. 0,15); H1 promotes, H0 does not, undecided "
                             "at --gate-max-games falls back to the score rule (default: off, a fixed match)")
    parser.add_argument("--gate-sprt-alpha", type=float, default=0.05,
                        help="With --gate-sprt, the false-promotion rate (default: %(default)s)")
    parser.add_argument("--gate-sprt-beta", type=float, default=0.05,
                        help="With --gate-sprt, the false-rejection rate (default: %(default)s)")
    parser.add_argument("--gate-chunk", type=even_int, default=100,
                        help="With --gate-sprt, games played between SPRT checks (default: %(default)s)")
    parser.add_argument("--gate-max-games", type=even_int, default=None,
                        help="With --gate-sprt, the most games the gate plays; above --match-games "
                             "extends close gates (default: --match-games)")
    parser.add_argument("--search-balance", action="store_true",
                        help="Let the search balance controller (docs/search_balance.md) plan each iteration's "
                             "self-play segments and run its stages; --runs must be RUN/runs, its files go to "
                             "RUN/balance (default: off)")
    parser.add_argument("--search-balance-controller", type=Path, default=None,
                        help="With --search-balance, the controller script (default: $HIVEMIND_SEARCH_BALANCE, "
                             "else cluster/search_balance.py of the talisman repo above the fork)")
    parser.add_argument("--search-balance-prior", type=Path, default=None,
                        help="With --search-balance, the controller's prior (default: RUN/balance/prior.json; "
                             "without one it holds today's recipe and plans no challenger)")
    parser.add_argument("--search-balance-mini-steps", type=positive_int, default=200,
                        help="With --search-balance, steps of each mini-train (default: %(default)s)")
    parser.add_argument("--search-balance-probe-games", type=positive_int, default=150,
                        help="With --search-balance, games of a probe refresh (default: %(default)s)")
    parser.add_argument("--search-balance-minigate-games", type=even_int, default=100,
                        help="With --search-balance, games of the mini-gate (default: %(default)s)")
    parser.add_argument("--search-balance-gpus", type=float, default=1.0,
                        help="With --search-balance, GPUs each stage holds, for GPU-second accounting "
                             "(default: %(default)s)")
    parser.add_argument("--search-balance-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="With --search-balance, where the mini nets are scored (default: cuda if torch "
                             "sees one)")
    parser.add_argument("--no-gate", action="store_true",
                        help="Skip the matches (and promotion): a smoke test of self-play and training")
    parser.add_argument("--label-check", choices=("on", "off"), default="on",
                        help="Check each self-play segment's value labels against its game lines "
                             "(hivemind.distill.labels) and fail the stage on a problem (default: on)")
    parser.add_argument("--allow-stock-engine", action="store_true",
                        help="With wave 2 options, an engine whose --help lacks them plays stock v3 "
                             "games instead of stopping (for a smoke test before the wave 2 engine)")
    parser.add_argument("--wide", action="store_true",
                        help="Wave 2b: the generator is a wide network (twin-s-clock-wide lineage, "
                             "`hivemind.wave2 init --wide`); seat timing and arrivals labels go into "
                             "the chunks and training adds the arrivals loss")
    parser.add_argument("--arrivals-weight", type=float, default=0.05,
                        help="With --wide, the weight of the arrivals loss (default 0.05)")
    parser.add_argument("--gate2", action="store_true",
                        help="Promote by the real-clock two-score gate (talisman cluster/gate2_match.sh on PACE, "
                             "hivemind.gate2) instead of the 100 ms match, which still plays (default: off)")
    parser.add_argument("--gate2-script", type=Path, default=None,
                        help="With --gate2, the matches script (default: $HIVEMIND_GATE2_SCRIPT, else "
                             "cluster/gate2_match.sh of the talisman repo above the fork)")
    parser.add_argument("--gate2-games", type=even_int, default=400,
                        help="With --gate2, games per network and arm (default: %(default)s)")
    parser.add_argument("--gate2-parts", type=positive_int, default=4,
                        help="With --gate2, jobs per network and arm (default: %(default)s)")
    parser.add_argument("--gate2-base-ms", type=positive_int, default=60000,
                        help="With --gate2, each seat's starting clock (default: %(default)s)")
    parser.add_argument("--gate2-free-k", type=positive_int, default=160,
                        help="With --gate2, our clock_k in the free arm, think head on (default: %(default)s)")
    parser.add_argument("--gate2-equal-k", type=positive_int, default=110,
                        help="With --gate2, our clock_k in the equal arm (default: %(default)s)")
    parser.add_argument("--gate2-seed-base", type=int, default=7000,
                        help="With --gate2, part k plays seed SEED_BASE + k (default: %(default)s)")
    parser.add_argument("--gate2-opp-sit", choices=("observed", "own_k"), default="observed",
                        help="With --gate2, how our side reads an opponent's sit (default: %(default)s)")
    parser.add_argument("--gate2-drop-margin", type=float, default=two_score.DROP_MARGIN,
                        help="With --gate2, a drop is d below minus this and clearly so (default: %(default)s)")
    parser.add_argument("--gate2-min-pairs", type=positive_int, default=two_score.MIN_PAIRS,
                        help="With --gate2, shared pairs an arm needs to count (default: %(default)s)")
    parser.add_argument("--gate2-poll-s", type=float, default=300.0,
                        help="With --gate2, seconds between calls of the matches script (default: %(default)s)")
    parser.add_argument("--gate2-timeout-h", type=float, default=10.0,
                        help="With --gate2, hours after which an unfinished gate does not promote "
                             "(default: %(default)s)")
    wave2.add_arguments(parser)
    # Spec v7 hooks: --league/--pace (hivemind.league) and --think-* (hivemind.think) add their own flags.
    modules = hook_modules(sys.argv[1:])
    module_dests = {}
    for name, module in modules.items():
        if module is not None:
            before = {action.dest for action in parser._actions}
            module.add_arguments(parser)
            module_dests[name] = [action.dest for action in parser._actions if action.dest not in before]
    args = parser.parse_args()
    try:
        args.wave2_options = wave2.from_args(args)
    except ValueError as error:
        parser.error(str(error))
    for module in modules.values():
        if module is not None:
            module.validate(args, parser)
    args.league_on = modules["league"] is not None and bool(modules["league"].active(args))
    args.think_on = modules["think"] is not None and any(
        getattr(args, dest) != parser.get_default(dest) for dest in module_dests.get("think", ()))
    args.label_check = args.label_check == "on"
    if args.wave2:
        args.lr = args.lr or "1e-4"
        args.trunk_lr = args.trunk_lr or "5e-5"
        args.train_from = args.train_from or "generator"
        if args.promote_above is None:
            args.promote_above = 0.5
        if args.time_model == "flag":
            args.time_model = "clock"  # the network reads lead and speed from the seat clocks
    args.lr = args.lr or "1e-4"
    args.train_from = args.train_from or "previous"
    for name in ("lr", "trunk_lr"):
        value = getattr(args, name)
        if value is not None and not 0 < float(value) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if not args.wave2_options.is_stock and args.distributed:
        parser.error("--distributed does not carry the wave 2 options to contributors yet "
                     "(on one cluster, use --shared-selfplay with --selfplay-worker jobs)")
    args.shared_selfplay = args.shared_selfplay or args.selfplay_worker
    if args.shared_selfplay and args.distributed:
        parser.error("--shared-selfplay and --distributed are two ways to share self-play: pick one")
    if args.selfplay_hook and not args.shared_selfplay:
        parser.error("--selfplay-hook needs --shared-selfplay")
    if not (0 < args.gate_sprt_alpha < 1 and 0 < args.gate_sprt_beta < 1):
        parser.error("--gate-sprt-alpha and --gate-sprt-beta must be in (0, 1)")
    if args.gate_max_games is not None and not args.gate_sprt:
        parser.error("--gate-max-games needs --gate-sprt")
    if not -1 < args.resign_threshold <= 0:
        parser.error("--resign-threshold must be in (-1, 0]")
    if args.new_passes <= 0 or args.replay_passes < 0:
        parser.error("--new-passes must be positive and --replay-passes non-negative")
    if args.last is not None and args.first is not None and args.last < args.first:
        parser.error("--last is before --first")
    if args.fixed_time_model is None:
        args.fixed_time_model = args.time_model
    if args.fixed_time_model == "clock" and args.time_model != "clock":
        parser.error("--fixed-time-model clock needs --time-model clock")
    if args.archive_only and args.keep_selfplay:
        parser.error("--archive-only and --keep-selfplay contradict each other")
    if args.segment_games is None:
        args.segment_games = 1000 if args.distributed else 10_000
    args.local_selfplay = not args.no_local_selfplay
    if args.search_balance:
        if args.distributed:
            parser.error("--search-balance plans local segments: use it without --distributed")
        if args.games % args.segment_games:
            parser.error("--search-balance needs --games to be a multiple of --segment-games "
                         "(the controller plans whole segments)")
        controller = search_balance_controller(args.search_balance_controller)
        if controller is None:
            parser.error(f"--search-balance: no controller at "
                         f"{args.search_balance_controller or 'cluster/search_balance.py above ' + str(PROJECT_ROOT)}")
        args.search_balance_controller = controller
        if args.search_balance_prior is not None:
            args.search_balance_prior = args.search_balance_prior.resolve()
            if not args.search_balance_prior.is_file():
                parser.error(f"--search-balance-prior {args.search_balance_prior} not found")
    elif any(getattr(args, name) != parser.get_default(name) for name in (
            "search_balance_controller", "search_balance_prior", "search_balance_mini_steps",
            "search_balance_probe_games", "search_balance_minigate_games", "search_balance_gpus",
            "search_balance_device")):
        parser.error("the --search-balance-* options need --search-balance")
    if args.league_on:
        if args.search_balance:
            parser.error("--league/--pace and --search-balance both plan the segments: pick one")
        if args.distributed:
            parser.error("--league/--pace plan local segments: use them without --distributed")
        if not args.wave2 or not args.wave2_options.base_mix:
            parser.error("--league/--pace need --wave2 and a base mix (real seat clocks)")
    if args.gate2:
        if args.no_gate:
            parser.error("--gate2 is a gate: it does not go with --no-gate")
        if args.gate2_games % (2 * args.gate2_parts):
            parser.error("--gate2-games must split into colour-swapped pairs over --gate2-parts")
        if args.gate2_min_pairs > args.gate2_games // 2:
            parser.error("--gate2-min-pairs is more pairs than --gate2-games plays")
        if not (args.gate2_drop_margin >= 0 and args.gate2_poll_s >= 0 and args.gate2_timeout_h > 0):
            parser.error("--gate2-drop-margin and --gate2-poll-s must be non-negative, --gate2-timeout-h positive")
        script = gate2_script(args.gate2_script)
        if script is None:
            parser.error(f"--gate2: no matches script at "
                         f"{args.gate2_script or 'cluster/gate2_match.sh above ' + str(PROJECT_ROOT)}")
        args.gate2_script = script
    elif any(getattr(args, name) != parser.get_default(name) for name in V7_DEFAULTS
             if name.startswith("gate2_")):
        parser.error("the --gate2-* options need --gate2")

    args.generator = args.generator.resolve()
    args.generator_checkpoint = args.generator_checkpoint.resolve()
    if not args.generator.is_file():
        sys.exit(f"Generator not found at {args.generator}; run `hivemind fetch-network --variant onnx` "
                 f"or pass --generator.")
    width = onnx_input_planes(args.generator)
    if args.wide != (width == NUM_WIDE_CHANNELS):
        sys.exit(f"{args.generator} reads {width} planes: "
                 + ("--wide needs a wide generator (`python -m hivemind.wave2 init --wide`)" if args.wide
                    else "a wide generator needs --wide"))
    if not args.archive_only:
        if not args.selfplay_engine.is_file():
            sys.exit(f"Self-play engine not found at {args.selfplay_engine}; build it with\n"
                     f"  {selfplay.BUILD_COMMAND}")
        if not args.match_engine.is_file():
            sys.exit(f"Match engine not found at {args.match_engine}; run `hivemind build-engine`.")
        if not args.openings.is_file():
            sys.exit(f"Openings not found at {args.openings}; pass --openings.")
    args.openings = args.openings.resolve()
    args.selfplay_engine = args.selfplay_engine.resolve()
    args.match_engine = args.match_engine.resolve()
    args.runs = args.runs.resolve()
    if not args.archive_only and not args.wave2_options.is_stock:
        missing = sorted({*wave2.missing_engine_options(args.wave2_options, engine_help(args.selfplay_engine)),
                          *(() if args.no_gate else wave2.missing_engine_options(
                              args.wave2_options, engine_help(args.match_engine)))})
        if missing:
            if not args.allow_stock_engine:
                sys.exit(f"The engines lack the wave 2 options {', '.join(missing)}: build the wave 2 "
                         f"engine (docs/wave2_training.md), or pass --allow-stock-engine for a stock smoke test.")
            print(f"wave 2 options {', '.join(missing)} missing from the engines: playing stock v3 games",
                  flush=True)
            args.wave2_options = wave2.STOCK

    loop = Loop(args, generator=args.generator, checkpoint=args.generator_checkpoint)
    if args.selfplay_worker:
        # No lock, no stop-file clearing, no archive or upload: those are the main loop's. The worker loads
        # its own copy of the generator, so its TensorRT cache is its own too.
        args.keep_selfplay, args.no_upload = True, True
        own = args.runs / "workers" / re.sub(r"[^A-Za-z0-9._-]", "_", loop.owner)
        own.mkdir(parents=True, exist_ok=True)
        loop.generator_sp1 = own / selfplay.one_worker_copy(args.generator).name
        loop.opponent_dir = own
        try:
            loop.work()
        finally:
            shutil.rmtree(own, ignore_errors=True)
        return
    skipped = unarchived = None
    if not args.no_upload:
        from huggingface_hub import HfApi, get_token
        loop.api = HfApi()
        skipped = upload_blocker(loop.api, args.repo, get_token())
        args.no_upload = skipped is not None
        if not args.keep_selfplay:
            unarchived = upload_blocker(loop.api, args.archive, get_token())
    elif not args.keep_selfplay:
        unarchived = "--no-upload"
    if unarchived:
        if args.archive_only:
            sys.exit(f"Cannot archive to {args.archive}: {unarchived}.")
        args.keep_selfplay = True

    import fcntl
    args.runs.mkdir(parents=True, exist_ok=True)
    lock = open(args.runs / "rl-loop.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another RL loop is running on {args.runs}.")
    stop = args.runs / "rl-loop.stop"
    if stop.exists():
        stop.unlink()
        loop.log(f"cleared an earlier stop request ({stop})")
    if skipped:
        loop.log(f"not uploading: {skipped}")
    if unarchived:
        loop.log(f"keeping self-play on disk: {unarchived}")
    if args.archive_only:
        loop.n = args.first or next_iteration(args.runs)
        loop.log(f"archiving the finished self-play of it1..it{loop.n} to {args.archive}")
        loop.archive(*range(1, loop.n + 1))
        return
    if args.distributed:
        loop.exchange = open_exchange(args.exchange, args.repo, loop.api)
        if isinstance(loop.exchange, HubExchange) and args.no_upload:
            sys.exit(f"--distributed on {args.exchange} needs uploads to Hugging Face: "
                     f"{skipped or 'drop --no-upload'}.")
        loop.exchange.create()
    loop.run(args.first or next_iteration(args.runs), args.last)


if __name__ == "__main__":
    main()
