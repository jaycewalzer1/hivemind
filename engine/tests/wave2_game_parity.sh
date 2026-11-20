#!/bin/sh
# Spec v5 acceptance test 0.6 (talisman docs/custom_engine_spec_v5_continuity.md):
# with every wave 2c option at its default, whole games and multi-search UCI
# sessions of this engine equal the wave 2 engine before them (261081b) bit
# for bit. wave2_parity.sh (0.4) runs one search per fresh process; this one
# exercises what wave 2c changed between searches and between games: the
# per-game resets in selfplay.cc and tournament.cc, the run_search entry
# block that drops team trees, and the end-of-search store/record path.
#
#   sh engine/tests/wave2_game_parity.sh             # build both, compare
#   BASE=261081b NODES=64 JOBS=2 sh engine/tests/wave2_game_parity.sh
#
# Both engines are built with ONE search worker, as wave2_parity.sh builds
# them (and into the same directories, so a run of either reuses the other's
# builds). Three checks, each run by both binaries:
#   selfplay   2 seat-clock games (headline wave 2 flags, 20 s, seed 7, no
#              Fairy-Stockfish mate probe): games.pgn less its Date tag and
#              seat_clocks.tsv less its `nodes` column must be identical.
#              Self-play is not bit-reproducible beyond that even on one
#              binary: a search's node count can overshoot its target by a
#              visit or so from run to run (worker timing), which moves the
#              `nodes` column and the visit-count policy targets of the
#              training records, so those are not compared. The mate probe
#              runs on its own thread and its proofs reach the tree at
#              wall-clock times, which can change a move: it is off here;
#   tournament 2 seat-clock games, the same network on both sides:
#              games.pgn less Date and games.tsv less the config line and the
#              two nanosecond columns;
#   uci        one process, no ucinewgame between searches: every position of
#              the positions file in turn under the wave 2 options, then a
#              ucinewgame and the first two again; every line compared after
#              the wall-clock fields are removed.
set -eu

ENGINE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ENGINE/.." && pwd)
BASE=${BASE:-261081b}
JOBS=${JOBS:-2}
NODES=${NODES:-64}
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
MODEL=${MODEL:-$ENGINE/models/twin-s-clock-fp32.onnx}
POSITIONS=${POSITIONS:-$ENGINE/tests/data/stock_parity_positions.txt}
OURS_BUILD=$ENGINE/build-parity
BASE_TREE=$ENGINE/build-wave2-base/src
BASE_BUILD=$ENGINE/build-wave2-base/build
OUT=${OUT:-$ENGINE/build-parity/wave2_game_parity}

if [ ! -d "$BASE_TREE" ]; then
    mkdir -p "$ENGINE/build-wave2-base"
    git -C "$REPO" worktree add --detach "$BASE_TREE" "$BASE" >/dev/null
fi
git -C "$BASE_TREE" checkout --detach -q "$BASE"
echo "base: $(git -C "$BASE_TREE" rev-parse --short HEAD) at $BASE_TREE"
echo "ours: $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (+ local changes)')"
echo "model: $MODEL"
build() {
    src=$1; dir=$2
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DHIVEMIND_FAST_BUILD=OFF -DHIVEMIND_USE_UNITY_BUILD=ON -DHIVEMIND_USE_CCACHE=OFF \
        -DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="$ORT_ROOT" \
        -DHIVEMIND_SEARCH_WORKERS=1 -DBUILD_TESTING=OFF >/dev/null
    cmake --build "$dir" --target hivemind -j "$JOBS" >/dev/null
}
build "$ENGINE" "$OURS_BUILD"
build "$BASE_TREE/engine" "$BASE_BUILD"

# A fresh output directory per run: nothing of an earlier run is resumed.
# The old one goes to the Trash (macOS), never rm.
if [ -d "$OUT" ]; then
    mv "$OUT" "$HOME/.Trash/wave2_game_parity.$(date +%s)"
fi
mkdir -p "$OUT"
# CPU provider: Core ML may pick different kernels between two sessions.
export HIVEMIND_COREML=off
WAVE2="--durative-tree true --sit-gate false --flag-loss true --instant-transfer false
       --base-mix 20000:1 --seat-move-times true"

selfplay() {
    bin=$1; dir=$2
    # shellcheck disable=SC2086
    "$bin" selfplay --model "$MODEL" --games 2 --nodes "$NODES" --seed 7 \
        --max-macro-plies 400 --fairy-stockfish-mate-nodes 0 --output "$dir" $WAVE2 \
        >"$dir.log" 2>&1
    grep -v '^\[Date ' "$dir/games.pgn" >"$dir.pgn"
    awk -F'\t' 'BEGIN{OFS="\t"} {$5=""; print}' "$dir/seat_clocks.tsv" >"$dir.clocks"
}
tournament() {
    bin=$1; dir=$2
    # shellcheck disable=SC2086
    "$bin" tournament --contender "$MODEL" --baseline "$MODEL" --games 2 --nodes "$NODES" \
        --seed 7 --max-macro-plies 400 --output "$dir" $WAVE2 >"$dir.log" 2>&1
    grep -v '^\[Date ' "$dir/games.pgn" >"$dir.pgn"
    grep -v '^config' "$dir/games.tsv" | awk -F'\t' 'BEGIN{OFS="\t"} {$8=""; $11=""; print}' >"$dir.tsv"
}
uci() {
    bin=$1
    {
        echo uci
        for opt in "Ponder false" "BackgroundSearch false" "MateProbe false" \
                   "ConditionalMateProbe false" "CertifySelectedMove false" \
                   "TimeModel clock" "DurativeTree true" "SitGate false" "FlagLoss true" \
                   "InstantTransfer false" "MateRaceByLanding false" "PassCapK 8" \
                   "SeatPaceMs 300 300 300 300" "SeatElapsedMs 50 50 50 50"; do
            name=${opt%% *}; value=${opt#* }
            echo "setoption name $name value $value"
        done
        echo isready
        echo ucinewgame
        while IFS='	' read -r team lead own opp root fen; do
            case "$team" in ''|'#'*) continue ;; esac
            half=$((lead / 2))
            if [ "$team" = white ]; then w=$((60000 + half)); b=$((60000 - half))
            else w=$((60000 - half)); b=$((60000 + half)); fi
            echo "setoption name Team value $team"
            echo "setoption name RootMoveTimeMs value $root"
            echo "setoption name SeatClocksMs value $w $b $b $w"
            echo "position fen $fen"
            echo "go nodes $NODES"
            sleep 4
        done < "$POSITIONS"
        echo ucinewgame
        # After the ucinewgame: the first two positions again.
        grep -v '^#' "$POSITIONS" | head -2 | while IFS='	' read -r team lead own opp root fen; do
            echo "setoption name Team value $team"
            echo "setoption name RootMoveTimeMs value $root"
            echo "position fen $fen"
            echo "go nodes $NODES"
            sleep 4
        done
        echo quit
    } | "$bin" --model "$MODEL" 2>&1 \
      | grep -v -e '^option ' -e '^id ' -e 'backend ONNX Runtime' -e 'warning this network' \
      | sed -E 's/ (time|nps|ms|hashfull) -?[0-9.]+//g; s/[0-9.]+ ?ms//g; s/_ms=-?[0-9]+//g; s/\([0-9.]+%\)//g' \
      | awk '/^info depth /{last=$0; next} /^bestmove/{if (last != "") print last; last=""} {print}'
}

fail=0
same() {
    label=$1; a=$2; b=$3
    if cmp -s "$a" "$b"; then
        echo "$label: identical ($(wc -c <"$a" | tr -d ' ') bytes)"
    else
        echo "$label: DIFFERENT ($a vs $b)"
        diff "$a" "$b" | head -20
        fail=$((fail + 1))
    fi
}

selfplay "$BASE_BUILD/hivemind" "$OUT/selfplay_base"
selfplay "$OURS_BUILD/hivemind" "$OUT/selfplay_ours"
for kind in pgn clocks; do
    same "selfplay $kind" "$OUT/selfplay_base.$kind" "$OUT/selfplay_ours.$kind"
done
tournament "$BASE_BUILD/hivemind" "$OUT/tournament_base"
tournament "$OURS_BUILD/hivemind" "$OUT/tournament_ours"
for kind in pgn tsv; do
    same "tournament $kind" "$OUT/tournament_base.$kind" "$OUT/tournament_ours.$kind"
done
uci "$BASE_BUILD/hivemind" >"$OUT/uci_base.txt"
uci "$OURS_BUILD/hivemind" >"$OUT/uci_ours.txt"
same "uci session ($(grep -c '^bestmove' "$OUT/uci_ours.txt") searches)" \
    "$OUT/uci_base.txt" "$OUT/uci_ours.txt"
echo "$((5 - fail))/5 identical (outputs in $OUT)"
[ "$fail" -eq 0 ]
