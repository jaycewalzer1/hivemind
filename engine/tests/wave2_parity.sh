#!/bin/sh
# Spec v5 acceptance test 0.4 (talisman docs/custom_engine_spec_v5_continuity.md):
# with every wave 2c option at its default this engine's output equals the
# wave 2 engine before them (261081b) bit for bit, under the headline wave 2
# options (durative tree, no sit gate, flag loss, delayed transfer) with
# per-seat clocks, paces, elapsed times and recent landings.
#
#   sh engine/tests/wave2_parity.sh                  # build both, compare
#   BASE=261081b NODES=300 JOBS=2 sh engine/tests/wave2_parity.sh
#
# Like stock_parity.sh: both engines are built with ONE search worker, the
# base commit from a git worktree under build-wave2-base (kept for the next
# run), and every line of `policy` and `go nodes N` is compared after the
# wall-clock fields are removed. Each line of the positions file gives the
# seat clocks: 60 s for every seat, moved by half the team lead each way;
# paces are the two move times; the other team's seats have thought 50 ms;
# one pawn reached Ab's pocket 200 ms ago.
set -eu

ENGINE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ENGINE/.." && pwd)
BASE=${BASE:-261081b}
JOBS=${JOBS:-2}
NODES=${NODES:-300}
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
MODEL=${MODEL:-$ENGINE/models/twin-s-clock-fp32.onnx}
POSITIONS=${POSITIONS:-$ENGINE/tests/data/stock_parity_positions.txt}
OURS_BUILD=$ENGINE/build-parity
BASE_TREE=$ENGINE/build-wave2-base/src
BASE_BUILD=$ENGINE/build-wave2-base/build
OUT=${OUT:-$ENGINE/build-parity/wave2_parity}

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

mkdir -p "$OUT"
# CPU provider: Core ML may pick different kernels between two sessions.
export HIVEMIND_COREML=off
run() {
    bin=$1; team=$2; lead=$3; own=$4; opp=$5; root=$6; fen=$7
    half=$((lead / 2))
    if [ "$team" = white ]; then w=$((60000 + half)); b=$((60000 - half)); wp=$own; bp=$opp
    else w=$((60000 - half)); b=$((60000 + half)); wp=$opp; bp=$own; fi
    {
        echo uci
        for opt in "Ponder false" "BackgroundSearch false" "MateProbe false" \
                   "ConditionalMateProbe false" "CertifySelectedMove false" \
                   "Team $team" "TimeModel clock" "TimeLeadMs $lead" \
                   "TeamMoveTimeMs $own" "OpponentMoveTimeMs $opp" "RootMoveTimeMs $root" \
                   "DurativeTree true" "SitGate false" "FlagLoss true" \
                   "InstantTransfer false" "MateRaceByLanding false" "PassCapK 8" \
                   "SeatClocksMs $w $b $b $w" "SeatPaceMs $wp $bp $bp $wp" \
                   "SeatElapsedMs 50 50 50 50" "SeatArrivalsMs Ab:P:200"; do
            name=${opt%% *}; value=${opt#* }
            echo "setoption name $name value $value"
        done
        echo isready
        echo ucinewgame
        echo "position fen $fen"
        echo policy
        echo "go nodes $NODES"
        sleep 8
        echo quit
    } | "$bin" --model "$MODEL" 2>&1 \
      | grep -v -e '^option ' -e '^id ' -e 'backend ONNX Runtime' -e 'warning this network' \
      | sed -E 's/ (time|nps|ms|hashfull) -?[0-9.]+//g; s/[0-9.]+ ?ms//g; s/_ms=-?[0-9]+//g; s/\([0-9.]+%\)//g' \
      | awk '/^info depth /{last=$0; next} /^bestmove/{if (last != "") print last; last=""} {print}'
}

n=0; fail=0
while IFS='	' read -r team lead own opp root fen; do
    case "$team" in ''|'#'*) continue ;; esac
    n=$((n + 1))
    run "$OURS_BUILD/hivemind" "$team" "$lead" "$own" "$opp" "$root" "$fen" > "$OUT/ours_$n.txt"
    run "$BASE_BUILD/hivemind" "$team" "$lead" "$own" "$opp" "$root" "$fen" > "$OUT/base_$n.txt"
    if cmp -s "$OUT/ours_$n.txt" "$OUT/base_$n.txt"; then
        echo "position $n: identical ($(grep -c '' "$OUT/ours_$n.txt") lines, $(grep '^bestmove' "$OUT/ours_$n.txt"))"
    else
        echo "position $n: DIFFERENT"; diff "$OUT/base_$n.txt" "$OUT/ours_$n.txt" | head -20
        fail=$((fail + 1))
    fi
done < "$POSITIONS"
echo "$((n - fail))/$n positions identical (outputs in $OUT)"
[ "$fail" -eq 0 ]
