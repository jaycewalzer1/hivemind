#!/bin/sh
# Spec v4 acceptance test 7, binary half: with every wave 2 option at its
# default this engine's output equals stock aminwoo/hivemind v3.0.3 bit for bit.
#
#   sh engine/tests/stock_parity.sh                  # build both, compare
#   STOCK_SRC=/path/to/upstream-hivemind sh engine/tests/stock_parity.sh
#   NODES=400 JOBS=2 sh engine/tests/stock_parity.sh
#
# Both engines are built with ONE search worker (HIVEMIND_SEARCH_WORKERS=1) so
# a search is deterministic, into build dirs under this engine (stock is built
# out of tree; its source is only read). For each position and clock in
# tests/data/stock_parity_positions.txt both run `policy` and `go nodes N` on
# the twin-s-clock network; every line is compared after removing wall-clock
# fields (time, nps, ms, percentages of thinking time). Of the "info depth"
# progress lines only the last before bestmove is kept: the others are printed
# when the reporting thread polls, which is wall-clock timing (two stock runs
# differ there too). The final line carries the whole search: nodes, score
# and PV. The time-boxed root mate scans are switched off in both, since
# their result depends on wall time, not on the code.
set -eu

ENGINE=$(cd "$(dirname "$0")/.." && pwd)
STOCK_SRC=${STOCK_SRC:-$ENGINE/../../upstream-hivemind}
STOCK_SRC=$(cd "$STOCK_SRC" && pwd)
JOBS=${JOBS:-2}
NODES=${NODES:-300}
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
MODEL=${MODEL:-$STOCK_SRC/engine/models/twin-s-clock-fp32.onnx}
POSITIONS=${POSITIONS:-$ENGINE/tests/data/stock_parity_positions.txt}
OURS_BUILD=$ENGINE/build-parity
STOCK_BUILD=$ENGINE/build-parity-stock
OUT=${OUT:-$ENGINE/build-parity/parity}

echo "stock: $(git -C "$STOCK_SRC" describe --tags 2>/dev/null || echo '?') at $STOCK_SRC"
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
build "$STOCK_SRC/engine" "$STOCK_BUILD"

mkdir -p "$OUT"
# CPU provider: Core ML may pick different kernels between two sessions.
export HIVEMIND_COREML=off
run() {
    bin=$1; team=$2; lead=$3; own=$4; opp=$5; root=$6; fen=$7
    {
        echo uci
        for opt in "Ponder false" "BackgroundSearch false" "MateProbe false" \
                   "ConditionalMateProbe false" "CertifySelectedMove false" \
                   "Team $team" "TimeModel clock" "TimeLeadMs $lead" \
                   "TeamMoveTimeMs $own" "OpponentMoveTimeMs $opp" "RootMoveTimeMs $root"; do
            name=${opt% *}; value=${opt##* }
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
    run "$STOCK_BUILD/hivemind" "$team" "$lead" "$own" "$opp" "$root" "$fen" > "$OUT/stock_$n.txt"
    if cmp -s "$OUT/ours_$n.txt" "$OUT/stock_$n.txt"; then
        echo "position $n: identical ($(grep -c '' "$OUT/ours_$n.txt") lines, $(grep '^bestmove' "$OUT/ours_$n.txt"))"
    else
        echo "position $n: DIFFERENT"; diff "$OUT/stock_$n.txt" "$OUT/ours_$n.txt" | head -20
        fail=$((fail + 1))
    fi
done < "$POSITIONS"
echo "$((n - fail))/$n positions identical (outputs in $OUT)"
[ "$fail" -eq 0 ]
