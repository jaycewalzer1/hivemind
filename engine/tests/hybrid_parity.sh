#!/bin/sh
# Spec v6 acceptance test 0.3 (talisman docs/custom_engine_spec_v6_hybrid.md):
# no trigger, no change. On the 8 positions of stock_parity_positions.txt,
# with seat options under which the hybrid's trigger cannot fire (every seat
# of the other team sitting), `HybridMode gate` gives the same output as
# `HybridMode off` on this engine (nodur: DurativeTree false, SitGate true),
# every line compared after the wall-clock fields and the hybrid's own info
# strings are removed.
#
# Positive control (an always-empty trigger would pass the half above
# trivially): the same positions with a triggering setup (our lead -2 s, our
# seats 59 s and theirs 61 s, every pace 300, the other team's seats thinking
# for 200 ms, RootMoveTimeMs 200: an event 100 ms away) must print a
# non-empty `Hybrid: sit boards` line on at least 3 of the 8. Positions where
# our team is not on turn on exactly one board are listed as skipped there.
#
#   sh engine/tests/hybrid_parity.sh                 # build, compare
#   NODES=300 JOBS=2 sh engine/tests/hybrid_parity.sh
#
# Built like wave2_parity.sh, with ONE search worker into build-parity (the
# same build, so a run of either reuses the other's). A pair that differs is
# run again up to twice: position 8 (a mate-in-2 loss at ~100 nodes) is
# known to change from run to run on one binary (docs/wave2_engine.md), and a
# real difference repeats.
set -eu

ENGINE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ENGINE/.." && pwd)
JOBS=${JOBS:-2}
NODES=${NODES:-300}
ORT_ROOT=${ONNXRuntime_ROOT:-/opt/homebrew/opt/onnxruntime}
MODEL=${MODEL:-$ENGINE/models/twin-s-clock-fp32.onnx}
POSITIONS=${POSITIONS:-$ENGINE/tests/data/stock_parity_positions.txt}
OURS_BUILD=$ENGINE/build-parity
OUT=${OUT:-$ENGINE/build-parity/hybrid_parity}

echo "ours: $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (+ local changes)')"
echo "model: $MODEL"
cmake -S "$ENGINE" -B "$OURS_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DHIVEMIND_FAST_BUILD=OFF -DHIVEMIND_USE_UNITY_BUILD=ON -DHIVEMIND_USE_CCACHE=OFF \
    -DHIVEMIND_BACKEND=onnxruntime -DONNXRuntime_ROOT="$ORT_ROOT" \
    -DHIVEMIND_SEARCH_WORKERS=1 -DBUILD_TESTING=OFF >/dev/null
cmake --build "$OURS_BUILD" --target hivemind -j "$JOBS" >/dev/null

mkdir -p "$OUT"
export HIVEMIND_COREML=off
# run <mode> <team> <lead> <own> <opp> <root> <clocks> <paces> <elapsed> <sitting> <fen>
run() {
    mode=$1; team=$2; lead=$3; own=$4; opp=$5; root=$6; clocks=$7; paces=$8; elapsed=$9; shift 9
    sitting=$1; fen=$2
    {
        echo uci
        for opt in "Ponder false" "BackgroundSearch false" "MateProbe false" \
                   "ConditionalMateProbe false" "CertifySelectedMove false" \
                   "Team $team" "TimeModel clock" "TimeLeadMs $lead" \
                   "TeamMoveTimeMs $own" "OpponentMoveTimeMs $opp" "RootMoveTimeMs $root" \
                   "DurativeTree false" "SitGate true" \
                   "SeatClocksMs $clocks" "SeatPaceMs $paces" "SeatElapsedMs $elapsed" \
                   "SeatSitting $sitting" "HybridMode $mode"; do
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
    } | "$OURS_BUILD/hivemind" --model "$MODEL" 2>&1 \
      | grep -v -e '^option ' -e '^id ' -e 'backend ONNX Runtime' -e 'warning this network' \
                -e '^info string Hybrid' \
      | sed -E 's/ (time|nps|ms|hashfull) -?[0-9.]+//g; s/[0-9.]+ ?ms//g; s/_ms=-?[0-9]+//g; s/\([0-9.]+%\)//g' \
      | awk '/^info depth /{last=$0; next} /^bestmove/{if (last != "") print last; last=""} {print}'
}
# The raw output of a triggering search: only its first Hybrid line is read.
trigger_line() {
    mode=$1; team=$2; clocks=$3; elapsed=$4; fen=$5
    {
        echo uci
        for opt in "Ponder false" "BackgroundSearch false" "MateProbe false" \
                   "ConditionalMateProbe false" "CertifySelectedMove false" \
                   "Team $team" "TimeModel clock" "TimeLeadMs -2000" \
                   "TeamMoveTimeMs 300" "OpponentMoveTimeMs 300" "RootMoveTimeMs 200" \
                   "DurativeTree false" "SitGate true" \
                   "SeatClocksMs $clocks" "SeatPaceMs 300 300 300 300" "SeatElapsedMs $elapsed" \
                   "SeatSitting 0 0 0 0" "HybridMode $mode"; do
            name=${opt%% *}; value=${opt#* }
            echo "setoption name $name value $value"
        done
        echo isready
        echo ucinewgame
        echo "position fen $fen"
        echo "go nodes 64"
        sleep 4
        echo quit
    } | "$OURS_BUILD/hivemind" --model "$MODEL" 2>&1 | grep -m1 '^info string Hybrid: sit boards' || true
}

n=0; fail=0; fired=0; skipped=0
while IFS='	' read -r team lead own opp root fen; do
    case "$team" in ''|'#'*) continue ;; esac
    n=$((n + 1))
    # Parity half: the position's own clock, seat clocks from it, the other
    # team's seats sitting (T3 fails on any board).
    half=$((lead / 2))
    if [ "$team" = white ]; then w=$((60000 + half)); b=$((60000 - half)); wp=$own; bp=$opp; sit="0 1 1 0"
    else w=$((60000 - half)); b=$((60000 + half)); wp=$opp; bp=$own; sit="1 0 0 1"; fi
    same=0
    for attempt in 1 2 3; do
        run off "$team" "$lead" "$own" "$opp" "$root" "$w $b $b $w" "$wp $bp $bp $wp" \
            "50 50 50 50" "$sit" "$fen" > "$OUT/off_$n.txt"
        run gate "$team" "$lead" "$own" "$opp" "$root" "$w $b $b $w" "$wp $bp $bp $wp" \
            "50 50 50 50" "$sit" "$fen" > "$OUT/gate_$n.txt"
        if cmp -s "$OUT/off_$n.txt" "$OUT/gate_$n.txt"; then same=1; break; fi
    done
    if [ "$same" -eq 1 ]; then
        echo "position $n: identical ($(grep -c '' "$OUT/gate_$n.txt") lines, $(grep '^bestmove' "$OUT/gate_$n.txt"), attempt $attempt)"
    else
        echo "position $n: DIFFERENT"; diff "$OUT/off_$n.txt" "$OUT/gate_$n.txt" | head -20
        fail=$((fail + 1))
    fi
    # Positive control: our seats 59 s, theirs 61 s (lead -2 s), theirs thinking 200 ms.
    if [ "$team" = white ]; then clocks="59000 61000 61000 59000"; el="0 200 200 0"
    else clocks="61000 59000 59000 61000"; el="200 0 0 200"; fi
    sideA=$(echo "$fen" | cut -d'|' -f1 | awk '{print $2}')
    sideB=$(echo "$fen" | cut -d'|' -f2 | awk '{print $2}')
    onA=0; onB=0
    [ "$team" = white ] && [ "$sideA" = w ] && onA=1
    [ "$team" = black ] && [ "$sideA" = b ] && onA=1
    [ "$team" = white ] && [ "$sideB" = b ] && onB=1
    [ "$team" = black ] && [ "$sideB" = w ] && onB=1
    if [ $((onA + onB)) -ne 1 ]; then
        echo "  control $n: skipped (on turn on $((onA + onB)) boards)"
        skipped=$((skipped + 1))
        continue
    fi
    line=$(trigger_line gate "$team" "$clocks" "$el" "$fen")
    echo "  control $n: ${line:-no Hybrid line}"
    case "$line" in *"sit boards A"*|*"sit boards B"*) fired=$((fired + 1)) ;; esac
done < "$POSITIONS"
echo "$((n - fail))/$n positions identical with no trigger; trigger fired on $fired of $((n - skipped)) one-board positions ($skipped skipped) (outputs in $OUT)"
[ "$fail" -eq 0 ] && [ "$fired" -ge 3 ]
