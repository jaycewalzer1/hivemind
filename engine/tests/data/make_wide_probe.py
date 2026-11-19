"""Writes wide_probe.onnx, the wave 2b test network (engine/tests/test_wave2b.cc).

It takes the 130-plane wide input and has the six heads the engine reads.
value, pi_a, pi_b, wdl_out and moves_left are 0; `arrivals` echoes 20 of the
wide planes (square a1), so a test can see exactly what reached the network:
for each board +0..+6, +11, +16 and +21 of its extra planes.

Run from the repo root: uv run python engine/tests/data/make_wide_probe.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PER_BOARD = 39
EXTRA = 26
WIDE_PER_BOARD = PER_BOARD + EXTRA
OFFSETS = [0, 1, 2, 3, 4, 5, 6, 11, 16, 21]
CHANNELS = [board * WIDE_PER_BOARD + PER_BOARD + offset for board in (0, 1) for offset in OFFSETS]


def main() -> None:
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", 2 * WIDE_PER_BOARD, 8, 8])
    inits = [
        numpy_helper.from_array(np.array(CHANNELS, dtype=np.int64), "channels"),
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "hw_axes"),
        numpy_helper.from_array(np.array([0, len(CHANNELS)], dtype=np.int64), "flat"),
    ]
    nodes = [
        helper.make_node("Gather", ["data", "channels"], ["picked"], axis=1),
        helper.make_node("Slice", ["picked", "starts", "ends", "hw_axes"], ["corner"]),
        helper.make_node("Reshape", ["corner", "flat"], ["arrivals"]),
    ]
    outputs = [helper.make_tensor_value_info("arrivals", TensorProto.FLOAT, ["batch", len(CHANNELS)])]
    for name, width in [("value", 1), ("pi_a", 4672), ("pi_b", 4672), ("wdl_out", 3), ("moves_left", 1)]:
        inits.append(numpy_helper.from_array(np.array([len(CHANNELS), width], dtype=np.int64), name + "_shape"))
        nodes.append(helper.make_node("ConstantOfShape", [name + "_shape"], [name + "_w"],
                                      value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])))
        nodes.append(helper.make_node("MatMul", ["arrivals", name + "_w"], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width]))
    graph = helper.make_graph(nodes, "wide_probe", [data], outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name("wide_probe.onnx"))


if __name__ == "__main__":
    main()
