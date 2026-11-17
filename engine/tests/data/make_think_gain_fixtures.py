"""Writes the tiny ONNX fixtures test_custom_engine.cc loads (spec v2 B and C).

Each graph takes `data` [batch, C, 8, 8] and has the stock heads (value, pi_a,
pi_b, wdl_out, moves_left, all zero-valued) plus the spec v2 heads:
  think_gain [batch, 2]: logits (mean of channel 40, mean of channel C-1)
  arrivals   [batch, 20]: zeros (the engine must tolerate and ignore it)
With C = 88 (spec v3) the second logit reads board B's diagonal plane, with
C = 86 board B's last elapsed plane; with C = 82
(a v1 clock network) it reads board B's last clock plane, which checks the
engine packs 43-channel planes into the 41-channel layout correctly.

plain_<C>.onnx: the same graph with the stock heads only.

Run: python make_think_gain_fixtures.py  (needs the onnx package)
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make(channels: int, path: str, extra_heads: bool = True) -> None:
    nodes = []
    inits = []

    def const(name, array):
        inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    # zero [batch, 1] derived from the input so every head has a batch axis
    nodes.append(helper.make_node("ReduceSum", ["data", const("all_axes", np.array([1, 2, 3], np.int64))],
                                  ["sum"], keepdims=0))
    nodes.append(helper.make_node("Unsqueeze", ["sum", const("ax1", np.array([1], np.int64))], ["sum1"]))
    nodes.append(helper.make_node("Mul", ["sum1", const("zero", np.array(0.0, np.float32))], ["zero_b1"]))

    def broadcast(out, width):
        nodes.append(helper.make_node("Add", ["zero_b1", const(out + "_c", np.zeros((1, width), np.float32))], [out]))

    nodes.append(helper.make_node("Identity", ["zero_b1"], ["value"]))
    broadcast("pi_a", 4672)
    broadcast("pi_b", 4672)
    broadcast("wdl_out", 3)
    nodes.append(helper.make_node("Identity", ["zero_b1"], ["moves_left"]))
    if extra_heads:
        broadcast("arrivals", 20)

    def channel_mean(out, channel):
        nodes.append(helper.make_node(
            "Slice", ["data", const(out + "_s", np.array([channel], np.int64)),
                      const(out + "_e", np.array([channel + 1], np.int64)),
                      const(out + "_a", np.array([1], np.int64))], [out + "_slice"]))
        nodes.append(helper.make_node("ReduceMean", [out + "_slice", const(out + "_axes", np.array([2, 3], np.int64))],
                                      [out], keepdims=0))  # [batch, 1]

    if extra_heads:
        channel_mean("tg_a", 40)
        channel_mean("tg_b", channels - 1)
        nodes.append(helper.make_node("Concat", ["tg_a", "tg_b"], ["think_gain"], axis=1))

    def out(name, width):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width])

    graph = helper.make_graph(
        nodes, "think_gain_fixture",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", channels, 8, 8])],
        [out("value", 1), out("pi_a", 4672), out("pi_b", 4672), out("wdl_out", 3),
         out("moves_left", 1)] + ([out("think_gain", 2), out("arrivals", 20)] if extra_heads else []),
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for channels in (74, 82, 86, 88):
        make(channels, os.path.join(here, f"think_gain_{channels}.onnx"))
    for channels in (74, 86, 88):
        make(channels, os.path.join(here, f"plain_{channels}.onnx"), extra_heads=False)
