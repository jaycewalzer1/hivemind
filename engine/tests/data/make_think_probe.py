"""Writes think_probe.onnx, the spec v7 think_gain test network (engine/tests/test_think_policy.cc).

uniform.onnx (value 0, uniform policies) plus a constant `think_gain` output,
logits [-ln 2, ln 3] for every row: probabilities 1/3 and 3/4. Batch 1.

Run from the repo root: uv run python engine/tests/data/make_think_probe.py
"""

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LOGITS = [-math.log(2.0), math.log(3.0)]


def main() -> None:
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 78, 8, 8])
    inits = [
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "value_shape"),
        numpy_helper.from_array(np.array([1, 4672], dtype=np.int64), "policy_shape"),
        numpy_helper.from_array(np.array([LOGITS], dtype=np.float32), "think_logits"),
    ]
    nodes = [
        helper.make_node("ConstantOfShape", ["value_shape"], ["value"]),
        helper.make_node("ConstantOfShape", ["policy_shape"], ["pi_a"]),
        helper.make_node("Identity", ["pi_a"], ["pi_b"]),
        helper.make_node("Identity", ["think_logits"], ["think_gain"]),
    ]
    outputs = [
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("pi_a", TensorProto.FLOAT, [1, 4672]),
        helper.make_tensor_value_info("pi_b", TensorProto.FLOAT, [1, 4672]),
        helper.make_tensor_value_info("think_gain", TensorProto.FLOAT, [1, 2]),
    ]
    graph = helper.make_graph(nodes, "think_probe", [obs], outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name("think_probe.onnx"))


if __name__ == "__main__":
    main()
