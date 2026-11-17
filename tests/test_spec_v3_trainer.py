"""Spec v3 WP-D trainer: two learning rates (trunk vs new input columns and aux heads), batch/lr consistency,
sample weights in the batch, arrivals loss fixes, the clock recipe, and the drift metric."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hivemind.architectures.model_config import create_model, get_model_args
from hivemind.config.train_config import rl_clock_train_config, rl_train_config
from hivemind.training.train_loop import configure_batch_size
from hivemind.training.trainer_agent import (
    aux_head_losses,
    create_optimizer,
    drift_metrics,
    drift_outputs,
    finetune_param_groups,
    narrow_board_planes,
    new_columns_max_abs,
    new_input_columns,
    split_rl_batch_extras,
    widen_board_planes,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool():
    spec = importlib.util.spec_from_file_location("make_clock_net_v3t", TOOLS / "make_clock_net.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_clock_net_v3t"] = module
    spec.loader.exec_module(module)
    return module


def _model(channels, aux, seed=0):
    torch.manual_seed(seed)
    args = get_model_args(architecture="crossboard-risev33", input_channels=channels, aux_heads=aux)
    args.use_wdl = args.use_plys_to_end = True
    return create_model(args)


@pytest.fixture(scope="module")
def widened():
    """(it04-shaped source model, the widened 86-channel model with aux heads, as make_clock_net builds it)."""
    tool = _tool()
    old = _model(74, aux=False).eval()
    ckpt = tool.widen_checkpoint({"model_state_dict": old.state_dict()}, 43)
    new = tool.load_model(ckpt["model_state_dict"]).eval()
    return old, new


def test_new_columns_cover_every_widened_tensor(widened):
    old, new = widened
    old_sd, new_sd = old.state_dict(), new.state_dict()
    masks = new_input_columns(new, 37)
    changed = {k for k in old_sd if old_sd[k].shape != new_sd[k].shape}
    assert changed and changed == set(masks)
    for name, mask in masks.items():
        w = new_sd[name]
        assert not w[:, mask].any()  # the new columns start at exactly 0
        assert torch.equal(w[:, ~mask], old_sd[name])  # the old columns, in order, are the source weights
    assert new_input_columns(old, 37) == {}


def test_param_groups_partition_the_network(widened):
    _, new = widened
    tc = rl_clock_train_config()
    groups = {g["name"]: g for g in finetune_param_groups(new, tc)}
    names = [n for g in groups.values() for n in g["param_names"]]
    assert sorted(names) == sorted(n for n, _ in new.named_parameters()) and len(names) == len(set(names))
    assert all(n.startswith(("think_gain_head.", "arrivals_head.")) for n in groups["new"]["param_names"])
    assert set(groups["widened"]["param_names"]) == set(new_input_columns(new, 37))
    assert tc.max_lr * groups["trunk"]["lr_scale"] == pytest.approx(1e-4)
    assert groups["new"]["lr_scale"] == 1.0 and groups["widened"]["lr_scale"] == 10.0
    assert groups["widened"]["weight_decay"] == 0.0


def test_column_hook_equals_a_per_column_learning_rate():
    class Net(torch.nn.Module):
        board_channels = 3

        def __init__(self):
            super().__init__()
            self.shared_stem = torch.nn.Module()
            self.shared_stem.body = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1, bias=False))
            self.head = torch.nn.Linear(2, 1)

    net = Net()
    tc = rl_train_config()
    tc.finetune_param_groups, tc.source_board_channels, tc.trunk_max_lr = True, 2, tc.max_lr / 10
    w0 = net.shared_stem.body[0].weight.detach().clone()
    opt = create_optimizer(net, tc)
    x = torch.randn(4, 3, 1, 1)
    loss = net.head(net.shared_stem.body(x).flatten(1)).pow(2).sum()
    loss.backward()
    # gradient without the hook, for the manual update
    net2 = Net()
    net2.load_state_dict(net.state_dict())
    with torch.no_grad():
        net2.shared_stem.body[0].weight.copy_(w0)
    net2.head(net2.shared_stem.body(x).flatten(1)).pow(2).sum().backward()
    g = net2.shared_stem.body[0].weight.grad
    for group in opt.param_groups:
        group["lr"] = tc.max_lr * group["lr_scale"]
    opt.step()
    lr_col = torch.tensor([tc.max_lr / 10, tc.max_lr / 10, tc.max_lr]).view(1, 3, 1, 1)
    expect = w0 - lr_col * (g + tc.wd * w0)
    torch.testing.assert_close(net.shared_stem.body[0].weight.detach(), expect, rtol=1e-6, atol=1e-8)


def test_adam_is_refused_with_param_groups(widened):
    tc = rl_clock_train_config()
    tc.optimizer_name = "adam"
    with pytest.raises(ValueError, match="SGD"):
        create_optimizer(widened[1], tc)


def test_batch_lr_consistency():
    tc = rl_clock_train_config()
    lr = tc.max_lr
    configure_batch_size(tc, "crossboard-risev33")
    assert tc.batch_size == 256 and tc.max_lr == pytest.approx(lr / 2) and tc.trunk_max_lr == pytest.approx(5e-5)
    stock = rl_train_config()
    configure_batch_size(stock, "crossboard-risev33")
    assert stock.max_lr == lr  # stock recipe unchanged


def test_clock_recipe_defaults():
    tc = rl_clock_train_config()
    assert tc.plys_to_end_loss_factor == 0.0 and tc.q_value_ratio == 0.2
    assert tc.arrivals_piece_weights == (1.0, 2.0, 2.0, 2.0, 4.0) and tc.arrivals_loss == "log1p"
    assert tc.sample_weight_targets == "policy" and tc.finetune_param_groups and tc.freeze_batchnorm_stats
    stock = rl_train_config()
    assert not stock.finetune_param_groups and stock.sample_weight_targets == "none"
    assert stock.arrivals_loss == "mse"


def test_batch_split_with_sample_weights():
    base = tuple(torch.zeros(2) for _ in range(6))
    aux = tuple(torch.full((2,), float(i)) for i in range(4))
    w = torch.tensor([0.5, 1.0])
    joint, labels, weights = split_rl_batch_extras(base + aux + (w,), True, True, with_weights=True)
    assert joint is None and len(labels) == 4 and torch.equal(weights, w)
    joint, labels = split_rl_batch_extras(base + aux + (w,), True, True)
    assert float(labels[-1][0]) == 3.0
    assert split_rl_batch_extras(base + aux, True)[1][0][0] == 0.0  # the v2 call is unchanged


def test_arrivals_losses():
    pred = torch.zeros(2, 20)
    label = torch.zeros(2, 20)
    label[0, 4] = 3.0  # three queens to ours-A
    mask = torch.tensor([1.0, 0.0])
    think = (torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2))
    _, _, mse, n = aux_head_losses(think[0], pred, think[1], think[2], label, mask)
    assert float(mse) == pytest.approx(9.0 / 20) and float(n) == 1.0
    _, _, l1p, _ = aux_head_losses(think[0], pred, think[1], think[2], label, mask, arrivals_loss="log1p")
    assert float(l1p) == pytest.approx(torch.log1p(torch.tensor(3.0)).item() ** 2 / 20)
    w = (1.0, 2.0, 2.0, 2.0, 4.0)
    _, _, wl, _ = aux_head_losses(think[0], pred, think[1], think[2], label, mask, arrivals_loss="log1p",
                                  piece_weights=w)
    assert float(wl) == pytest.approx(float(l1p) * 4.0 / (11.0 / 5.0))  # Q weight / mean weight
    _, _, pois, _ = aux_head_losses(think[0], pred, think[1], think[2], label, mask, arrivals_loss="poisson")
    assert float(pois) == pytest.approx(1.0)  # exp(0) - k * 0 averaged over 20 values
    with pytest.raises(ValueError):
        aux_head_losses(think[0], pred, think[1], think[2], label, mask, arrivals_loss="bogus")


def test_drift_is_zero_for_the_widened_start(widened):
    old, new = widened
    torch.manual_seed(3)
    x74 = torch.rand(6, 74, 8, 8)
    x86 = widen_board_planes(x74, 43)
    assert torch.equal(narrow_board_planes(x86, 37), x74)
    ref = drift_outputs(old, x86, torch.device("cpu"))
    cur = drift_outputs(new, x86, torch.device("cpu"))
    d = drift_metrics(ref, cur)
    assert d["policy_kl"] < 1e-6 and d["value_mse"] < 1e-10 and d["positions"] == 6
    assert new_columns_max_abs(new) == 0.0


def test_frozen_batchnorm_stats_do_not_move(widened):
    from hivemind.training.trainer_agent import set_train_mode
    _, new = widened
    tc = rl_clock_train_config()
    bn = next(m for m in new.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    set_train_mode(new, tc)
    assert new.training and not bn.training
    with torch.no_grad():
        new(torch.rand(4, 86, 8, 8))
    assert torch.equal(bn.running_mean, before)
    set_train_mode(new, rl_train_config())
    assert bn.training
    new.eval()
