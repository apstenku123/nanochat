import sys
import types

import torch

import nanochat.common as common


def _clear_dist_env(monkeypatch):
    for key in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "PJRT_DEVICE",
        "TPU_ACCELERATOR_TYPE",
        "ACCELERATOR_TYPE",
    ]:
        monkeypatch.delenv(key, raising=False)


def _install_fake_xla_runtime(monkeypatch, *, world_size, rank, local_rank):
    torch_xla_mod = types.ModuleType("torch_xla")
    runtime_mod = types.ModuleType("torch_xla.runtime")
    runtime_mod.world_size = lambda: world_size
    runtime_mod.global_ordinal = lambda: rank
    runtime_mod.local_ordinal = lambda: local_rank

    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla_mod)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime_mod)


def _install_fake_xla_model(monkeypatch, *, world_size, rank, local_rank):
    torch_xla_mod = types.ModuleType("torch_xla")
    core_mod = types.ModuleType("torch_xla.core")
    xla_model_mod = types.ModuleType("torch_xla.core.xla_model")
    xla_model_mod.xrt_world_size = lambda: world_size
    xla_model_mod.get_ordinal = lambda: rank
    xla_model_mod.get_local_ordinal = lambda: local_rank

    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla_mod)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core_mod)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model_mod)


def test_get_dist_info_prefers_initialized_torch_distributed(monkeypatch):
    _clear_dist_env(monkeypatch)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(common.dist, "get_world_size", lambda: 8)

    ddp, rank, local_rank, world_size = common.get_dist_info()

    assert ddp is True
    assert rank == 3
    assert local_rank == 1
    assert world_size == 8


def test_get_dist_info_uses_torchrun_env_when_no_xla(monkeypatch):
    _clear_dist_env(monkeypatch)
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    ddp, rank, local_rank, world_size = common.get_dist_info()

    assert ddp is True
    assert rank == 2
    assert local_rank == 0
    assert world_size == 4


def test_get_dist_info_uses_xla_runtime_when_tpu_requested(monkeypatch):
    _clear_dist_env(monkeypatch)
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    _install_fake_xla_runtime(monkeypatch, world_size=8, rank=2, local_rank=2)

    ddp, rank, local_rank, world_size = common.get_dist_info()

    assert ddp is False
    assert rank == 2
    assert local_rank == 2
    assert world_size == 8


def test_get_dist_info_falls_back_to_legacy_xla_model_api(monkeypatch):
    _clear_dist_env(monkeypatch)
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.delitem(sys.modules, "torch_xla.runtime", raising=False)
    _install_fake_xla_model(monkeypatch, world_size=4, rank=1, local_rank=1)

    ddp, rank, local_rank, world_size = common.get_dist_info()

    assert ddp is False
    assert rank == 1
    assert local_rank == 1
    assert world_size == 4


def test_get_tpu_accelerator_type_prefers_env(monkeypatch):
    _clear_dist_env(monkeypatch)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-4")

    assert common.get_tpu_accelerator_type() == "v6e-4"


def test_get_tpu_accelerator_type_reads_metadata(monkeypatch):
    _clear_dist_env(monkeypatch)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"v5litepod-8"

    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())

    value = common.get_tpu_accelerator_type()

    assert value == "v5litepod-8"
    assert common.os.environ["TPU_ACCELERATOR_TYPE"] == "v5litepod-8"


def test_xla_all_reduce_gradients_noop_for_world_size_one(monkeypatch):
    _clear_dist_env(monkeypatch)
    model = torch.nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = model(x).sum()
    y.backward()

    called = {"value": False}

    xla_model_mod = types.ModuleType("torch_xla.core.xla_model")
    xla_model_mod.REDUCE_SUM = "sum"

    def _all_reduce(*args, **kwargs):
        called["value"] = True

    xla_model_mod.all_reduce = _all_reduce
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model_mod)

    common.xla_all_reduce_gradients(model, world_size=1)

    assert called["value"] is False


def test_xla_all_reduce_gradients_averages(monkeypatch):
    _clear_dist_env(monkeypatch)
    model = torch.nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = model(x).sum()
    y.backward()

    captured = {}

    torch_xla_mod = types.ModuleType("torch_xla")
    core_mod = types.ModuleType("torch_xla.core")
    xla_model_mod = types.ModuleType("torch_xla.core.xla_model")
    xla_model_mod.REDUCE_SUM = "sum"

    def _all_reduce(reduce_type, grads, scale):
        captured["reduce_type"] = reduce_type
        captured["num_grads"] = len(grads)
        captured["scale"] = scale

    xla_model_mod.all_reduce = _all_reduce
    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla_mod)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core_mod)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model_mod)

    common.xla_all_reduce_gradients(model, world_size=4)

    assert captured["reduce_type"] == "sum"
    assert captured["num_grads"] > 0
    assert captured["scale"] == 0.25


def test_print0_suppresses_nonzero_rank(monkeypatch, capsys):
    _clear_dist_env(monkeypatch)
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(common.dist, "get_world_size", lambda: 2)

    common.print0("hidden")
    captured = capsys.readouterr()

    assert captured.out == ""


def test_print0_prints_rank_zero(monkeypatch, capsys):
    _clear_dist_env(monkeypatch)
    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(common.dist, "get_world_size", lambda: 2)

    common.print0("visible")
    captured = capsys.readouterr()

    assert captured.out.strip() == "visible"
