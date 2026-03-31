import os

import pytest

from st_service import config as config_module


def test_resolve_torch_device_cpu():
    d = config_module.resolve_torch_device("cpu")
    assert str(d) == "cpu"


def test_resolve_torch_device_invalid():
    with pytest.raises(ValueError):
        config_module.resolve_torch_device("metal")


def test_get_settings_reads_env(monkeypatch):
    config_module.get_settings.cache_clear()
    monkeypatch.setenv("ST_MODEL_NAME", "dummy/model")
    monkeypatch.setenv("ST_PORT", "9000")
    s = config_module.get_settings()
    assert s.st_model_name == "dummy/model"
    assert s.st_port == 9000
    config_module.get_settings.cache_clear()
