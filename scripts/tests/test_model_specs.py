import json

import pytest

from lughagen_eval.model_specs import (
    ModelSpec,
    check_model_specs,
    estimate_vram_gb,
    parse_model_spec,
    parse_model_specs,
    read_adapter_base,
    resolve_model_spec,
    weights_size_gb,
)


def make_full_model(folder, size_bytes=2_000_000):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"0" * size_bytes)
    return folder


def make_adapter(folder, base="meta-llama/Llama-3.1-8B"):
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (folder / "adapter_model.safetensors").write_bytes(b"0" * 1000)
    return folder


def test_parse_string_and_object_forms():
    assert parse_model_spec("m", "/drive/m") == ModelSpec(path="/drive/m")
    spec = parse_model_spec("m", {"path": "/drive/a", "base_model_path": "/drive/base"})
    assert spec.base_path == "/drive/base"


@pytest.mark.parametrize("bad", [123, None, {"nopath": 1}, {"path": 5}, {"path": "/x", "base_model_path": 3}])
def test_parse_rejects_malformed_entries(bad):
    with pytest.raises(ValueError):
        parse_model_spec("m", bad)


def test_parse_model_specs_maps_every_entry():
    specs = parse_model_specs({"A": "/a", "B": {"path": "/b"}})
    assert set(specs) == {"A", "B"}


def test_adapter_detection_and_base_from_adapter_config(tmp_path):
    adapter = make_adapter(tmp_path / "qlora", base="/drive/base-model")
    spec = ModelSpec(path=str(adapter))
    assert spec.is_adapter
    assert read_adapter_base(str(adapter)) == "/drive/base-model"
    assert resolve_model_spec("q", spec).base_path == "/drive/base-model"


def test_explicit_base_path_wins_over_adapter_config(tmp_path):
    adapter = make_adapter(tmp_path / "qlora", base="from-config")
    spec = ModelSpec(path=str(adapter), base_path="/explicit")
    assert resolve_model_spec("q", spec).base_path == "/explicit"


def test_adapter_without_known_base_is_a_clear_error(tmp_path):
    folder = tmp_path / "qlora"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="base_model_path"):
        resolve_model_spec("q", ModelSpec(path=str(folder)))


def test_full_model_is_not_treated_as_adapter(tmp_path):
    model = make_full_model(tmp_path / "full")
    assert not ModelSpec(path=str(model)).is_adapter


def test_weights_size_counts_only_weight_files(tmp_path):
    model = make_full_model(tmp_path / "full", size_bytes=3_000_000_000 // 1000)
    (model / "tokenizer.json").write_bytes(b"0" * 5_000_000)
    assert weights_size_gb(str(model)) == pytest.approx(3_000_000 / 1e9)
    assert weights_size_gb(str(tmp_path / "missing")) == 0.0


def test_estimate_vram_scales_down_with_quantization():
    est = estimate_vram_gb(16.0)
    assert est["none"] > est["8bit"] > est["4bit"]
    assert est["8bit"] == pytest.approx(est["none"] / 2)
    assert est["4bit"] == pytest.approx(est["none"] / 4)


def test_check_model_specs_flags_each_kind_of_problem(tmp_path):
    good = make_full_model(tmp_path / "good")
    no_weights = tmp_path / "noweights"
    no_weights.mkdir()
    (no_weights / "config.json").write_text("{}")
    no_config = tmp_path / "noconfig"
    no_config.mkdir()
    (no_config / "model.safetensors").write_bytes(b"0")
    specs = parse_model_specs({
        "Good": str(good), "NoWeights": str(no_weights), "NoConfig": str(no_config), "Gone": str(tmp_path / "gone"),
    })
    problems = check_model_specs(specs, ["Good", "NoWeights", "NoConfig", "Gone", "Unlisted"])
    text = "\n".join(problems)
    assert "Good:" not in text
    assert "NoWeights: no .safetensors" in text
    assert "NoConfig: no config.json" in text
    assert "Gone: folder not found" in text and "Drive" in text
    assert "Unlisted: not listed" in text
    assert len(problems) == 4


def test_check_model_specs_accepts_adapter_with_known_base(tmp_path):
    adapter = make_adapter(tmp_path / "qlora")
    assert check_model_specs(parse_model_specs({"Q": str(adapter)}), ["Q"]) == []
