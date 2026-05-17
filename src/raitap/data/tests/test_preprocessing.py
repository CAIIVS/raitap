"""Tests for ``raitap.data.preprocessing.resolve_preprocessing``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from raitap.configs.schema import DataConfig, ModelConfig
from raitap.data.preprocessing import ResolvedPreprocessing, resolve_preprocessing

FIXTURE = Path(__file__).parent / "fixtures" / "preproc_imagenet.py"
_ENV = "RAITAP_ALLOW_PREPROCESSING_EXEC"


# ---------------------------------------------------------------------------
# Off
# ---------------------------------------------------------------------------


def test_off_no_preprocessing_key_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(ModelConfig(source="resnet50"), DataConfig())
    assert isinstance(result, ResolvedPreprocessing)
    assert result.origin == "off"
    assert result.data_module is None
    assert result.model_module is None
    assert not result.is_active
    assert len(result.warnings) == 1
    assert "preprocessing is OFF" in result.warnings[0]


def test_off_suppression_via_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(acknowledge_preprocessing_off=True),
    )
    assert result.origin == "off"
    assert result.warnings == []


def test_off_suppression_via_non_image_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(input_metadata={"kind": "tabular"}),
    )
    assert result.origin == "off"
    assert result.warnings == []


def test_off_suppression_via_non_image_kind_dictconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: when ``input_metadata`` is an OmegaConf ``DictConfig``
    (i.e. composed via Hydra/YAML), the non-image suppression check still
    fires. Without this, tabular CLI runs leak the image-only warning."""
    from omegaconf import OmegaConf

    monkeypatch.delenv(_ENV, raising=False)
    cfg = DataConfig()
    cfg.input_metadata = OmegaConf.create({"kind": "tabular"})  # type: ignore[assignment]
    result = resolve_preprocessing(ModelConfig(source="resnet50"), cfg)
    assert result.origin == "off"
    assert result.warnings == []


def test_off_kind_explicit_none_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(input_metadata={"kind": None}),
    )
    assert result.origin == "off"
    assert len(result.warnings) == 1
    assert "preprocessing is OFF" in result.warnings[0]


# ---------------------------------------------------------------------------
# Model-bundled
# ---------------------------------------------------------------------------


def test_model_bundled_via_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(preprocessing="model-bundled"),
    )
    assert result.origin == "model-bundled"
    assert isinstance(result.data_module, nn.Module)
    assert isinstance(result.model_module, nn.Module)
    assert "ResNet50_Weights" in result.description
    assert "DEFAULT" in result.description
    # data_module reshapes per-image: Resize(232) + CenterCrop(224).
    shaped = result.data_module(torch.zeros(3, 300, 400))
    assert shaped.shape == (3, 224, 224)
    # model_module normalises but preserves shape.
    normed = result.model_module(torch.zeros(1, 3, 224, 224))
    assert normed.shape == (1, 3, 224, 224)
    # Normalize with non-zero std means zero input maps to negative numbers.
    assert torch.all(normed < 0)


def test_model_bundled_via_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(preprocessing="model-bundled"),
    )
    assert result.origin == "model-bundled"
    assert isinstance(result.data_module, nn.Module)
    assert isinstance(result.model_module, nn.Module)
    assert "ResNet50_Weights" in result.description


def test_model_bundled_semantic_segmentation_preset_splits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``SemanticSegmentation.resize_size`` is a list (e.g.
    ``[520]``) — the splitter must pass it through to ``v2.Resize`` as a
    sequence, not coerce it to ``int``. Reproduces with ``fcn_resnet50``."""
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="fcn_resnet50"),
        DataConfig(preprocessing="model-bundled"),
    )
    assert result.origin == "model-bundled"
    assert isinstance(result.data_module, nn.Module)
    assert isinstance(result.model_module, nn.Module)
    # Native preset semantics: shortest-edge resize, aspect ratio preserved.
    shaped = result.data_module(torch.zeros(3, 600, 800))
    assert shaped.shape == (3, 520, 693)


def test_model_bundled_no_arch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(ValueError, match="requires a torchvision arch"):
        resolve_preprocessing(
            ModelConfig(source="/some/local/file.pt"),
            DataConfig(preprocessing="model-bundled"),
        )


# ---------------------------------------------------------------------------
# Custom file
# ---------------------------------------------------------------------------


def test_custom_file_refusal_without_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(PermissionError) as exc_info:
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(FIXTURE)),
        )
    msg = str(exc_info.value)
    assert "--allow-preprocessing-exec" in msg
    assert "acknowledge_preprocessing_exec" in msg


def test_custom_file_consent_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(
            preprocessing=str(FIXTURE),
            acknowledge_preprocessing_exec=False,
        ),
    )
    assert result.origin == "custom-file"
    assert isinstance(result.model_module, nn.Module)
    # The bundled fixture only exports ``make_preprocessing`` (no data factory).
    assert result.data_module is None
    assert result.file_sha256 is not None
    assert len(result.file_sha256) == 64
    int(result.file_sha256, 16)  # is hex
    assert result.file_path == FIXTURE.resolve()


def test_custom_file_consent_via_config_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(
            preprocessing=str(FIXTURE),
            acknowledge_preprocessing_exec=True,
        ),
    )
    assert result.origin == "custom-file"
    assert isinstance(result.model_module, nn.Module)
    assert result.data_module is None
    assert result.file_sha256 is not None
    assert len(result.file_sha256) == 64
    int(result.file_sha256, 16)
    assert result.file_path == FIXTURE.resolve()


def test_custom_file_data_factory_escape_hatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User files may export ``make_data_preprocessing`` for the loader side."""
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "user_preproc.py"
    fixture.write_text(
        "from torch import nn\n"
        "from torchvision.transforms import v2\n"
        "\n"
        "def make_preprocessing() -> nn.Module:\n"
        "    return v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])\n"
        "\n"
        "def make_data_preprocessing() -> nn.Module:\n"
        "    return nn.Sequential(\n"
        "        v2.Resize([232, 232], antialias=True),\n"
        "        v2.CenterCrop([224, 224]),\n"
        "    )\n"
    )
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(preprocessing=str(fixture)),
    )
    assert isinstance(result.model_module, nn.Module)
    assert isinstance(result.data_module, nn.Module)
    shaped = result.data_module(torch.zeros(3, 300, 400))
    assert shaped.shape == (3, 224, 224)


def test_custom_file_data_factory_wrong_return_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "bad_data_factory.py"
    fixture.write_text(
        "from torch import nn\n"
        "from torchvision.transforms import v2\n"
        "\n"
        "def make_preprocessing() -> nn.Module:\n"
        "    return v2.Normalize(mean=[0.0]*3, std=[1.0]*3)\n"
        "\n"
        "def make_data_preprocessing():\n"
        "    return 'not a module'\n"
    )
    with pytest.raises(TypeError, match=r"nn\.Module"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(fixture)),
        )


def test_custom_file_rejects_shared_model_and_data_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "shared_module.py"
    fixture.write_text(
        "from torch import nn\n"
        "_shared = nn.Identity()\n"
        "def make_preprocessing() -> nn.Module:\n"
        "    return _shared\n"
        "def make_data_preprocessing() -> nn.Module:\n"
        "    return _shared\n"
    )

    with pytest.raises(ValueError, match=r"separate nn\.Module instances"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(fixture)),
        )


def test_custom_file_sha256_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    data_cfg = DataConfig(preprocessing=str(FIXTURE))
    r1 = resolve_preprocessing(ModelConfig(source="resnet50"), data_cfg)
    r2 = resolve_preprocessing(ModelConfig(source="resnet50"), data_cfg)
    assert r1.file_sha256 == r2.file_sha256
    expected = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert r1.file_sha256 == expected


def test_custom_file_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    with pytest.raises(FileNotFoundError):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing="/does/not/exist.py"),
        )


def test_custom_file_wrong_extension(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "not_python.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match=r"expected a \.py file"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(bad)),
        )


def test_custom_file_factory_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "no_factory.py"
    bad.write_text("x = 1\n")
    with pytest.raises(AttributeError, match="make_preprocessing"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(bad)),
        )


def test_custom_file_factory_returns_non_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "bad_factory.py"
    bad.write_text("def make_preprocessing():\n    return 42\n")
    with pytest.raises(TypeError, match=r"nn\.Module"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(bad)),
        )


# ---------------------------------------------------------------------------
# Bad type
# ---------------------------------------------------------------------------


def test_bad_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(TypeError, match="must be a string"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=42),  # type: ignore[arg-type]
        )
