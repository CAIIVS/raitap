"""Tests for ``raitap.data.preprocessing.resolve_preprocessing``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from raitap.configs.schema import DataConfig, ModelConfig
from raitap.data import Preprocessing
from raitap.data.preprocessing import ResolvedPreprocessing, resolve_preprocessing

FIXTURE = Path(__file__).parent / "fixtures" / "preproc_imagenet.py"
_ENV = "RAITAP_ALLOW_PREPROCESSING_EXEC"


# ---------------------------------------------------------------------------
# Both off
# ---------------------------------------------------------------------------


def test_both_off_no_keys_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(ModelConfig(source="resnet50"), DataConfig())
    assert isinstance(result, ResolvedPreprocessing)
    assert result.data_origin == "off"
    assert result.model_origin == "off"
    assert result.data_module is None
    assert result.model_module is None
    assert not result.is_active
    assert len(result.warnings) == 1
    assert "No image preprocessing set in config" in result.warnings[0]


def test_both_off_suppression_via_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(),
        acknowledge_off=True,
    )
    assert result.data_origin == "off"
    assert result.model_origin == "off"
    assert result.warnings == []


def test_both_off_suppression_via_non_image_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(input_metadata={"kind": "tabular"}),
    )
    assert result.data_origin == "off"
    assert result.model_origin == "off"
    assert result.warnings == []


def test_both_off_suppression_via_non_image_kind_dictconfig(
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
    assert result.data_origin == "off"
    assert result.warnings == []


def test_both_off_kind_explicit_none_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(input_metadata={"kind": None}),
    )
    assert result.data_origin == "off"
    assert result.model_origin == "off"
    assert len(result.warnings) == 1
    assert "No image preprocessing set in config" in result.warnings[0]


# ---------------------------------------------------------------------------
# Model side off but data side set: separate warning fires (image data only)
# ---------------------------------------------------------------------------


def test_model_input_transformation_off_warns_when_data_side_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(preprocessing="model-bundled"),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "off"
    assert result.data_module is not None
    assert result.model_module is None
    assert any("No data.model_input_transformation set in config" in w for w in result.warnings)


def test_model_input_transformation_off_suppressed_for_tabular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(
            preprocessing="model-bundled",
            input_metadata={"kind": "tabular"},
        ),
    )
    assert result.warnings == []


def test_data_off_with_model_set_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already-uniform images with only Normalize at the boundary is legitimate."""
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(model_input_transformation="model-bundled"),
    )
    assert result.data_origin == "off"
    assert result.model_origin == "model-bundled"
    assert result.warnings == []


# ---------------------------------------------------------------------------
# Model-bundled (both sides)
# ---------------------------------------------------------------------------


def test_model_bundled_both_sides_via_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(
            preprocessing="model-bundled",
            model_input_transformation="model-bundled",
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "model-bundled"
    assert isinstance(result.data_module, nn.Module)
    assert isinstance(result.model_module, nn.Module)
    assert "ResNet50_Weights" in result.description
    assert "DEFAULT" in result.description
    assert result.warnings == []


def test_model_bundled_accepts_preprocessing_enum_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python API: ``Preprocessing.model_bundled`` resolves like the raw string."""
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(
            preprocessing=Preprocessing.model_bundled,
            model_input_transformation=Preprocessing.model_bundled,
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "model-bundled"
    assert result.data_module is not None
    assert result.model_module is not None
    # data_module reshapes per-image: Resize(232) + CenterCrop(224).
    shaped = result.data_module(torch.zeros(3, 300, 400))
    assert shaped.shape == (3, 224, 224)
    # model_module normalises but preserves shape.
    normed = result.model_module(torch.zeros(1, 3, 224, 224))
    assert normed.shape == (1, 3, 224, 224)
    # Normalize with non-zero std means zero input maps to negative numbers.
    assert torch.all(normed < 0)


def test_preprocessing_yaml_values_survive_structured_config_merge() -> None:
    from omegaconf import OmegaConf

    structured = OmegaConf.structured(DataConfig)

    merged = OmegaConf.to_object(
        OmegaConf.merge(
            structured,
            OmegaConf.create(
                {
                    "preprocessing": "model-bundled",
                    "model_input_transformation": "./normalize.py",
                }
            ),
        )
    )
    assert isinstance(merged, DataConfig)
    assert merged.preprocessing == "model-bundled"
    assert merged.model_input_transformation == "./normalize.py"
    assert type(merged.preprocessing) is str
    assert type(merged.model_input_transformation) is str


def test_model_bundled_via_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(
            preprocessing="model-bundled",
            model_input_transformation="model-bundled",
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "model-bundled"
    assert "ResNet50_Weights" in result.description


def test_model_bundled_via_detection_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # Detection builders live under torchvision.models.detection, not top-level
    # torchvision.models. Resolution must fall back there (mirrors the model
    # loader's _resolve_torchvision_factory). Regression guard for #196.
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="fasterrcnn_resnet50_fpn_v2"),
        DataConfig(
            preprocessing="model-bundled",
            model_input_transformation="model-bundled",
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "model-bundled"
    assert "FasterRCNN" in result.description
    # ObjectDetection preset is a no-op split: detectors normalise internally,
    # so neither side carries a transform.
    assert result.data_module is None
    assert result.model_module is None


def test_model_bundled_semantic_segmentation_preset_splits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(arch="fcn_resnet50"),
        DataConfig(
            preprocessing="model-bundled",
            model_input_transformation="model-bundled",
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "model-bundled"
    assert result.data_module is not None
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
# Mixed: bundled on one knob, file on the other
# ---------------------------------------------------------------------------


def test_mixed_data_bundled_model_custom_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    result = resolve_preprocessing(
        ModelConfig(arch="resnet50"),
        DataConfig(
            preprocessing="model-bundled",
            model_input_transformation=str(FIXTURE),
        ),
    )
    assert result.data_origin == "model-bundled"
    assert result.model_origin == "custom-file"
    assert isinstance(result.data_module, nn.Module)
    assert isinstance(result.model_module, nn.Module)
    assert result.model_file_path == FIXTURE.resolve()
    assert result.model_file_sha256 is not None
    assert result.data_file_path is None
    assert result.data_file_sha256 is None
    assert "data: ResNet50_Weights" in result.description
    assert "model: Custom model input transformation" in result.description


# ---------------------------------------------------------------------------
# Custom file (model side)
# ---------------------------------------------------------------------------


def test_custom_file_refusal_without_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(PermissionError) as exc_info:
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(model_input_transformation=str(FIXTURE)),
        )
    msg = str(exc_info.value)
    assert "--allow-preprocessing-exec" in msg
    assert "acknowledge_preprocessing_exec" in msg


def test_custom_file_consent_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(model_input_transformation=str(FIXTURE)),
    )
    assert result.model_origin == "custom-file"
    assert isinstance(result.model_module, nn.Module)
    assert result.data_module is None
    assert result.model_file_sha256 is not None
    assert len(result.model_file_sha256) == 64
    int(result.model_file_sha256, 16)
    assert result.model_file_path == FIXTURE.resolve()


def test_custom_file_consent_via_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(model_input_transformation=str(FIXTURE)),
        acknowledge_exec=True,
    )
    assert result.model_origin == "custom-file"
    assert isinstance(result.model_module, nn.Module)
    assert result.model_file_sha256 is not None
    assert result.model_file_path == FIXTURE.resolve()


# ---------------------------------------------------------------------------
# Custom file (data side, and both sides from same file)
# ---------------------------------------------------------------------------


def test_custom_file_data_side_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "data_only.py"
    fixture.write_text(
        "from torch import nn\n"
        "from torchvision.transforms import v2\n"
        "from raitap.data import raitap_preprocessing_factory\n"
        "\n"
        "@raitap_preprocessing_factory\n"
        "def resize_for_batching() -> nn.Module:\n"
        "    return v2.CenterCrop([8, 8])\n"
    )
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(preprocessing=str(fixture)),
    )
    assert result.data_origin == "custom-file"
    assert isinstance(result.data_module, nn.Module)
    assert result.model_module is None
    assert result.data_file_path == fixture.resolve()
    # model side off + image data → warning about model side
    assert any("No data.model_input_transformation set in config" in w for w in result.warnings)


def test_custom_file_both_sides_from_same_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single file with both decorators, pointed at both knobs. Hashed once."""
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "user_preproc.py"
    fixture.write_text(
        "from torch import nn\n"
        "from torchvision.transforms import v2\n"
        "from raitap.data import (\n"
        "    raitap_model_input_transformation_factory,\n"
        "    raitap_preprocessing_factory,\n"
        ")\n"
        "\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_for_model() -> nn.Module:\n"
        "    return v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])\n"
        "\n"
        "@raitap_preprocessing_factory\n"
        "def resize_for_batching() -> nn.Module:\n"
        "    return nn.Sequential(\n"
        "        v2.Resize([232, 232], antialias=True),\n"
        "        v2.CenterCrop([224, 224]),\n"
        "    )\n"
    )
    result = resolve_preprocessing(
        ModelConfig(source="resnet50"),
        DataConfig(
            preprocessing=str(fixture),
            model_input_transformation=str(fixture),
        ),
    )
    assert result.data_origin == "custom-file"
    assert result.model_origin == "custom-file"
    assert result.data_module is not None
    assert result.model_module is not None
    assert result.data_file_path == fixture.resolve()
    assert result.model_file_path == fixture.resolve()
    assert result.data_file_sha256 == result.model_file_sha256
    assert result.warnings == []
    shaped = result.data_module(torch.zeros(3, 300, 400))
    assert shaped.shape == (3, 224, 224)


def test_custom_file_missing_required_decorator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File with only the model decorator, pointed at the data knob, errors."""
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "model_only.py"
    fixture.write_text(
        "from torch import nn\n"
        "from raitap.data import raitap_model_input_transformation_factory\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_for_model() -> nn.Module:\n"
        "    return nn.Identity()\n"
    )
    with pytest.raises(AttributeError, match="raitap_preprocessing_factory"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(fixture)),
        )


def test_custom_file_data_factory_wrong_return_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "bad_data_factory.py"
    fixture.write_text(
        "from torch import nn\n"
        "from raitap.data import raitap_preprocessing_factory\n"
        "\n"
        "@raitap_preprocessing_factory\n"
        "def resize_for_batching():\n"
        "    return 'not a module'\n"
    )
    with pytest.raises(TypeError, match=r"nn\.Module"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(fixture)),
        )


def test_custom_file_rejects_factory_with_required_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "arity_mismatch.py"
    fixture.write_text(
        "from torch import nn\n"
        "from torchvision.transforms import v2\n"
        "from raitap.data import raitap_model_input_transformation_factory\n"
        "\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_for_model(arg) -> nn.Module:\n"
        "    return v2.Normalize(mean=[0.0]*3, std=[1.0]*3)\n"
    )
    with pytest.raises(TypeError, match=r"must be callable with no arguments"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(model_input_transformation=str(fixture)),
        )


def test_custom_file_rejects_shared_module_across_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    fixture = tmp_path / "shared_module.py"
    fixture.write_text(
        "from torch import nn\n"
        "from raitap.data import (\n"
        "    raitap_model_input_transformation_factory,\n"
        "    raitap_preprocessing_factory,\n"
        ")\n"
        "_shared = nn.Identity()\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_for_model() -> nn.Module:\n"
        "    return _shared\n"
        "@raitap_preprocessing_factory\n"
        "def resize_for_batching() -> nn.Module:\n"
        "    return _shared\n"
    )
    with pytest.raises(ValueError, match=r"separate nn\.Module instances"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(
                preprocessing=str(fixture),
                model_input_transformation=str(fixture),
            ),
        )


def test_custom_file_sha256_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    data_cfg = DataConfig(model_input_transformation=str(FIXTURE))
    r1 = resolve_preprocessing(ModelConfig(source="resnet50"), data_cfg)
    r2 = resolve_preprocessing(ModelConfig(source="resnet50"), data_cfg)
    assert r1.model_file_sha256 == r2.model_file_sha256
    expected = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert r1.model_file_sha256 == expected


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
    with pytest.raises(AttributeError, match="decorated"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(bad)),
        )


def test_custom_file_undecorated_factory_raises_no_factory_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "old_factory.py"
    bad.write_text(
        "from torch import nn\ndef make_preprocessing() -> nn.Module:\n    return nn.Identity()\n"
    )
    with pytest.raises(
        AttributeError, match=r"no factory decorated with `@raitap_preprocessing_factory`"
    ):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=str(bad)),
        )


def test_custom_file_factory_returns_non_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "bad_factory.py"
    bad.write_text(
        "from raitap.data import raitap_model_input_transformation_factory\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_for_model():\n"
        "    return 42\n"
    )
    with pytest.raises(TypeError, match=r"nn\.Module"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(model_input_transformation=str(bad)),
        )


def test_custom_file_duplicate_decorated_factories_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_ENV, "1")
    bad = tmp_path / "duplicates.py"
    bad.write_text(
        "from torch import nn\n"
        "from raitap.data import raitap_model_input_transformation_factory\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_a() -> nn.Module:\n"
        "    return nn.Identity()\n"
        "@raitap_model_input_transformation_factory\n"
        "def normalize_b() -> nn.Module:\n"
        "    return nn.Identity()\n"
    )
    with pytest.raises(ValueError, match="multiple model input transformation"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(model_input_transformation=str(bad)),
        )


# ---------------------------------------------------------------------------
# Bad type
# ---------------------------------------------------------------------------


def test_bad_type_preprocessing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(TypeError, match=r"data\.preprocessing must be a string"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(preprocessing=42),  # type: ignore[arg-type]
        )


def test_bad_type_model_input_transformation_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(TypeError, match=r"data\.model_input_transformation must be a string"):
        resolve_preprocessing(
            ModelConfig(source="resnet50"),
            DataConfig(model_input_transformation=42),  # type: ignore[arg-type]
        )
