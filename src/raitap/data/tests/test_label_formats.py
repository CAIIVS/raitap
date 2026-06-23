import pytest
from raitap.data.types import LabelFormat
from raitap.configs.schema import LabelsConfig
from raitap.data.label_formats import (
    LABEL_FORMAT_ADAPTERS,
    label_format,
    resolve_label_format_adapter,
)
from raitap.types import TaskKind


def test_label_format_members_are_string_values():
    assert LabelFormat.native == "native"
    assert {f.value for f in LabelFormat} == {"native", "coco", "yolo", "voc"}


def test_labels_config_defaults_to_native_format():
    assert LabelsConfig().format is LabelFormat.native


def test_label_format_decorator_registers_instance():
    @label_format
    class _Dummy:
        format = LabelFormat.coco  # reuse an enum member; popped below
        supported_tasks = frozenset({TaskKind.detection})

    try:
        assert LABEL_FORMAT_ADAPTERS[LabelFormat.coco].supported_tasks == frozenset(
            {TaskKind.detection}
        )
    finally:
        LABEL_FORMAT_ADAPTERS.pop(LabelFormat.coco, None)


def test_registry_rejects_unknown_native():
    with pytest.raises(ValueError, match="No adapter"):
        resolve_label_format_adapter(LabelFormat.native, task_kind=TaskKind.detection)


@pytest.mark.xfail(reason="adapter added in task 4/5", strict=False)
def test_registry_resolves_supported_task():
    adapter = resolve_label_format_adapter(LabelFormat.coco, task_kind=TaskKind.detection)
    assert adapter.format is LabelFormat.coco
    assert TaskKind.detection in adapter.supported_tasks


@pytest.mark.xfail(reason="adapter added in task 4/5", strict=False)
def test_registry_rejects_unsupported_task():
    with pytest.raises(ValueError, match="does not support task"):
        resolve_label_format_adapter(LabelFormat.yolo, task_kind=TaskKind.classification)
