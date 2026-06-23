from raitap.data.types import LabelFormat
from raitap.configs.schema import LabelsConfig


def test_label_format_members_are_string_values():
    assert LabelFormat.native == "native"
    assert {f.value for f in LabelFormat} == {"native", "coco", "yolo", "voc"}


def test_labels_config_defaults_to_native_format():
    assert LabelsConfig().format is LabelFormat.native
