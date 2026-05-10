from __future__ import annotations

from collections import Counter
from pathlib import Path

from raitap.reporting.manifest import ReportManifest
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.reporting.view_model import build_view

_SNAPSHOT_MANIFEST = (
    Path(__file__).resolve().parents[4]
    / "outputs"
    / "2026-05-10"
    / "17-34-22"
    / "reports"
    / "report_manifest.json"
)


def test_build_view_groups_local_explainers_and_splits_metadata_tiers() -> None:
    manifest = ReportManifest.load(_SNAPSHOT_MANIFEST)

    view = build_view(manifest.sections, manifest.metadata)

    selected_samples = manifest.metadata["selected_samples"]
    expected_sample_count = len(selected_samples)
    expected_indices = [int(sample["sample_index"]) for sample in selected_samples]
    expected_visualisers_by_sample = _local_visualiser_counts(manifest)

    assert [sample.sample_index for sample in view.local_samples] == expected_indices
    assert expected_indices == [3, 8, 19]
    assert len(view.local_samples) == expected_sample_count
    assert {
        sample.sample_index: len(sample.explainers) for sample in view.local_samples
    } == expected_visualisers_by_sample

    gradcam = next(
        explainer
        for sample in view.local_samples
        for explainer in sample.explainers
        if explainer.explainer_name == "gradcam_localisation"
    )

    assert gradcam.context["layer_path"] == "1.layer4.2.conv3"
    assert gradcam.context["visualiser_sign"] == "positive"
    assert all(key in gradcam.technical for key in _keys_with_prefix(gradcam.rows, "call."))
    assert all(key in gradcam.technical for key in _keys_with_prefix(gradcam.rows, "input_"))


def test_build_view_groups_robustness_evidence_by_metadata_order() -> None:
    manifest = ReportManifest.load(_SNAPSHOT_MANIFEST)

    view = build_view(manifest.sections, manifest.metadata)

    assessors = {assessor.assessor_name: assessor for assessor in view.robustness_assessors}
    assert set(assessors) == {"fgsm_linf_fast", "pgd_linf_small"}

    for assessor in assessors.values():
        assert [sample.sample_index for sample in assessor.samples] == [3, 8, 19]
        for sample in assessor.samples:
            assert "ImagePairVisualiser" in sample.visualisers
            assert "PerturbationHeatmapVisualiser" in sample.visualisers


def test_build_view_ignores_multirun_heading_prefixes_when_metadata_is_present() -> None:
    section = ReportSection.from_groups(
        "Local Explanations",
        [
            ReportGroup(
                heading="job1 - Sample - wrong | pred=5",
                images=(Path("_assets/sample_3_thumbnail_0.png"),),
                table_rows=(
                    ("bucket", "wrong"),
                    ("sample_index", "3"),
                    ("predicted_class", "5"),
                ),
                metadata={"role": "sample_header", "bucket": "wrong", "sample_index": 3},
            ),
            ReportGroup(
                heading="job1 - Explainer: misleading heading - Visualiser: other",
                images=(Path("_assets/sample_3_gradcam_localisation_CaptumImageVisualiser_0.png"),),
                table_rows=(
                    ("explainer", "gradcam_localisation"),
                    ("algorithm", "LayerGradCam"),
                    ("visualiser_sign", "positive"),
                ),
                metadata={
                    "role": "local_visualiser",
                    "bucket": "wrong",
                    "sample_index": 3,
                    "explainer_name": "gradcam_localisation",
                    "algorithm": "LayerGradCam",
                    "visualiser_name": "Grad-CAM lesion localisation",
                },
            ),
        ],
        metadata={"section_role": "local_explanations"},
    )

    view = build_view((section,), {"selected_samples": [{"sample_index": 3}]})

    assert len(view.local_samples) == 1
    assert view.local_samples[0].sample_index == 3
    assert view.local_samples[0].bucket == "wrong"
    assert view.local_samples[0].explainers[0].explainer_name == "gradcam_localisation"
    assert view.local_samples[0].explainers[0].algorithm == "LayerGradCam"


def _local_visualiser_counts(manifest: ReportManifest) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for section in manifest.sections:
        if section.metadata.get("section_role") != "local":
            continue
        for group in section.groups:
            if group.metadata.get("role") == "local_visualiser":
                sample_index = group.metadata["sample_index"]
                assert isinstance(sample_index, int | str)
                counter[int(sample_index)] += 1
    return dict(counter)


def _keys_with_prefix(rows: tuple[tuple[str, str], ...], prefix: str) -> list[str]:
    return [key for key, _value in rows if key.startswith(prefix)]
