from __future__ import annotations

from collections import Counter
from pathlib import Path

from raitap.reporting.manifest import ReportManifest
from raitap.reporting.sections import ReportGroup, ReportSection
from raitap.reporting.view_model import build_view


def test_build_view_captures_reproducibility_banner_and_excludes_from_appendix() -> None:
    caveat = (
        "This run includes stochastic methods (pgd); results are not "
        "bit-reproducible unless seeds are pinned."
    )
    sections = (
        ReportSection.from_groups(
            "Reproducibility",
            [ReportGroup(heading=caveat)],
            metadata={"section_role": "reproducibility"},
        ),
        ReportSection.from_groups(
            "Metrics",
            [ReportGroup(heading="Performance Metrics", table_rows=(("accuracy", "0.9"),))],
            metadata={"section_role": "metrics"},
        ),
    )

    view = build_view(sections)

    assert view.reproducibility == caveat
    assert "Reproducibility" not in [section.title for section in view.appendix.sections]


def test_build_view_reproducibility_none_for_deterministic_run() -> None:
    sections = (
        ReportSection.from_groups(
            "Metrics",
            [ReportGroup(heading="Performance Metrics", table_rows=(("accuracy", "0.9"),))],
            metadata={"section_role": "metrics"},
        ),
    )

    view = build_view(sections)

    assert view.reproducibility is None


def test_build_view_groups_local_explainers_and_splits_metadata_tiers() -> None:
    manifest = _snapshot_manifest()

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
    manifest = _snapshot_manifest()

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


def test_build_view_carries_baseline_image_onto_explainer() -> None:
    section = ReportSection.from_groups(
        "Local Explanations",
        [
            ReportGroup(
                heading="Explainer: shap_grad - Visualiser: SHAP",
                images=(
                    Path("_assets/sample_0_shap_grad.png"),
                    Path("_assets/baseline_sample_0_shap_grad.png"),
                ),
                table_rows=(
                    ("explainer", "shap_grad"),
                    ("algorithm", "GradientExplainer"),
                    ("baseline.n_samples", "20"),
                ),
                metadata={
                    "role": "local_visualiser",
                    "sample_index": 0,
                    "explainer_name": "shap_grad",
                    "algorithm": "GradientExplainer",
                    "visualiser_name": "SHAP",
                    "baseline_image": "reports/_assets/baseline_sample_0_shap_grad.png",
                },
            ),
        ],
        metadata={"section_role": "local_explanations"},
    )

    view = build_view((section,), {})

    explainer = view.local_samples[0].explainers[0]
    # One field drives both the link and the reference-card anchor (review #6).
    assert explainer.baseline_image_src == "_assets/baseline_sample_0_shap_grad.png"
    # The local-panel image is the visualiser render, not the baseline.
    assert explainer.image_srcs[0] == "_assets/sample_0_shap_grad.png"


def test_build_view_renders_verbose_local_detail_groups_as_samples() -> None:
    section = ReportSection.from_groups(
        "Local Explanations",
        [
            ReportGroup(
                heading="Detail - user_selected case_gamma.png",
                images=(
                    Path("reports/_assets/detail_user_selected_gradcam_2_0.png"),
                    Path("reports/_assets/detail_user_selected_integrated_gradients_2_0.png"),
                ),
                metadata={
                    "role": "local_detail",
                    "bucket": "user_selected",
                    "sample_index": 2,
                    "requested_sample": "case_gamma.png",
                },
            ),
            ReportGroup(
                heading="Detail - user_selected case_alpha.png",
                images=(Path("reports/_assets/detail_user_selected_gradcam_0_0.png"),),
                metadata={
                    "role": "local_detail",
                    "bucket": "user_selected",
                    "sample_index": 0,
                    "requested_sample": "case_alpha.png",
                },
            ),
        ],
        metadata={"section_role": "local_explanations"},
    )

    view = build_view((section,), {})

    assert [sample.sample_index for sample in view.local_samples] == [2, 0]
    assert [sample.bucket for sample in view.local_samples] == ["user_selected", "user_selected"]
    assert view.local_samples[0].sample_id == "case_gamma.png"
    assert [
        image
        for sample in view.local_samples
        for explainer in sample.explainers
        for image in explainer.image_srcs
    ] == [
        "_assets/detail_user_selected_gradcam_2_0.png",
        "_assets/detail_user_selected_integrated_gradients_2_0.png",
        "_assets/detail_user_selected_gradcam_0_0.png",
    ]


def test_build_view_builds_evaluation_section_and_excludes_it_from_appendix(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "_assets" / "gradcam_localisation_quantus_scores.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    section = ReportSection.from_groups(
        "Explanation quality (Quantus)",
        [
            ReportGroup(
                heading="gradcam_localisation",
                images=(image_path,),
                table_rows=(("faithfulness_correlation", "0.8000"),),
                metadata={"role": "evaluation", "algorithm": "LayerGradCam"},
            )
        ],
        metadata={"section_role": "evaluation"},
    )

    view = build_view((section,), {})

    assert view.evaluation is not None
    assert len(view.evaluation.groups) == 1
    group = view.evaluation.groups[0]
    assert group.heading == "gradcam_localisation"
    assert group.table_rows == (("faithfulness_correlation", "0.8000"),)
    assert group.image_srcs == ("_assets/gradcam_localisation_quantus_scores.png",)
    assert "Explanation quality (Quantus)" not in [
        appendix_section.title for appendix_section in view.appendix.sections
    ]


def test_build_view_evaluation_none_when_no_evaluation_section() -> None:
    view = build_view((), {})

    assert view.evaluation is None


def test_build_view_populates_summary_model_and_data_from_metadata() -> None:
    view = build_view(
        (),
        {
            "experiment_name": "demo",
            "model_source": "/abs/path/to/lwise_ham10000_eager.pt",
            "data_name": "ham10000-presentation-balanced",
        },
    )

    assert view.summary.model_name == "lwise_ham10000_eager.pt"
    assert view.summary.data_name == "ham10000-presentation-balanced"


def test_build_view_keeps_non_path_model_source_as_is() -> None:
    view = build_view((), {"model_source": "resnet50", "data_name": "isic2018"})

    assert view.summary.model_name == "resnet50"
    assert view.summary.data_name == "isic2018"


def test_build_view_defaults_summary_model_and_data_to_na_when_missing() -> None:
    view = build_view((), {})

    assert view.summary.model_name == "n/a"
    assert view.summary.data_name == "n/a"


def test_build_view_treats_blank_model_and_data_as_na() -> None:
    view = build_view((), {"model_source": "   ", "data_name": ""})

    assert view.summary.model_name == "n/a"
    assert view.summary.data_name == "n/a"


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


def _snapshot_manifest() -> ReportManifest:
    return ReportManifest(
        kind="run",
        metadata={
            "selected_samples": [
                {"sample_index": 3},
                {"sample_index": 8},
                {"sample_index": 19},
            ]
        },
        sections=(
            ReportSection.from_groups(
                "Local Explanations",
                [_sample_header(sample_index) for sample_index in (3, 8, 19)]
                + [_local_visualiser(sample_index) for sample_index in (3, 8, 19)],
                metadata={"section_role": "local"},
            ),
            ReportSection.from_groups(
                "Robustness",
                [
                    _robustness_assessor("fgsm_linf_fast", 0),
                    _robustness_assessor("pgd_linf_small", 1),
                ],
                metadata={"section_role": "robustness"},
            ),
        ),
    )


def _sample_header(sample_index: int) -> ReportGroup:
    return ReportGroup(
        heading=f"Sample {sample_index}",
        images=(Path(f"_assets/sample_{sample_index}_thumbnail_0.png"),),
        table_rows=(("sample_index", str(sample_index)),),
        metadata={"role": "sample_header", "sample_index": sample_index},
    )


def _local_visualiser(sample_index: int) -> ReportGroup:
    return ReportGroup(
        heading=f"Sample {sample_index} - Explainer: gradcam_localisation",
        images=(
            Path(f"_assets/sample_{sample_index}_gradcam_localisation_CaptumImageVisualiser_0.png"),
        ),
        table_rows=(
            ("explainer", "gradcam_localisation"),
            ("algorithm", "LayerGradCam"),
            ("layer_path", "1.layer4.2.conv3"),
            ("visualiser_sign", "positive"),
            ("call.target", "1"),
            ("input_shape", "(1, 3, 224, 224)"),
        ),
        metadata={
            "role": "local_visualiser",
            "sample_index": sample_index,
            "explainer_name": "gradcam_localisation",
            "algorithm": "LayerGradCam",
        },
    )


def _robustness_assessor(assessor_name: str, result_index: int) -> ReportGroup:
    return ReportGroup(
        heading=f"Robustness: {assessor_name}",
        images=tuple(
            Path(
                f"_assets/robustness_{result_index}_{assessor_name}_sample_{sample_index}_"
                f"{visualiser}_0.png"
            )
            for sample_index in (3, 8, 19)
            for visualiser in ("ImagePairVisualiser", "PerturbationHeatmapVisualiser")
        ),
        table_rows=(
            ("assessor", assessor_name),
            ("algorithm", assessor_name),
            ("assessment_kind", "empirical"),
            ("clean_accuracy", "0.9000"),
            ("adversarial_accuracy", "0.8000"),
        ),
        metadata={
            "role": "robustness",
            "assessor_name": assessor_name,
            "algorithm": assessor_name,
            "assessment_kind": "empirical",
            "sample_indices": [3, 8, 19],
        },
    )


def _keys_with_prefix(rows: tuple[tuple[str, str], ...], prefix: str) -> list[str]:
    return [key for key, _value in rows if key.startswith(prefix)]
