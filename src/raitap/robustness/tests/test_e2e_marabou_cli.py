"""End-to-end CLI smoke tests for ``MarabouAssessor``.

Each test drives ``python -m raitap.run`` as a subprocess against a tiny
synthetic fixture, exercising every layer between the user and the
solver:

* Hydra config resolution + interpolation
* :class:`raitap.models.Model` + :class:`raitap.data.Data` construction
* Backend shape adapter (reshape path covered by the image cell)
* Pipeline forward pass through ONNX Runtime
* :class:`MarabouAssessor.assess` + ``verify_sample`` per-sample loop
* Result serialisation (``robustness_data.pt``)
* Optional reporting hand-off (HTML cell)

The existing ``test_e2e_marabou_acas_xu`` tests call ``verify_sample``
directly with a pre-loaded ONNX path, so they miss every layer above
that — these CLI variants catch the regression class encountered while
wiring UC1 (HOME interpolation, ``MarabouUtils.Equation`` wrapper,
mixed image+CSV dir, etc.).

Skips when ``maraboupy`` / ``onnx`` / ``onnxruntime`` / ``jinja2`` (for
the HTML cell) is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.e2e

_SUBPROCESS_TIMEOUT_S = 180


def _build_tiny_mlp_onnx(
    target_path: Path,
    *,
    in_features: int,
    hidden: int,
    out_features: int,
    seed: int = 0,
) -> None:
    """Write a ``in_features → hidden → out_features`` ReLU MLP to ``target_path``.

    Pure Linear + ReLU so the maraboupy ONNX parser accepts it without
    hitting the ``Shape``/``Reshape``/``Gather`` op limitations that
    ``nn.Flatten`` (or any dynamic reshape) would introduce. Batch dim
    is declared as the symbolic ``"batch"`` so the backend shape adapter
    treats it as dynamic.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((in_features, hidden)).astype(np.float32) * 0.5
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.standard_normal((hidden, out_features)).astype(np.float32) * 0.5
    b2 = np.zeros(out_features, dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["h1_relu"]),
            helper.make_node("Gemm", ["h1_relu", "w2", "b2"], ["logits"]),
        ],
        "tiny_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_features])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", out_features])],
        [
            numpy_helper.from_array(w1, name="w1"),
            numpy_helper.from_array(b1, name="b1"),
            numpy_helper.from_array(w2, name="w2"),
            numpy_helper.from_array(b2, name="b2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(target_path))


def _write_tabular_config(target_path: Path, fixture_dir: Path) -> None:
    """Tabular-CSV config — minimal coverage, no shape reshape."""
    fixture_str = fixture_dir.as_posix()
    target_path.write_text(
        textwrap.dedent(
            f"""\
            # @package _global_
            defaults:
              - _self_

            hardware: cpu
            experiment_name: e2e-marabou-cli-tabular

            model:
              source: {fixture_str}/model.onnx

            data:
              name: e2e-tabular
              source: {fixture_str}/inputs.csv
              input_metadata:
                kind: tabular
                layout: "(B,F)"
              labels:
                source: {fixture_str}/labels.csv
                id_column: id
                column: label
                encoding: index

            metrics: null
            tracking: null
            reporting: null
            transparency: {{}}

            robustness:
              marabou_linf:
                _target_: MarabouAssessor
                algorithm: linf-box
                constructor:
                  epsilon: 0.001
                  norm: Linf
                  timeout_s: 30
                visualisers: []
            """,
        ),
        encoding="utf-8",
    )


def _write_image_config(
    target_path: Path,
    fixture_dir: Path,
    *,
    non_batch_numel: int,
    with_reporting: bool = False,
) -> None:
    """Image-dir config — exercises the backend shape adapter reshape path.

    Data loader emits ``(N, 3, H, W)`` from the PNG dir; the MLP wants
    ``(N, non_batch_numel)``, so ``data.input_metadata.shape`` forces
    the adapter to fold the image into the flat layout.
    """
    fixture_str = fixture_dir.as_posix()
    reporting_block = (
        "reporting:\n"
        "              _target_: HTMLReporter\n"
        "              filename: report\n"
        "              include_config: false\n"
        "              include_metadata: false\n"
        "              multirun_report: false\n"
        if with_reporting
        else "reporting: null"
    )
    target_path.write_text(
        textwrap.dedent(
            f"""\
            # @package _global_
            defaults:
              - _self_

            hardware: cpu
            experiment_name: e2e-marabou-cli-image

            model:
              source: {fixture_str}/model.onnx

            data:
              name: e2e-image
              source: {fixture_str}/images
              input_metadata:
                kind: image
                layout: NCHW
                shape: [{non_batch_numel}]
              labels:
                source: {fixture_str}/labels.csv
                id_column: image
                column: label
                encoding: index

            metrics: null
            tracking: null
            {reporting_block}
            transparency: {{}}

            robustness:
              marabou_linf:
                _target_: MarabouAssessor
                algorithm: linf-box
                constructor:
                  epsilon: 0.001
                  norm: Linf
                  timeout_s: 30
                visualisers: []
            """,
        ),
        encoding="utf-8",
    )


def _seed_image_fixture(fixture_dir: Path, *, num_samples: int) -> int:
    """Generate ``num_samples`` 4x4 RGB PNG samples + a labels.csv.

    Labels.csv lives next to the image dir (not inside) so the loader's
    "mixed image+tabular files" guard doesn't trip. Returns the per-sample
    non-batch ``numel`` (= 3 channels x 4 x 4 = 48) that the matching
    config must declare as ``input_metadata.shape``.
    """
    pil_image = pytest.importorskip("PIL.Image")
    images_dir = fixture_dir / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = ["image,label"]
    for index in range(num_samples):
        arr = (rng.random((4, 4, 3)) * 255).astype(np.uint8)
        filename = f"sample_{index:02d}.png"
        pil_image.fromarray(arr, mode="RGB").save(images_dir / filename)
        rows.append(f"{filename},{index % 3}")
    (fixture_dir / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 3 * 4 * 4


def _run_raitap(
    config_dir: Path,
    config_name: str,
    run_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "raitap.run",
            "--config-dir",
            str(config_dir),
            "--config-name",
            config_name,
            f"hydra.run.dir={run_dir.as_posix()}",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )


def _assert_succeeded(proc: subprocess.CompletedProcess[str], run_dir: Path) -> None:
    if proc.returncode != 0:
        run_contents = (
            sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) if run_dir.exists() else []
        )
        pytest.fail(
            f"raitap CLI exited with code {proc.returncode}\n"
            f"Run dir contents: {run_contents}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


def _assert_robustness_persisted(proc: subprocess.CompletedProcess[str], run_dir: Path) -> None:
    persisted = run_dir / "robustness" / "marabou_linf" / "robustness_data.pt"
    legacy = run_dir / "robustness_data.pt"
    if persisted.exists() or legacy.exists():
        return
    pytest.fail(
        "Marabou assessor did not persist results.\n"
        f"Tried: {persisted}\n"
        f"Tried: {legacy}\n"
        f"Run dir contents: {sorted(p.relative_to(run_dir) for p in run_dir.rglob('*'))}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_raitap_cli_marabou_tabular_smoke(tmp_path: Path) -> None:
    """Tabular CSV fixture: covers CLI + data + model + assessor + persistence."""
    pytest.importorskip("maraboupy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    _build_tiny_mlp_onnx(
        fixture_dir / "model.onnx",
        in_features=3,
        hidden=4,
        out_features=3,
    )

    rng = np.random.default_rng(0)
    samples = rng.standard_normal((2, 3)).astype(np.float32)
    inputs_csv = ["f0,f1,f2"]
    for row in samples:
        inputs_csv.append(",".join(f"{value:.6f}" for value in row))
    (fixture_dir / "inputs.csv").write_text("\n".join(inputs_csv) + "\n", encoding="utf-8")

    labels_csv = ["id,label"]
    for row_idx in range(samples.shape[0]):
        labels_csv.append(f"{row_idx},{row_idx % 3}")
    (fixture_dir / "labels.csv").write_text("\n".join(labels_csv) + "\n", encoding="utf-8")

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_tabular_config(config_dir / "e2e_marabou_cli_tabular.yaml", fixture_dir)

    run_dir = tmp_path / "run"
    proc = _run_raitap(config_dir, "e2e_marabou_cli_tabular", run_dir)
    _assert_succeeded(proc, run_dir)
    _assert_robustness_persisted(proc, run_dir)


def test_raitap_cli_marabou_image_shape_adapter(tmp_path: Path) -> None:
    """Image-dir fixture: covers the backend shape adapter reshape path.

    Loader emits ``(N, 3, 4, 4)`` from the PNG dir; the bundled MLP
    expects ``(N, 48)``. ``data.input_metadata.shape: [48]`` forces the
    adapter to flatten before the forward pass — if the adapter is
    silently a no-op the ONNX Runtime call would raise.
    """
    pytest.importorskip("maraboupy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")

    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    non_batch_numel = _seed_image_fixture(fixture_dir, num_samples=2)
    _build_tiny_mlp_onnx(
        fixture_dir / "model.onnx",
        in_features=non_batch_numel,
        hidden=8,
        out_features=3,
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_image_config(
        config_dir / "e2e_marabou_cli_image.yaml",
        fixture_dir,
        non_batch_numel=non_batch_numel,
    )

    run_dir = tmp_path / "run"
    proc = _run_raitap(config_dir, "e2e_marabou_cli_image", run_dir)
    _assert_succeeded(proc, run_dir)
    _assert_robustness_persisted(proc, run_dir)


def test_raitap_cli_marabou_emits_html_report(tmp_path: Path) -> None:
    """Image fixture + HTMLReporter — covers the reporting hand-off."""
    pytest.importorskip("maraboupy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")
    pytest.importorskip("jinja2")

    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    non_batch_numel = _seed_image_fixture(fixture_dir, num_samples=2)
    _build_tiny_mlp_onnx(
        fixture_dir / "model.onnx",
        in_features=non_batch_numel,
        hidden=8,
        out_features=3,
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_image_config(
        config_dir / "e2e_marabou_cli_image_html.yaml",
        fixture_dir,
        non_batch_numel=non_batch_numel,
        with_reporting=True,
    )

    run_dir = tmp_path / "run"
    proc = _run_raitap(config_dir, "e2e_marabou_cli_image_html", run_dir)
    _assert_succeeded(proc, run_dir)
    _assert_robustness_persisted(proc, run_dir)

    reports = list((run_dir / "reports").rglob("*.html")) if (run_dir / "reports").exists() else []
    if not reports:
        reports = list(run_dir.rglob("*.html"))
    assert reports, (
        "HTMLReporter did not emit a report.\n"
        f"Run dir contents: {sorted(p.relative_to(run_dir) for p in run_dir.rglob('*'))}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
