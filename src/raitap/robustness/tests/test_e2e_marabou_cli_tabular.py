"""End-to-end CLI smoke test for ``MarabouAssessor``.

Drives ``python -m raitap.run`` as a subprocess against a tiny synthetic
3-feature Linear+ReLU MLP exported to ONNX, exercising every layer
between the user and the solver:

* Hydra config resolution + interpolation
* :class:`raitap.models.Model` + :class:`raitap.data.Data` construction
* Backend shape adapter (no reshape needed here, but the call site runs)
* Pipeline forward pass through ONNX Runtime
* :class:`MarabouAssessor.assess` + ``verify_sample`` per-sample loop
* Result serialisation (``robustness_data.pt``)

The existing ``test_e2e_marabou_acas_xu`` tests call ``verify_sample``
directly with a pre-loaded ONNX path, so they miss every layer above
that — this CLI variant catches the regressions encountered while
wiring UC1 (HOME interpolation, MarabouUtils.Equation wrapper, mixed
image+CSV dir, etc.).

Skips when ``maraboupy`` or ``onnx`` is unavailable.
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


def _build_tiny_mlp_onnx(target_path: Path) -> None:
    """Write a 3-input → 4-hidden → 3-output ReLU MLP to ``target_path``.

    Pure Linear + ReLU so the maraboupy ONNX parser accepts it without
    hitting the ``Shape``/``Reshape``/``Gather`` op limitations that
    ``nn.Flatten`` (or any dynamic reshape) would introduce. Batch dim is
    declared as the symbolic ``"batch"`` so the backend shape adapter
    treats it as dynamic.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    b1 = np.zeros(4, dtype=np.float32)
    w2 = rng.standard_normal((4, 3)).astype(np.float32) * 0.5
    b2 = np.zeros(3, dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["h1_relu"]),
            helper.make_node("Gemm", ["h1_relu", "w2", "b2"], ["logits"]),
        ],
        "tiny_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
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


def _write_config(target_path: Path, fixture_dir: Path) -> None:
    """Render a standalone Hydra config that points at the fixture dir."""
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


def test_raitap_cli_marabou_tabular_smoke(tmp_path: Path) -> None:
    """Run ``raitap`` end-to-end against a tiny synthetic MLP + tabular CSV.

    Asserts the subprocess exits zero and the assessor persisted its
    verdict tensor — that's the minimum signal that every layer between
    the CLI and the solver executed.
    """
    pytest.importorskip("maraboupy")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    _build_tiny_mlp_onnx(fixture_dir / "model.onnx")

    # Two tabular samples with three features each — small enough that
    # Marabou returns near-instantly per sample.
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
    _write_config(config_dir / "e2e_marabou_cli_tabular.yaml", fixture_dir)

    run_dir = tmp_path / "run"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "raitap.run",
            "--config-dir",
            str(config_dir),
            "--config-name",
            "e2e_marabou_cli_tabular",
            f"hydra.run.dir={run_dir.as_posix()}",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )

    assert proc.returncode == 0, (
        f"raitap CLI exited with code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    persisted = run_dir / "robustness" / "marabou_linf" / "robustness_data.pt"
    legacy = run_dir / "robustness_data.pt"
    assert persisted.exists() or legacy.exists(), (
        "Marabou assessor did not persist results.\n"
        f"Tried: {persisted}\n"
        f"Tried: {legacy}\n"
        f"Run dir contents: {sorted(p.relative_to(run_dir) for p in run_dir.rglob('*'))}\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
