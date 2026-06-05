from __future__ import annotations

from typing import TYPE_CHECKING, cast

from raitap.task_families.base import ExplainContext, ForwardContext, TaskFamily

if TYPE_CHECKING:
    from raitap.data.data import Data
    from raitap.models.backend import ModelBackend
    from raitap.pipeline.outputs import ForwardOutput
    from raitap.transparency.phase import PreparedExplainer


def test_taskfamily_is_runtime_checkable_protocol() -> None:
    class Dummy:
        kind = "classification"
        fixed_output_space = None

        def validate_payload(self, payload: object) -> None:
            pass

        def adapt_loaded_inputs(self, tensor: object) -> object:
            pass

        def validate_inputs(self, tensor: object) -> None:
            pass

        def load_labels(self, cfg: object, *, tensor: object, sample_ids: object) -> object:
            pass

        def validate_labels(self, labels: object) -> None:
            pass

        def extract_forward(self, ctx: object) -> object:
            pass

        def payload_batch_size(self, payload: object) -> int:
            raise NotImplementedError

        def explain(self, ctx: object) -> list:
            raise NotImplementedError

        def metrics_inputs(self, config: object, forward_output: object, labels: object) -> object:
            pass

        def supports_robustness(self) -> bool:
            raise NotImplementedError

        def prediction_summaries(
            self, payload: object, *, sample_ids: object = None, targets: object = None
        ) -> list | None:
            pass

        def matches_model(self, model: object) -> bool:
            raise NotImplementedError

        @property
        def allows_preprocessing(self) -> bool:
            raise NotImplementedError

    assert isinstance(Dummy(), TaskFamily)

    class Incomplete:
        kind = "classification"
        # missing fixed_output_space and every method

    assert not isinstance(Incomplete(), TaskFamily)


def test_context_dataclasses_carry_extras() -> None:
    fwd = ForwardContext(
        backend=cast("ModelBackend", object()), inputs=[1, 2], extras={"tokenizer": "x"}
    )
    exp = ExplainContext(
        prepared=cast("PreparedExplainer", object()),
        forward_output=cast("ForwardOutput", object()),
        data=cast("Data", object()),
        extras={},
    )
    assert fwd.extras["tokenizer"] == "x"
    assert exp.extras == {}
