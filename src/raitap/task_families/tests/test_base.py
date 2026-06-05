from __future__ import annotations

from raitap.task_families.base import ExplainContext, ForwardContext, TaskFamily


def test_taskfamily_is_runtime_checkable_protocol() -> None:
    class Dummy:
        kind = "classification"
        output_space = None

        def validate_payload(self, payload: object) -> None: ...
        def adapt_loaded_inputs(self, tensor: object) -> object: ...
        def validate_inputs(self, tensor: object) -> None: ...
        def load_labels(self, cfg: object, *, tensor: object, sample_ids: object) -> object: ...
        def extract_forward(self, ctx: object) -> object: ...
        def explain(self, ctx: object) -> list: ...
        def metrics_inputs(
            self, config: object, forward_output: object, labels: object
        ) -> object: ...
        def supports_robustness(self) -> bool: ...
        def prediction_summaries(self, payload: object) -> list | None: ...
        @property
        def allows_preprocessing(self) -> bool: ...

    assert isinstance(Dummy(), TaskFamily)

    class Incomplete:
        kind = "classification"
        # missing output_space and every method

    assert not isinstance(Incomplete(), TaskFamily)


def test_context_dataclasses_carry_extras() -> None:
    fwd = ForwardContext(backend=object(), inputs=[1, 2], extras={"tokenizer": "x"})
    exp = ExplainContext(prepared=object(), forward_output=object(), data=object(), extras={})
    assert fwd.extras["tokenizer"] == "x"
    assert exp.extras == {}
