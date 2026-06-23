from __future__ import annotations


def test_register_indexes_backend_by_extension() -> None:
    from raitap.models import registration
    from raitap.models.registration import (
        backend_for_extension,
        register,
        supported_model_formats,
    )

    # ``@register`` mutates the module-global index at class-statement time;
    # snapshot before and restore in-place after so the fake extensions don't leak.
    saved = registration._BACKENDS_BY_EXTENSION.copy()
    try:
        # ``register`` is bound to ``ModelBackend``; this fake intentionally is not one
        # (the registry only stores/looks up the class object), so silence the bound check.
        @register(provides=frozenset(), extensions={".fake1", ".fake2"})  # pyright: ignore[reportArgumentType]
        class _FakeBackend:  # minimal; registry only needs the class
            pass

        assert backend_for_extension(".fake1") is _FakeBackend
        assert backend_for_extension(".fake2") is _FakeBackend
        assert backend_for_extension(".nope") is None
        assert {".fake1", ".fake2"} <= set(supported_model_formats())
        assert supported_model_formats() == sorted(supported_model_formats())
    finally:
        registration._BACKENDS_BY_EXTENSION.clear()
        registration._BACKENDS_BY_EXTENSION.update(saved)


def test_real_backends_resolve_by_extension() -> None:
    from raitap.models.onnx_backend import OnnxBackend
    from raitap.models.registration import backend_for_extension
    from raitap.models.torch_backend import TorchBackend

    assert backend_for_extension(".onnx") is OnnxBackend
    assert backend_for_extension(".pth") is TorchBackend
    assert backend_for_extension(".pt") is TorchBackend
