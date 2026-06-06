"""Public decorator surface for registering model backends.

    from raitap import backends
    from raitap.types import Capability

    @backends.register(provides={Capability.AUTOGRAD}, extensions={".pth", ".pt"})
    class MyBackend(ModelBackend): ...

The decorator sets the ``provides`` + ``extensions`` class constants and indexes
the backend by file extension, so ``model._load_from_path`` resolves it without a
hardcoded dispatch. In-tree only (no external entry-point plugins yet).
"""

from __future__ import annotations

from raitap.models.registration import register

__all__ = ["register"]
