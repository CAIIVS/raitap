"""Public decorator surface for registering model backends.

    from raitap import backends
    from raitap.types import Capability

    @backends.register(provides=frozenset({Capability.AUTOGRAD}))
    class MyBackend(ModelBackend): ...

Backends are not Hydra-config adapters (no registry, no entry-point plugins);
the decorator only sets + type-checks the required ``provides``
class constant at the decoration site.
"""

from __future__ import annotations

from raitap.models.registration import register

__all__ = ["register"]
