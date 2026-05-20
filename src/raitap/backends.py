"""Public decorator surface for registering model backends.

    from raitap import backends

    @backends.register(supports_torch_autograd=True)
    class MyBackend(ModelBackend): ...

Backends are not Hydra-config adapters (no registry, no entry-point plugins);
the decorator only sets + type-checks the required ``supports_torch_autograd``
class constant at the decoration site.
"""

from __future__ import annotations

from raitap.models.registration import register

__all__ = ["register"]
