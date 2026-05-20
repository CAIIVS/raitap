"""Fixture plugin that crashes at import — exercises discovery failure isolation.

Its ``raitap`` pin is satisfiable, so the version check passes and
``discover_third_party_adapters`` proceeds to ``ep.load()``, which raises here.
The discovery loop must catch this, log a warning, and keep going.
"""

raise RuntimeError("fake plugin boom at import time")
