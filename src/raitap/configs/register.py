from hydra.core.config_store import ConfigStore

from .schema import AppConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="schema", name="config", node=AppConfig)
