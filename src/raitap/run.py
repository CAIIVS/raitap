import hydra
from configs.register import register_configs
from configs.schema import AppConfig
from omegaconf import OmegaConf

register_configs()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: AppConfig):
    # Print the configuration loaded by Hydra
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Selected model: {cfg.model.name}")
    print(f"Transparency: {cfg.transparency.framework} - {cfg.transparency.algorithm}")
    print(f"Dataset: {cfg.data.name} - {cfg.data.description}")

    # Check where Hydra is running from
    import os

    print(f"Current working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
