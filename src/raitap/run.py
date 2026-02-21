import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Print the configuration loaded by Hydra
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Selected model: {cfg.model.name}")

    # Check where Hydra is running from
    import os

    print(f"Current working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
