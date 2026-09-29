# src/utils/load_environment.py

import os
import sys
from pathlib import Path
import yaml


class LoadEnvironment:
    """
    Loads and resolves environment-specific configuration paths and settings 
    from environment.yaml. Supports auto-detection across local_dev, local_prod, 
    and Google Colab.
    """

    def __init__(self, config_rel_path: str = "configs/environment.yaml"):
        self.config_rel_path = config_rel_path
        self._repo_root = self._find_repo_root()
        self.config_path = self._repo_root / self.config_rel_path
        
        self.ENVIRONMENT: str = ""
        self.IS_COLAB: bool = False
        self.ROOT_BASE_DIR: Path = Path()
        self.DATABASE_DIR: Path = Path()
        self.CHECKPOINT_DIR: Path = Path()
        self.OUTPUT_DIR: Path = Path()

        self._load_and_parse()

    def _find_repo_root(self) -> Path:
        """Determines the repository root directory relative to this file."""
        return Path(__file__).resolve().parent.parent.parent

    @staticmethod
    def is_colab_environment() -> bool:
        """Detects if execution is running inside a Google Colab instance."""
        return "COLAB_GPU" in os.environ or "Gdrive" in os.environ or "google.colab" in sys.modules

    def _load_and_parse(self) -> None:
        """Reads environment.yaml and sets typed attributes."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Environment configuration not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        env_override = raw_cfg.get("active_environment", "auto")

        if env_override == "auto":
            self.ENVIRONMENT = "colab" if self.is_colab_environment() else "local_dev"
        else:
            self.ENVIRONMENT = env_override

        if self.ENVIRONMENT not in raw_cfg["environments"]:
            valid_envs = list(raw_cfg["environments"].keys())
            raise ValueError(f"Unknown environment '{self.ENVIRONMENT}'. Choose from {valid_envs}")

        env_params = raw_cfg["environments"][self.ENVIRONMENT]

        self.IS_COLAB = bool(env_params["IS_COLAB"])
        self.ROOT_BASE_DIR = Path(env_params["ROOT_BASE_DIR"])
        self.DATABASE_DIR = Path(env_params["DATABASE_DIR"])
        self.CHECKPOINT_DIR = Path(env_params["CHECKPOINT_DIR"])
        self.OUTPUT_DIR = Path(env_params["OUTPUT_DIR"])

        # Auto-create writeable directories if missing
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Returns the resolved environment settings as a dictionary."""
        return {
            "ENVIRONMENT": self.ENVIRONMENT,
            "IS_COLAB": self.IS_COLAB,
            "ROOT_BASE_DIR": self.ROOT_BASE_DIR,
            "DATABASE_DIR": self.DATABASE_DIR,
            "CHECKPOINT_DIR": self.CHECKPOINT_DIR,
            "OUTPUT_DIR": self.OUTPUT_DIR,
        }

    def __repr__(self) -> str:
        return (
            f"<LoadEnvironment(ENVIRONMENT='{self.ENVIRONMENT}', "
            f"IS_COLAB={self.IS_COLAB}, "
            f"DATABASE_DIR='{self.DATABASE_DIR}')>"
        )


if __name__ == "__main__":
    # Smoke test execution
    env = LoadEnvironment()
    print("=== Environment Loaded Successfully ===")
    print(repr(env))
    print("\nResolved Paths:")
    for key, value in env.to_dict().items():
        print(f"  {key:15s}: {value}")
        