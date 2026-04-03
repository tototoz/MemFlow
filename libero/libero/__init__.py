import os
import yaml

# Default path root is the library directory itself
_libero_root = os.path.dirname(os.path.abspath(__file__))

# Path to Hydra config (used by train.py)
_HYDRA_CONFIG = os.path.join(_libero_root, "../configs/config.yaml")


def get_libero_path(query_key):
    """Get libero paths from environment variables, then Hydra config, then defaults."""
    # Check environment variables first (highest priority)
    env_mapping = {
        "benchmark_root": "LIBERO_BENCHMARK_ROOT",
        "bddl_files": "LIBERO_BDDL_PATH",
        "init_states": "LIBERO_INIT_STATES_PATH",
        "datasets": "LIBERO_DATASETS_PATH",
        "assets": "LIBERO_ASSETS_PATH",
    }
    if query_key in env_mapping and os.environ.get(env_mapping[query_key]):
        return os.environ.get(env_mapping[query_key])

    # Try to read from Hydra config.yaml
    if os.path.exists(_HYDRA_CONFIG):
        with open(_HYDRA_CONFIG, "r") as f:
            cfg = yaml.safe_load(f)
        path_mapping = {
            "datasets": cfg.get("folder"),
            "bddl_files": cfg.get("bddl_folder"),
            "init_states": cfg.get("init_states_folder"),
        }
        if query_key in path_mapping and path_mapping[query_key]:
            return path_mapping[query_key]

    # Fallback to LIBERO library defaults
    defaults = {
        "benchmark_root": _libero_root,
        "bddl_files": os.path.join(_libero_root, "./bddl_files"),
        "init_states": os.path.join(_libero_root, "./init_files"),
        "datasets": os.path.join(_libero_root, "../datasets"),
        "assets": os.path.join(_libero_root, "./assets"),
    }
    if query_key in defaults:
        return defaults[query_key]

    raise ValueError(f"Path for '{query_key}' not found")
