import os
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    with open(path or os.path.join(ROOT, 'config.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


def holdout_start(cfg):
    return pd.Timestamp(cfg['time']['FINAL_HOLDOUT_START'])


def dev_cut(cfg):
    """Exclusive upper bound of every timestamp the evaluation may touch."""
    return holdout_start(cfg)


def guard_holdout(cfg, end):
    """Refuse to touch data at/after FINAL_HOLDOUT_START while it is locked (until step 8)."""
    if cfg['time']['HOLDOUT_LOCKED'] and pd.Timestamp(end) >= holdout_start(cfg):
        raise PermissionError(f"final holdout (>= {holdout_start(cfg).date()}) is locked; requested end {end}")
