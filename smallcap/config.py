"""配置加载。仓库根目录的 config.yaml 是唯一事实来源。"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load()
DATA_ROOT = ROOT / CFG["data"]["root"]
