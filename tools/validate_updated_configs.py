import importlib.util
import pathlib

import yaml

PATTERNS = [
    "lego_feature_teacher_full_rehab_masked_white.yaml",
    "lego_feature_teacher_full_rehab_masked_white_smoke.yaml",
    "lego_feature_teacher_full_rehab_masked_white_smoke_s1.yaml",
    "lego_feature_teacher_full_rehab_masked_white_smoke_s1_tuned.yaml",
    "lego_feature_teacher_full_rehab_masked_white_smoke_s1_nomask.yaml",
    "lego_feature_teacher_full_rehab_masked_white_smoke_s2.yaml",
    "lego_feature_teacher_full_rehab_masked_white_stage2.yaml",
    "lego_feature_teacher_full_rehab_masked_white_repro.yaml",
]

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_validator():
    schema_path = PROJECT_ROOT / "distill" / "config_schema.py"
    spec = importlib.util.spec_from_file_location("config_schema", schema_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.validate_config_dict


def main() -> None:
    validate_config_dict = _load_validator()
    for pattern in PATTERNS:
        cfg_path = CONFIG_ROOT / pattern
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        validate_config_dict(data)
        print(f"Validated: {cfg_path}")


if __name__ == "__main__":
    main()
