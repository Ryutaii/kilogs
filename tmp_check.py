"""Compatibility shim for the legacy tmp_check helper.

Prefer invoking ``python tools/feature_mask_stats.py`` directly instead.
"""

from tools.feature_mask_stats import main


if __name__ == "__main__":
    main()
