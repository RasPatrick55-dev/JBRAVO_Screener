"""The historical pilot remains a regression fixture, never the daily universe."""
from scripts.daily_strategy_scan import load_config


def pilot_config():
    cfg = load_config()
    for key in ('universe', 'max_capture_seconds', 'max_total_seconds', 'max_provider_bytes'):
        cfg.pop(key)
    cfg.update(schema='jbravo.daily-strategy-config.v1', max_provider_gets=24, max_scan_seconds=600,
        symbols=['ALK','FN','HMY','LOGI','MIRM','NVST','PLUG','SKY','SRRK','STLD'])
    return cfg
