import copy
from typing import Optional


def merge_generate_cfgs(base_generate_cfg: Optional[dict], new_generate_cfg: Optional[dict]) -> dict:
    generate_cfg: dict = copy.deepcopy(base_generate_cfg or {})
    if new_generate_cfg:
        for k, v in new_generate_cfg.items():
            if k == 'stop':
                stop = generate_cfg.get('stop', [])
                stop = stop + [s for s in v if s not in stop]
                generate_cfg['stop'] = stop
            else:
                generate_cfg[k] = v
    return generate_cfg