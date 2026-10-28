"""Additional pattern adapters without changing the versioned Bull Flag files."""
from dataclasses import asdict
from datetime import datetime
import json
import re
from pathlib import Path

import numpy as np

from scripts.bull_flag_strategy import BULL_FLAG, BullFlagMixin
from scripts.research_strategies import BASELINE, prepare_signals
from scripts.strategy_backtest import require
from scripts.swing_patterns import DETECTORS, STRATEGIES

PATTERNS=(BULL_FLAG,*STRATEGIES)
ROOT=Path(__file__).resolve().parents[1]
CONFIG='config/chart_pattern_companions.json'


def configuration(base_dir=ROOT):
    from scripts.pipeline_postflight import _read
    from scripts.research_dataset import strict_json
    path=Path(base_dir)/CONFIG
    if not path.exists():return None
    value=strict_json(_read(path,4096))
    expected=dict(schema='jbravo.additional-chart-patterns.v1',enabled=True,
        strategy_ids=list(STRATEGIES),selection_policy='UNIFIED_BEST_SETUPS_V1',
        max_scan_seconds=600,max_new_entries_total=3,max_positions_total=4,
        allocation_pct_remaining_cash=.05,paper_only=True,research_qualified=False)
    if value==dict(expected,enabled=False):return None
    require(value==expected,'additional_pattern_configuration_changed')
    return value


def enabled_ids(base_dir=ROOT):
    policy=configuration(base_dir)
    return tuple(policy['strategy_ids']) if policy else ()


def plan_from_row(symbol,row):
    details=row.get('entry_details',{})
    if not isinstance(details,dict):return None
    value=details.get('pattern_exit')
    if not isinstance(value,dict):return None
    if value.get('pattern')==BULL_FLAG:
        from scripts.bull_flag_strategy import plan_from_row as original
        return original(symbol,row)
    if (value.get('pattern') not in STRATEGIES or value.get('symbol',symbol)!=symbol or
        type(value.get('version')) is not int or value.get('version')!=1 or value.get('status')!='confirmed' or
        not isinstance(value.get('signal_time'),str) or
        re.fullmatch(r'\d{4}-\d{2}-\d{2}T00:00:00',value['signal_time']) is None):return None
    try:
        datetime.fromisoformat(value['signal_time'])
        numbers=[value[k] for k in ('entry','resistance','invalidation','target')]
        if any(type(n) not in (int,float) or not np.isfinite(n) or n<=0 for n in numbers):return None
        entry,resistance,stop,target=numbers
        if not stop<resistance<entry<target:return None
        json.dumps(value,allow_nan=False)
    except (ValueError,TypeError,KeyError):return None
    return dict(value,symbol=symbol)


def prepare_many(raw,split,fold,strategy_ids,*,enrich_common=False,checkpoint=None):
    strategy_ids=tuple(strategy_ids)
    require(strategy_ids and len(strategy_ids)==len(set(strategy_ids)) and
            set(strategy_ids)<=set(STRATEGIES),'unknown_pattern_strategy')
    from scripts.setup_selection import enrich
    frame=prepare_signals(raw,split,BASELINE,fold)
    if enrich_common:
        # Technical ranking features do not depend on which pattern matched.
        # Compute them once, then copy them into each independent strategy frame.
        frame=enrich(frame,split)
    candidate=((split.close>split.open)&(split.volume>=split.volume.shift(1).rolling(20).mean()*1.5)&
               (split.close>split.close.shift(1).rolling(2).max()))
    days=frame.index[frame.entry_ready & candidate.reindex(frame.index).fillna(False)]
    output={}
    for strategy_id in strategy_ids:
        if checkpoint is not None:checkpoint()
        current=frame.copy(deep=True)
        current['score']=np.nan
        plans={}
        for day in days:
            if checkpoint is not None:checkpoint()
            signal=DETECTORS[strategy_id](split.loc[:day].tail(141))
            if signal is None:continue
            plan=asdict(signal)
            factor=float(raw.loc[day,'close']/split.loc[day,'close'])
            for key in ('entry','resistance','invalidation','target'):plan[key]*=factor
            plans[day]=plan
            current.loc[day,'score']=100*(plan['target']/plan['entry']-1)+min(signal.metrics['volume_ratio'],10)
        current['entry_details']=[dict(signal=d in plans,failures=[] if d in plans else ['NO_CONFIRMED_PATTERN'],
                                       pattern_exit=plans.get(d)) for d in current.index]
        output[strategy_id]=current
    return output


def prepare(raw,split,fold,strategy_id):
    require(strategy_id in STRATEGIES,'unknown_pattern_strategy')
    return prepare_many(raw,split,fold,(strategy_id,))[strategy_id]


class PatternMixin(BullFlagMixin):
    def _open_position(self,symbol,row,day,signal_row):
        plan=plan_from_row(symbol,signal_row)
        if plan is None or not plan['invalidation']<float(row['open'])<plan['target']:return
        # Skip only BullFlagMixin's hardcoded entry validator. Its existing
        # stop/target methods and the next portfolio's ownership hooks remain.
        super(BullFlagMixin,self)._open_position(symbol,row,day,signal_row)
        if symbol in self.positions:self.pattern_plans[symbol]=plan


def scanner_portfolio_class(base):
    class PatternScanner(PatternMixin,base):pass
    return PatternScanner


def research_portfolio_class():
    from scripts.research_simulation import ResearchPortfolio
    class PatternPortfolio(PatternMixin,ResearchPortfolio):
        def __init__(self,data,**kwargs):super().__init__(data,method=BASELINE,**kwargs)
    return PatternPortfolio
