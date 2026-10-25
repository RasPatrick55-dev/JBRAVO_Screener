"""Versioned scanner strategies. Registration is distinct from research approval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from scripts.research_strategies import CHALLENGER, prepare_signals


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    version: str
    research_method: str
    status: str
    warmup_sessions: int
    buy_rules: tuple[str, ...]
    paper_scan_enabled: bool = False

    def describe(self):
        result = asdict(self)
        # Bind the actual rule/indicator/consumer code, not just display labels.
        directory = Path(__file__).resolve().parent
        result['source_sha256'] = {
            name: sha256((directory / name).read_bytes()).hexdigest()
            for name in ('strategy_library.py', 'research_strategies.py', 'indicators.py',
                         'research_simulation.py', 'backtest_core.py')
        }
        definition = dict(result)
        # Adding another catalog entry must not change this strategy's logical
        # signal IDs. The catalog file is still bound in each batch's sources.
        definition['source_sha256'] = {name: value for name, value in result['source_sha256'].items()
                                       if name != 'strategy_library.py'}
        result['definition_sha256'] = sha256(json.dumps(
            definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result['research_qualified'] = False
        return result

    def prepare(self, raw, split, fold):
        return prepare_signals(raw, split, self.research_method, fold, self.warmup_sessions)


_STRATEGIES = {
    CHALLENGER: Strategy(
        CHALLENGER, 'Trend Pullback / Reclaim', '2', CHALLENGER, 'PAPER_TEST_CANDIDATE', 250,
        ('close > MA50 > MA200; MA50 higher than ten sessions ago',
         '2–5-session pullback after a close above EMA20',
         'pullback closes >= MA50 and <= EMA20 + 0.5 ATR; lows >= EMA20 - 1.5 ATR',
         'at least one pullback close <= EMA20',
         'bullish reclaim close > EMA20 and previous high; no lower pullback trough',
         'volume > preceding 20-session mean; positive ATR and all required features ready',
         'held/pending symbols cannot reenter; new pullback must start after consumed reclaim',
         'qualifying closes consume before ranking/capacity, including unselected setups'),
        paper_scan_enabled=True,
    ),
}


def get_strategy(strategy_id):
    try:
        return _STRATEGIES[strategy_id]
    except KeyError:
        raise ValueError('Unknown versioned scanner strategy: ' + str(strategy_id)) from None


def list_strategies():
    return [strategy.describe() for strategy in _STRATEGIES.values()]


def select_strategies(ids=None):
    chosen = [key for key, value in _STRATEGIES.items() if value.paper_scan_enabled] if ids is None else list(ids)
    if not chosen or len(chosen) != len(set(chosen)):
        raise ValueError('Select at least one strategy, without duplicates')
    strategies = [get_strategy(strategy_id) for strategy_id in chosen]
    if any(not strategy.paper_scan_enabled for strategy in strategies):
        raise ValueError('Strategy is registered but not enabled for paper scanning')
    return strategies
