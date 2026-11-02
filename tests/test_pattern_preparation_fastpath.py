"""Parity with the accepted validator through real detectors and preparation."""
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tests')]

from scripts import pattern_strategy as pattern, swing_patterns as swing
from scripts import strategy_prepared_market as prepared, strategy_market_replay as market
from scripts.strategy_library import get_strategy
from scripts.research_strategies import CHALLENGER
from scripts.bull_flag_strategy import BULL_FLAG
from scripts.strategy_runtime_diagnostics import checkpoint, StrategyBudgetError
from pattern_fixture_support import (fixtures, pattern_frame, reference_input,
                                     baseline_prepare_many, shared_fixture)

STRATEGIES = swing.STRATEGIES


def deny_network(*args, **kwargs):
    raise AssertionError('No network in pattern preparation tests')


class PatternInputParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for target, name in ((socket.socket, 'connect'), (socket, 'create_connection')):
            guard = patch.object(target, name, side_effect=deny_network)
            guard.start()
            cls.addClassCleanup(guard.stop)
        cls.supplied = fixtures(8, 600)

    def assert_same_input(self, frame):
        try:
            expected = reference_input(frame)
        except Exception as error:
            with self.assertRaises(type(error)):
                swing._input(frame)
            return
        actual = swing._input(frame)
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertIsNotNone(actual)
            pd.testing.assert_frame_equal(expected[0], actual[0], check_exact=True)
            pd.testing.assert_series_equal(expected[1], actual[1], check_exact=True)
            self.assertEqual(struct.pack('!d', expected[2]), struct.pack('!d', actual[2]))

    def temporary(self):
        # The generated cleanup target must stay inside the explicitly chosen temp root.
        base = REPO.resolve()
        context = tempfile.TemporaryDirectory(prefix='jbravo-pattern-', dir=base)
        self.assertTrue(Path(context.name).resolve().is_relative_to(base))
        self.addCleanup(context.cleanup)
        return Path(context.name)

    def test_positive_geometries_and_all_historical_windows_match(self):
        for raw, split, fold in self.supplied:
            for end in (20, 31, 100, 270, 450, 599, 600):
                window = split.iloc[:end].tail(141)
                self.assert_same_input(window)
                with patch.object(swing, '_input', reference_input):
                    expected = {sid: swing.DETECTORS[sid](window) for sid in STRATEGIES}
                actual = {sid: swing.DETECTORS[sid](window) for sid in STRATEGIES}
                self.assertEqual(expected, actual)
        for sid in STRATEGIES:
            self.assertIsNotNone(swing.DETECTORS[sid](pattern_frame(sid)))

    def test_integer_volume_and_dtype_fallbacks_preserve_exact_values(self):
        original = pattern_frame(STRATEGIES[0])
        frames = [original, original.astype('float64'), original.astype('float32'),
                  original.astype(str), original.astype('Float64')]
        unsigned = original.copy()
        unsigned['volume'] = unsigned.volume.astype('uint64')
        frames.append(unsigned)
        for dtype, maximum in (('int64', 2**63-1), ('uint64', 2**64-1)):
            large = original.copy()
            large['volume'] = pd.Series(maximum, index=large.index, dtype=dtype)
            frames.append(large)
        for frame in frames:
            with self.subTest(dtypes=tuple(str(v) for v in frame.dtypes)):
                self.assert_same_input(frame)

    def test_large_integer_prices_do_not_use_float_comparison_fastpath(self):
        bars = pattern_frame(STRATEGIES[0]).astype('int64')
        base = 2**53
        bars[['open', 'close', 'high', 'low']] = [base+2, base+4, base+3, base]
        self.assertIsNone(reference_input(bars))
        self.assertIsNone(swing._input(bars))

    def test_malformed_and_non_daily_inputs_keep_original_refusal(self):
        original = pattern_frame(STRATEGIES[0])
        frames = []
        for column, value in (('open', np.nan), ('volume', 0), ('high', 0.), ('close', np.inf)):
            frame = original.copy()
            frame.iloc[-5, frame.columns.get_loc(column)] = value
            frames.append(frame)
        duplicate = original.copy()
        duplicate.index = list(duplicate.index[:-1])+[duplicate.index[-2]]
        intraday = original.copy()
        index = list(intraday.index)
        index[-1] = index[-2]+pd.Timedelta(hours=1)
        intraday.index = pd.DatetimeIndex(index)
        frames.extend([duplicate, intraday, original.iloc[::-1], original.reset_index(drop=True),
                       original.drop(columns='low')])
        frames.extend(pd.concat([original, original[[name]]], axis=1)
                      for name in original.columns)
        for frame in frames:
            self.assert_same_input(frame)

    def test_float_boundaries_and_volume_ratio_preserve_bits(self):
        original = pattern_frame(STRATEGIES[0]).astype('float64')
        for value in (np.nextafter(1500., 0.), 1500., np.nextafter(1500., np.inf)):
            frame = original.copy()
            frame.loc[frame.index[-21:-1], 'volume'] = 1000.
            frame.loc[frame.index[-1], 'volume'] = value
            self.assert_same_input(frame)
        for direction in (0., np.inf):
            frame = original.copy()
            maximum = frame[['open', 'close']].max(axis=1)
            frame['high'] = np.nextafter(maximum.to_numpy(), direction)
            self.assert_same_input(frame)
        zoned = original.copy()
        zoned.index = zoned.index.tz_localize('America/New_York')
        self.assert_same_input(zoned)

    def test_fastpath_removes_redundant_numeric_conversion(self):
        frame = pattern_frame(STRATEGIES[0])
        calls = []
        original = pd.to_numeric
        def numeric(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        with patch.object(pd, 'to_numeric', numeric):
            expected = reference_input(frame)
            self.assertEqual(len(calls), 5)
        def refuse(*args, **kwargs):
            raise AssertionError('already numeric')
        with patch.object(pd, 'to_numeric', refuse):
            actual = swing._input(frame)
        pd.testing.assert_frame_equal(expected[0], actual[0], check_exact=True)

    def test_private_normalized_frame_remains_independent_of_input(self):
        frame = pattern_frame(STRATEGIES[0])
        before = frame.copy(deep=True)
        prior, current, ratio = swing._input(frame)
        prior.loc[prior.index[0], 'open'] = 999999.
        pd.testing.assert_frame_equal(frame, before, check_exact=True)

    def test_real_caller_frames_match_subsets_order_enrichment_and_split_units(self):
        subsets = [STRATEGIES, tuple(reversed(STRATEGIES)), *[(sid,) for sid in STRATEGIES]]
        for raw, split, fold in self.supplied:
            for selected in subsets:
                expected = baseline_prepare_many(raw, split, fold, selected, enrich_common=True)
                actual = pattern.prepare_many(raw, split, fold, selected, enrich_common=True)
                self.assertEqual(list(actual), list(selected))
                for sid in selected:
                    pd.testing.assert_frame_equal(expected[sid], actual[sid], check_exact=True)
        for sid in STRATEGIES:
            split = pattern_frame(sid)
            for factor in (.5, 1., 2.):
                raw = split.copy()
                raw[['open', 'high', 'low', 'close']] *= factor
                fold = dict(warmup_start=str(split.index[0].date()), test_start=str(split.index[270].date()),
                            test_end=str(split.index[-1].date()))
                expected = baseline_prepare_many(raw, split, fold, STRATEGIES)
                actual = pattern.prepare_many(raw, split, fold, STRATEGIES)
                for name in STRATEGIES:
                    pd.testing.assert_frame_equal(expected[name], actual[name], check_exact=True)

    def test_input_immutability_and_causal_prefix(self):
        for raw, split, fold in self.supplied[:4]:
            raw_before, split_before = raw.copy(deep=True), split.copy(deep=True)
            before = pattern.prepare_many(raw, split, fold, STRATEGIES, enrich_common=True)
            pd.testing.assert_frame_equal(raw, raw_before, check_exact=True)
            pd.testing.assert_frame_equal(split, split_before, check_exact=True)
            extra = split.tail(3).copy()
            extra.index = pd.bdate_range(split.index[-1]+pd.offsets.BDay(), periods=3)
            extra[['open', 'high', 'low', 'close']] *= 4
            future = pd.concat([split, extra])
            raw_extra = extra.copy()
            raw_extra[['open', 'high', 'low', 'close']] *= 2
            after = pattern.prepare_many(pd.concat([raw, raw_extra]), future,
                dict(fold, test_end=str(extra.index[-1].date())), STRATEGIES, enrich_common=True)
            for sid in STRATEGIES:
                pd.testing.assert_frame_equal(before[sid], after[sid].loc[before[sid].index],
                                              check_exact=True, check_freq=False)

    def test_checkpoint_order_and_partial_failure_are_unchanged(self):
        raw, split, fold = self.supplied[0]
        counts = []
        for implementation in (baseline_prepare_many, pattern.prepare_many):
            calls = []
            def check():
                calls.append(1)
                if len(calls) == 3:
                    raise StrategyBudgetError('STOP_FILE_PRESENT', 'SHARED_PATTERN_PREPARATION')
            with self.assertRaises(StrategyBudgetError) as caught:
                implementation(raw, split, fold, STRATEGIES, checkpoint=check)
            self.assertEqual(caught.exception.code, 'STOP_FILE_PRESENT')
            counts.append(len(calls))
        self.assertEqual(counts, [3, 3])

    def test_actual_shared_group_expired_deadline_and_stop_remain_effective(self):
        raw, split, fold = self.supplied[0]
        dataset = SimpleNamespace(raw={'SYN': raw}, features={'SYN': split})
        folder = self.temporary()
        stop = folder/'STOP'
        stop.write_text('synthetic stop', encoding='utf-8')
        for deadline, path, code in ((time.monotonic()-1, None, 'WALL_CLOCK_BUDGET_EXCEEDED'),
                                     (time.monotonic()+60, stop, 'STOP_FILE_PRESENT')):
            with patch.object(swing, '_input', side_effect=AssertionError('must stop first')):
                with self.assertRaises(StrategyBudgetError) as caught:
                    market.prepare_pattern_group(dataset, {}, [get_strategy(sid) for sid in STRATEGIES],
                        fold, 'SYN', deadline=deadline, stop_file=path)
            self.assertEqual(caught.exception.code, code)

    def test_six_strategy_chunks_readback_and_no_replay_are_identical(self):
        import json
        folder = self.temporary()
        selected = (CHALLENGER, BULL_FLAG, *STRATEGIES)
        saved = []
        for label, implementation in (('baseline', baseline_prepare_many), ('candidate', pattern.prepare_many)):
            root = folder/label
            root.mkdir()
            dataset, spec = shared_fixture(root, count=4, bars=600)
            with patch.object(pattern, 'prepare_many', implementation):
                ref = prepared.build_prepared(dataset, spec, selected, root/'prepared',
                                             deadline=time.monotonic()+60, batch_size=2)
            manifest = json.loads(Path(ref['path']).read_bytes())
            saved.append({sid: [(root/'prepared'/pin['path']).read_bytes()
                                for pin in manifest['chunks'][sid]] for sid in selected})
            with self.assertRaises(FileExistsError):
                prepared.build_prepared(dataset, spec, selected, root/'prepared', deadline=time.monotonic()+60)
            for sid in selected:
                data, latest, dividends, observations = prepared.read_prepared(dataset,
                    dict(spec, prepared_market=ref), get_strategy(sid), spec['folds'][0])
                self.assertEqual(set(latest), set(spec['universe']['symbols']))
        self.assertEqual(saved[0], saved[1])

    def test_expired_builder_never_publishes_completed_manifest(self):
        folder = self.temporary()
        dataset, spec = shared_fixture(folder)
        with self.assertRaises(StrategyBudgetError):
            prepared.build_prepared(dataset, spec, STRATEGIES, folder/'prepared', deadline=time.monotonic()-1)
        self.assertFalse((folder/'prepared/manifest.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
