"""Lossless PostgreSQL JSON fractional-second compatibility, including Python3.10 boundary."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import importlib.util
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

source=Path(__file__).resolve().parents[1]/'scripts/trade_disposition.py'
spec=importlib.util.spec_from_file_location('_precision_policy_under_test',source)
policy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=policy;spec.loader.exec_module(policy)

class LegacyFractionBoundary(datetime):
    """Documented3/6-digit fractional boundary double, not a full Python3.10 runtime."""
    @classmethod
    def fromisoformat(cls,value):
        fraction=re.search(r'[T ][0-9]{2}:[0-9]{2}:[0-9]{2}\.([0-9]+)',value)
        if fraction and len(fraction[1]) not in (3,6):raise ValueError('legacy fractional precision')
        return datetime.fromisoformat(value)

class TimestampPrecisionTests(unittest.TestCase):
    def test_fractional_seconds_preserve_exact_instant_and_offset(self):
        for parser in (datetime,LegacyFractionBoundary):
            for fraction in ('1','12','123','1234','12345','123456'):
                for offset,minutes in (('+00:00',0),('+05:30',330),('-04:00',-240)):
                    for separator in ('T',' '):
                        with self.subTest(parser=parser.__name__,digits=len(fraction),offset=offset,separator=separator):
                            text='2026-10-22'+separator+'12:34:56.'+fraction+offset
                            expected=datetime(2026,10,22,12,34,56,int(fraction.ljust(6,'0')),tzinfo=timezone(timedelta(minutes=minutes)))
                            with mock.patch.object(policy,'datetime',parser):actual=policy.aware_time(text)
                            self.assertEqual(actual,expected);self.assertEqual(actual.utcoffset(),expected.utcoffset())
                            self.assertEqual(actual.microsecond,expected.microsecond)
    def test_zulu_and_existing_supported_forms(self):
        cases = (('2026-10-22T12:34:56Z', 0), ('2026-10-22T12:34:56.123Z', 123000),
                 ('2026-10-22T12:34:56.12345Z', 123450),
                 ('2026-10-22T12:34:56.123456+00:00', 123456), ('2026-10-22 12:34:56+00:00', 0))
        for text, microseconds in cases:
            for parser in (datetime,LegacyFractionBoundary):
                with self.subTest(text_kind='synthetic',parser=parser.__name__),mock.patch.object(policy,'datetime',parser):
                    expected = datetime(2026, 10, 22, 12, 34, 56, microseconds, tzinfo=timezone.utc)
                    self.assertEqual(policy.aware_time(text), expected)
    def test_null_nonstring_blank_and_whitespace_still_refuse_identity(self):
        for text in (None,True,1,{},[],'',' ',' 2026-10-22T12:34:56.12345+00:00','2026-10-22T12:34:56.12345+00:00 '):
            for parser in (datetime,LegacyFractionBoundary):
                with self.subTest(kind=type(text).__name__,parser=parser.__name__),mock.patch.object(policy,'datetime',parser):
                    with self.assertRaisesRegex(policy.TradePolicyError,'^TRADE_POLICY_IDENTITY_INVALID$'):policy.aware_time(text)
    def test_naive_malformed_invalid_calendar_and_sentinels_still_refuse(self):
        values=['2026-10-22T12:34:56.'+f for f in ('1','12','123','1234','12345','123456')]
        values+=['2026-10-22','2026-02-30T12:34:56.12345+00:00','2026-10-22T25:34:56.12345+00:00',
         '2026-10-22T12:34:56.12345+99:00','infinity','-infinity','10000-01-01T00:00:00.12345+00:00','not-a-time']
        for text in values:
            for parser in (datetime,LegacyFractionBoundary):
                with self.subTest(kind='invalid synthetic time',parser=parser.__name__),mock.patch.object(policy,'datetime',parser):
                    with self.assertRaisesRegex(policy.TradePolicyError,'^TRADE_POLICY_TIME_INVALID$'):policy.aware_time(text)
    def test_normalization_never_changes_rows_or_fingerprints(self):
        row=dict(trade_id=9,status='CLOSED',qty=1,entry_price=10,exit_price=11,realized_pnl=1,
          entry_time='2026-10-22T12:34:56.12345+00:00',exit_time='2026-10-22T13:34:56.12345+00:00')
        original=deepcopy(row);fingerprint=policy.fingerprint(row)
        with mock.patch.object(policy,'datetime',LegacyFractionBoundary):policy.require_complete_accounting([row])
        self.assertEqual(row,original);self.assertEqual(policy.fingerprint(row),fingerprint)
        self.assertEqual(row['entry_time'],original['entry_time'])
    def test_accounting_holds_and_missing_pnl_remain(self):
        valid=dict(status='CLOSED',qty=1,entry_price=10,exit_price=11,realized_pnl=1,
         entry_time='2026-10-22T12:34:56.12345+00:00',exit_time='2026-10-22T13:34:56.12345+00:00')
        for key,value in (('realized_pnl',None),('realized_pnl',float('nan')),('qty',0),('entry_price',0),('exit_price',-1)):
            with self.subTest(field=key),mock.patch.object(policy,'datetime',LegacyFractionBoundary):
                row=dict(valid);row[key]=value
                with self.assertRaisesRegex(policy.TradePolicyError,'^TRADE_POLICY_ACCOUNTING_INCOMPLETE$'):policy.require_complete_accounting([row])
        row=dict(valid);del row['realized_pnl']
        with self.assertRaisesRegex(policy.TradePolicyError,'^TRADE_POLICY_ACCOUNTING_INCOMPLETE$'):policy.require_complete_accounting([row])
    def test_existing_native_datetime_accounting_path(self):
        stamp=datetime(2026,10,22,12,tzinfo=timezone.utc)
        row=dict(status='CLOSED',qty=1,entry_price=10,exit_price=11,realized_pnl=1,entry_time=stamp,exit_time=stamp)
        policy.require_complete_accounting([row]);self.assertIs(row['entry_time'],stamp)
    def test_outside_target_shape_is_forwarded_without_precision_rewrite(self):
        calls=[]
        class RecordingParser(datetime):
            @classmethod
            def fromisoformat(cls,value):calls.append(value);return datetime.fromisoformat(value)
        values=('2026-10-22T12:34:56.1234567+00:00','2026-10-22T12:34:56.12345',
         '2026-10-22X12:34:56.12345+00:00','2026-10-22T12:34:56.12345+00:00:01')
        for text in values:
            with self.subTest(shape='outside targeted normalizer'),mock.patch.object(policy,'datetime',RecordingParser):
                try:policy.aware_time(text)
                except policy.TradePolicyError:pass
                self.assertEqual(calls[-1],text)

if __name__=='__main__':unittest.main()
