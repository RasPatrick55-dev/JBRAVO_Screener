"""Offline provisional-mode contracts; no real Pipeline, provider or database."""
import contextlib
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from typing import Tuple, Optional
from zoneinfo import ZoneInfo
from unittest import mock
from scripts import pipeline_postflight as P, primary_pipeline as Primary, signal_handoff as S
from scripts import gross_paper_accounting as G, trade_disposition as D
import test_paper_gross_recovery as R
import test_pipeline_postflight as PF
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def extracted(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 'actual-provisional-source', 'exec'), namespace)
    return namespace.get(getattr(node, 'name', ''))


def context(day='2026-10-23', started=None):
    started = started or datetime.now(timezone.utc) - timedelta(seconds=2)
    return dict(mode='MANUAL_PROVISIONAL_PIPELINE_V1', id='1' * 32,
        started_at=started.isoformat(), expires_at=(started + timedelta(minutes=22)).isoformat(),
        signal_session=day, qualified_daily=False, entry_ready=False)


@contextlib.contextmanager
def mode(value):
    with mock.patch.dict(os.environ, {P.PROVISIONAL_ENV: json.dumps(value)}):
        yield


class ProvisionalContracts(unittest.TestCase):
    def test_source_label_preserves_original_and_refuses_cross_run_reuse(self):
        current = context()
        label = P.mark_provisional_source('iex', current)
        self.assertEqual(label, 'PROVISIONAL:' + current['id'] + ':iex')
        self.assertEqual(P.mark_provisional_source(label, current), label)
        with self.assertRaisesRegex(ValueError, 'another_run'):
            P.mark_provisional_source(label, current | {'id': '2' * 32})
    def test_missing_context_preserves_normal_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(P.provisional_context())
            self.assertTrue(P.metadata_context_matches({}))

    def test_context_rejects_bad_type_future_expiry_and_promotion(self):
        c = context()
        self.assertEqual(P.validate_provisional_context(c), c)
        for change in ({'entry_ready': True}, {'qualified_daily': True}, {'id': '../bad'},
                       {'expires_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
                       {'signal_session': 'today'}, {'extra': True}):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                P.validate_provisional_context(c | change)
        with self.assertRaises(ValueError):
            P.validate_provisional_context(c, now=datetime.now(timezone.utc) + timedelta(hours=1))

    def test_manual_session_identifies_day_but_never_becomes_entry_contract(self):
        calendar = [{'date': '2026-10-23', 'open': '09:30:00', 'close': '16:00:00'},
                    {'date': '2026-10-26', 'open': '09:30:00', 'close': '16:00:00'}]
        now = datetime(2026, 10, 23, 21, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, 'signal_session_not_completed'):
            S.session_contract(calendar, now=now)
        provisional = S.session_contract(calendar, now=now, provisional=True)
        self.assertEqual(provisional['signal_session'], '2026-10-23')
        self.assertFalse(provisional['qualified_daily'])
        with self.assertRaisesRegex(ValueError, 'provisional_session_not_qualified'):
            S.validate_session(provisional, started=now, finished=now, now=now, for_entry=True)
        ready = S.session_contract(calendar, now=datetime(2026, 10, 24, 0, 15, tzinfo=timezone.utc))
        self.assertNotIn('provisional', ready)

    def test_metadata_cannot_cross_manual_runs_or_be_reused_normally(self):
        c = context()
        with mode(c):
            self.assertFalse(P.metadata_context_matches({}))
            self.assertTrue(P.metadata_context_matches({'provisional_context': c}))
            self.assertFalse(P.metadata_context_matches({'provisional_context': c | {'id': '2' * 32}}))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(P.metadata_context_matches({'provisional_context': c}))

    def test_gross_publication_is_labeled_and_default_consumer_rejects(self):
        fixture = R.RecoveryTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        c = context(fixture.day)
        with mode(c):
            proof = fixture.publish()
            self.assertEqual(proof['provisional_context'], c)
            self.assertEqual(fixture.state.top[0]['source'], 'PROVISIONAL:' + c['id'] + ':synthetic-backtest')
            self.assertEqual(fixture.state.commits, 1)
            self.assertEqual(fixture.state.trades, fixture.original)
            with self.assertRaises(D.TradePolicyError):
                with fixture.evidence():
                    pass
            with fixture.evidence(allow_provisional=True) as observed:
                self.assertEqual(observed['provisional_context'], c)
                self.assertFalse(observed['entry_ready'])
                self.assertFalse(observed['net_eligible'])
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(D.TradePolicyError):
            with fixture.evidence():
                pass

    def test_postflight_separates_functional_pass_from_qualified_daily(self):
        fixture = PF.PostflightTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        c = context(fixture.started.date().isoformat(), fixture.started - timedelta(seconds=1))
        P._save_control(fixture.root, 'active-primary.json', {'id': c['id'], 'started_at': c['started_at']})
        with mode(c):
            record = fixture.publish(invocation_id=c['id'])
        ordinary = fixture.check()
        self.assertEqual(ordinary['status'], 'failed')
        self.assertIn('provisional_session_not_qualified', ordinary['failures'])
        functional = fixture.check(allow_provisional=True)
        self.assertEqual(functional['status'], 'ok')
        self.assertTrue(functional['functional_pass'])
        self.assertFalse(functional['pipeline_ready'])
        self.assertFalse(functional['qualified_daily'])
        self.assertFalse(functional['entry_ready'])
        with self.assertRaises(ValueError):
            P.complete_primary(fixture.root, {'id': c['id']}, functional | {'session': {'pretend': True}})
        self.assertFalse((fixture.root / 'reports/pipeline_postflight/completed-primary.json').exists())
        self.assertEqual(record['provisional_context'], c)

    def test_stale_source_or_failed_stage_is_not_functional_success(self):
        fixture = PF.PostflightTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        c = context(fixture.started.date().isoformat(), fixture.started - timedelta(seconds=1))
        P._save_control(fixture.root, 'active-primary.json', {'id': c['id'], 'started_at': c['started_at']})
        with mode(c):
            fixture.publish(invocation_id=c['id'], pipeline_rc=1,
                            step_rcs={**fixture.facts['step_rcs'], 'metrics': 2})
        result = fixture.check(allow_provisional=True)
        self.assertFalse(result['functional_pass'])
        self.assertEqual(result['status'], 'failed')


class ProvisionalController(unittest.TestCase):
    def test_explicit_paper_run_keeps_all_five_stages_and_restores_environment(self):
        seen = []
        def pipeline(arguments):
            seen.extend(arguments)
            current = P.provisional_context()
            current['signal_session'] = '2026-10-23'
            os.environ[P.PROVISIONAL_ENV] = json.dumps(current)
            return 0
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ,
                {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets', 'JBR_EXEC_PAPER': 'true'}, clear=True):
            with (mock.patch.object(P, 'check_health', return_value={'status': 'ok', 'functional_pass': True}) as check,
                  mock.patch.object(P, 'complete_primary') as complete, mock.patch('sys.stdout', new=io.StringIO())):
                rc = Primary.main(['--manual-provisional-test'], pipeline_main=pipeline, base_dir=folder)
            self.assertEqual(rc, 0)
            self.assertNotIn('--session-handoff', seen)
            self.assertEqual(seen[seen.index('--steps') + 1], 'screener,backtest,metrics,labels,ranker_eval')
            self.assertIn('--metrics-args=--paper-gross-pipeline', seen)
            self.assertTrue(check.call_args.kwargs['allow_provisional'])
            complete.assert_not_called()
            self.assertNotIn(P.PROVISIONAL_ENV, os.environ)
            self.assertNotIn('JBRAVO_SESSION_HANDOFF', os.environ)

    def test_conflicting_context_custom_arguments_or_live_endpoint_refuse_before_stage(self):
        for args, endpoint in ((['--manual-provisional-test', '--steps', 'metrics'], 'https://paper-api.alpaca.markets'),
                               (['--manual-provisional-test'], 'https://api.alpaca.markets')):
            with self.subTest(args=args, endpoint=endpoint), mock.patch.dict(os.environ,
                    {'APCA_API_BASE_URL': endpoint, 'JBR_EXEC_PAPER': 'true'}, clear=True):
                fake = mock.Mock()
                with mock.patch('sys.stdout', new=io.StringIO()):
                    self.assertEqual(Primary.main(args, pipeline_main=fake), 1)
                fake.assert_not_called()

    def test_ambient_context_does_not_enable_normal_scheduled_mode(self):
        with mode(context()), mock.patch('sys.stdout', new=io.StringIO()):
            fake = mock.Mock()
            self.assertEqual(Primary.main([], pipeline_main=fake), 1)
            fake.assert_not_called()


class ProvisionalDataflow(unittest.TestCase):
    def test_review_legacy_freshness_fixture_runs_actual_three_cases(self):
        from typing import Any
        test_tree = ast.parse((SOURCE_ROOT / 'tests/test_feature_inference_session.py').read_bytes())
        node = next(n for n in ast.walk(test_tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'test_pipeline_refreshes_legacy_or_wrong_session_features')
        namespace = dict(ast=ast, ROOT=SOURCE_ROOT, date=datetime.date,
                         Mock=mock.Mock, Any=Any, metadata_context_matches=P.metadata_context_matches)
        from datetime import date
        namespace['date'] = date
        actual = extracted(node, namespace)
        with mock.patch.dict(os.environ, {}, clear=True):
            actual(self)

    def test_review_screener_predictions_remain_private_even_if_later_work_fails(self):
        from typing import Mapping
        tree = ast.parse((SOURCE_ROOT / 'scripts/screener.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_predictions')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ordinary = root / 'data/predictions'; ordinary.mkdir(parents=True)
            (ordinary/'latest.csv').write_text('ordinary-latest')
            dated = ordinary / (datetime.now(timezone.utc).date().isoformat()+'.csv')
            dated.write_text('ordinary-dated')
            frame = pd.DataFrame({'symbol':['ONE'], 'score':[1.]})
            ns = dict(pd=pd, Optional=Optional, Mapping=Mapping, datetime=datetime, timezone=timezone,
                      Path=lambda path: root/path, LOGGER=mock.Mock(),
                      _prepare_predictions_frame=lambda *args,**kw: frame.copy())
            writer = extracted(node, ns)
            c = context()
            with mode(c):
                writer(frame, {})
            self.assertEqual((ordinary/'latest.csv').read_text(), 'ordinary-latest')
            self.assertEqual(dated.read_text(), 'ordinary-dated')
            private = ordinary/'manual_provisional'/c['id']/'latest.csv'
            self.assertTrue(private.is_file())
            self.assertEqual(pd.read_csv(private)['symbol'].tolist(), ['ONE'])
            c['expires_at'] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
            with mode(c), self.assertRaises(ValueError):
                writer(frame,{})
            self.assertEqual((ordinary/'latest.csv').read_text(), 'ordinary-latest')
            with mock.patch.dict(os.environ, {}, clear=True):
                writer(frame,{})
            self.assertEqual(pd.read_csv(ordinary/'latest.csv')['symbol'].tolist(), ['ONE'])

    def test_review_model_recipe_bootstrap_does_not_borrow_feature_rows(self):
        from typing import Any
        tree = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_bytes())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_latest_model_meta')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); models = root/'data/models'; models.mkdir(parents=True)
            path = models/'ranker_2026-10-20.pkl';path.write_bytes(b'inert-not-loaded')
            recipe = dict(model_path=str(path),feature_set='v1',feature_columns=['x'],feature_signature='valid',
                          feature_signature_stored='valid')
            select = mock.Mock(return_value=(path,{'selection_mode':'explicit'}))
            ns = dict(Any=Any, base_dir=root, provisional=context(),
                      _load_features_meta=lambda:({},'missing'), metadata_context_matches=P.metadata_context_matches,
                      load_model_artifact_meta=lambda path:dict(recipe), select_compatible_model=select,
                      compute_feature_signature=lambda columns:'valid')
            latest = extracted(node,ns)
            result=latest()
            self.assertEqual(result['selection_reason'],'provisional_model_recipe_bootstrap')
            self.assertEqual(select.call_args.args[1],{key:recipe[key] for key in
                ('feature_set','feature_columns','feature_signature')})
            self.assertEqual(select.call_args.kwargs,{'requested_model_path':path})
            # The real freshness predicate still requires freshly generated current-run metadata.
            fresh_node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_ensure_features_freshness')
            ns.update(_latest_model_meta=latest,pipeline_run_date=datetime(2026,10,23).date(),LOG=mock.Mock())
            check=extracted(fresh_node,ns)
            self.assertEqual(check('ranker_eval')['reason'],'features_meta_missing')
            with mode(context()):
                meta={**recipe, 'signal_session':'2026-10-23','provisional_context':P.provisional_context()}
                ns['_load_features_meta']=lambda:(meta,'private')
                self.assertFalse(check('ranker_eval')['stale'])
                meta['feature_signature']='wrong'
                meta['feature_columns']=['different']
                ns['compute_feature_signature']=lambda columns:'different' if columns==['different'] else 'valid'
                self.assertTrue(check('ranker_eval')['stale'])
            # Malformed/contradictory model recipes cannot be used to bootstrap.
            ns['_load_features_meta']=lambda:({},'missing')
            for change in ({'feature_set':'unknown'},{'feature_columns':[]},{'feature_columns':['x','x']},
                           {'feature_columns':[['not-a-column']]},
                           {'feature_signature_stored':'wrong'},{'feature_signature':'wrong'}):
                with self.subTest(change=change):
                    ns['load_model_artifact_meta']=lambda path,value=recipe|change:dict(value)
                    ns['select_compatible_model']=mock.Mock(return_value=(None,{'reason':'missing','selection_mode':'normal'}))
                    self.assertIsNone(latest()['model_path'])
                    self.assertNotIn('requested_model_path',ns['select_compatible_model'].call_args.kwargs)

    def test_review_postflight_midnight_uses_bound_invocation_and_signal_session(self):
        fixture = PF.PostflightTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        started=datetime(2026,10,23,23,59,tzinfo=timezone.utc)
        finished=started+timedelta(minutes=2)
        c=context('2026-10-23',started-timedelta(seconds=1))
        # Build a normal authentic record, then assess specific immutable window variants.
        record=fixture.publish()
        record.update(started_at=started.isoformat(),finished_at=finished.isoformat(),
                      provisional_context=c,invocation_id=c['id'])
        record['coverage']['run_ts_utc']=(started+timedelta(seconds=1)).isoformat()
        record['predictions']['snapshot_date']='2026-10-23'
        now=finished+timedelta(seconds=1)
        for day in (started.date(),finished.date()):
            with self.subTest(day=day):
                result=P.assess(record,now=now,expected_day=day,allow_provisional=True)
                self.assertTrue(result['functional_pass'])
                self.assertFalse(result['qualified_daily'])
        ordinary=deepcopy(record);ordinary.pop('provisional_context');ordinary.pop('invocation_id')
        self.assertIn('wrong_run_day',P.assess(ordinary,now=now,expected_day=finished.date())['failures'])
        wrong=deepcopy(record);wrong['predictions']['snapshot_date']='2026-10-22'
        self.assertIn('provisional_prediction_session_mismatch',P.assess(wrong,now=now,
                      expected_day=finished.date(),allow_provisional=True)['failures'])
        for change in ({'started_at':(started+timedelta(seconds=1)).isoformat()},
                       {'expires_at':(finished-timedelta(seconds=1)).isoformat()},
                       {'signal_session':None},{'signal_session':'2026-10-24'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                bad=deepcopy(record);bad['provisional_context'].update(change)
                P.assess(bad,now=now,expected_day=finished.date(),allow_provisional=True)

    def test_final_metrics_rebuild_actual_canonical_writer_and_dashboard_keep_label(self):
        from typing import Any, Mapping, Dict
        # Pinned f2 helper is evidence, not another published/modified source file.
        helper = ast.parse((SOURCE_ROOT / 'utils/screener_metrics.py').read_bytes())
        utility = dict(Any=Any, Mapping=Mapping, Path=Path, datetime=datetime, timezone=timezone,
            os=os, json=json, DEFAULT_REQUIRED_BARS=250,
            atomic_write_bytes=lambda path, data: path.write_bytes(data))
        for node in helper.body:
            if isinstance(node, ast.FunctionDef): extracted(node, utility)
        pipeline = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_bytes())
        wrapper_node = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'write_screener_metrics_json')
        write = extracted(wrapper_node, {'_write_screener_metrics_json': utility['write_screener_metrics_json']})
        dashboard = ast.parse((SOURCE_ROOT / 'dashboards/data_io.py').read_bytes())
        nodes = {n.name:n for n in dashboard.body if isinstance(n, ast.FunctionDef)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'screener_metrics.json'
            reader=dict(Any=Any, Mapping=Mapping, Dict=Dict, Optional=Optional, Path=Path, DATA_DIR=root,
                _read_json_safe=lambda value: json.loads(value.read_text()),
                _coerce_int=lambda value: int(value) if value is not None else None,
                _count_csv_rows=lambda value: 0, _parse_pipeline_log=lambda value: {},
                load_connection_health=lambda value: {}, _mtime_iso=lambda value: None,
                _normalize_feed=lambda value: value, _freshness=lambda value: {},
                _run_type_hint=lambda: 'nightly', _update_coverage_history=lambda value: {}, os=os)
            load=extracted(nodes['load_screener_metrics'],reader)
            health=extracted(nodes['screener_health'],reader)
            c=context()
            with mode(c):
                # Final reconstruction has no incoming label; wrapper must add it.
                write(path, {'rows':2,'pipeline_rc':0,'latest_source':'db'})
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load(root)['provisional_context'],c)
                snapshot=health(root)
                self.assertEqual(snapshot['provisional_context'],c)
                self.assertFalse(snapshot['qualified_daily'])
                self.assertFalse(snapshot['entry_ready'])
                self.assertIn('PROVISIONAL',snapshot['run_type'])
                self.assertEqual(snapshot['pipeline_rc'],0)
                write(path, {'rows':2,'pipeline_rc':0,'latest_source':'db'})
                self.assertNotIn('provisional_context',load(root))
                self.assertEqual(health(root)['run_type'],'nightly')

    def test_actual_fresh_stamp_and_summary_label_functional_zero_exit(self):
        tree=ast.parse((SOURCE_ROOT/'scripts/run_pipeline.py').read_bytes())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=next(n for n in ast.walk(main) if isinstance(n,ast.Try)
                   and any(isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fresh_payload'
                           for t in s.targets) for s in n.body))
        start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='fresh_payload' for t in n.targets))
        code=ast.fix_missing_locations(ast.Module(body=block.body[start:start+3],type_ignores=[]))
        namespace=dict(now_ny=datetime.now(timezone.utc),datetime=datetime,timezone=timezone,rows_for_stamp=2)
        c=context()
        with mode(c):exec(compile(code,'actual-fresh-stamp','exec'),namespace)
        value=namespace['fresh_payload']
        self.assertEqual(value['rc'],0)
        self.assertEqual(value['provisional_context'],c)
        self.assertFalse(value['qualified_daily'])
        self.assertIn('PROVISIONAL',value['run_type'])
        with mock.patch.dict(os.environ,{},clear=True):
            exec(compile(code,'actual-fresh-stamp','exec'),namespace)
        self.assertNotIn('provisional_context',namespace['fresh_payload'])
        summary_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                          and n.name=='safe_write_pipeline_summary')
        with tempfile.TemporaryDirectory() as directory:
            summary=extracted(summary_node,dict(Path=Path,PROJECT_ROOT=Path(directory),datetime=datetime,
                timezone=timezone,json=json,LOG=mock.Mock(),atomic_write_bytes=lambda p,data:p.write_bytes(data)))
            with mode(c):path=summary(0,base_dir=Path(directory))
            result=json.loads(path.read_text())
            self.assertEqual(result['rc'],0)
            self.assertEqual(result['provisional_context'],c)
            self.assertFalse(result['qualified_daily'])

    def test_actual_status_writer_labels_functional_status_without_promoting_it(self):
        from typing import Any, Mapping
        tree = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_status_json')
        writer = mock.Mock()
        function = extracted(node, {'Path': Path, 'Mapping': Mapping, 'Any': Any,
            'json': json, 'atomic_write_bytes': writer, 'LOG': mock.Mock()})
        with tempfile.TemporaryDirectory() as directory:
            c = context()
            with mode(c):
                function(Path(directory), {'status': 'ok'})
            value = json.loads(writer.call_args.args[1])
            self.assertEqual(value['provisional_context'], c)
            self.assertFalse(value['qualified_daily'])
            self.assertFalse(value['entry_ready'])
            with mock.patch.dict(os.environ, {}, clear=True):
                function(Path(directory), {'status': 'ok'})
            self.assertEqual(json.loads(writer.call_args.args[1]), {'status': 'ok'})

    def test_run_private_artifacts_never_replace_or_read_ordinary_outputs(self):
        c = context()
        outputs = ('daily_bars', 'labels', 'features', 'predictions', 'predictions_scoped', 'ranker_eval')
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(P.provisional_artifact_type('features'), 'features')
        with mode(c):
            for name in outputs:
                with self.subTest(name=name):
                    self.assertEqual(P.provisional_artifact_type(name), 'provisional:' + c['id'] + ':' + name)
                    self.assertEqual(P.provisional_artifact_type(name, writing=True), P.provisional_artifact_type(name))
            self.assertEqual(P.provisional_artifact_type('ranker_monitor'), 'ranker_monitor')
            self.assertNotEqual(P.provisional_artifact_type('ranker_monitor', writing=True), 'ranker_monitor')
        with mode(c | {'id': '2' * 32}):
            self.assertNotEqual(P.provisional_artifact_type('features'), 'provisional:' + c['id'] + ':features')

    def test_actual_db_key_and_payload_scope_are_atomic_and_normal_defaults_unchanged(self):
        from typing import Any, Mapping
        tree = ast.parse((SOURCE_ROOT / 'scripts/db.py').read_bytes())
        nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        storage = {}; calls = []
        class Connection:
            def __init__(self): self.row = None
            def cursor(self): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, parameters):
                calls.append((sql, deepcopy(parameters)))
                key = parameters['artifact_type']
                if 'INSERT INTO ml_artifacts' in sql:
                    storage[key] = deepcopy(parameters)
                else:
                    value = storage.get(key)
                    self.row = (value['run_date'], None, value['rows_count'], value['payload'],
                                value['csv_data'], value['source'], value['file_name']) if value else None
            def fetchone(self): return self.row
            def commit(self): pass
            def close(self): pass
        namespace = {'Any': Any, 'Mapping': Mapping, 'db_enabled': lambda: True,
            'ensure_ml_artifacts_table': mock.Mock(), '_conn_or_none': Connection,
            '_coerce_date': lambda value: value, '_policy_write_lock': lambda connection: None,
            'extras': SimpleNamespace(Json=lambda value: value), 'logger': mock.Mock()}
        for name in ('upsert_ml_artifact', 'fetch_latest_ml_artifact', 'fetch_ml_artifact'):
            extracted(nodes[name], namespace)
        put, get = namespace['upsert_ml_artifact'], namespace['fetch_ml_artifact']
        c = context()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(put('daily_bars', c['signal_session'], csv_data='normal', payload={}))
        with mode(c):
            self.assertIsNone(get('daily_bars'))
            self.assertTrue(put('daily_bars', c['signal_session'], csv_data='partial', payload={'source_fact': 7}))
            self.assertEqual(get('daily_bars')['csv_data'], 'partial')
            self.assertEqual(get('daily_bars', c['signal_session'])['payload']['provisional_context'], c)
            with self.assertRaisesRegex(ValueError, 'context_mismatch'):
                put('daily_bars', c['signal_session'], payload={'provisional_context': c | {'id': '2' * 32}})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get('daily_bars')['csv_data'], 'normal')
        with mode(c | {'id': '2' * 32}):
            self.assertIsNone(get('daily_bars'))
        self.assertEqual(storage['daily_bars']['payload'], {})

    def test_isolated_file_directory_and_expiry_preserve_ordinary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old_csv = base / 'features.csv';old_csv.write_text('ordinary')
            old_meta = base / 'latest_meta.json';old_meta.write_text('{"ordinary": true}')
            c = context()
            with mode(c):
                private = P.provisional_output_directory(base)
                private.mkdir(parents=True)
                (private / 'features.csv').write_text('partial write then failure')
                self.assertEqual(private, base / 'manual_provisional' / c['id'])
            self.assertEqual(old_csv.read_text(), 'ordinary')
            self.assertEqual(json.loads(old_meta.read_text()), {'ordinary': True})
            expired = context(started=datetime.now(timezone.utc) - timedelta(hours=1))
            with mode(expired), self.assertRaisesRegex(ValueError, 'expired'):
                P.provisional_output_directory(base)
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(P.provisional_output_directory(base), base)

    def test_actual_feature_writer_fails_closed_without_touching_normal_sidecars(self):
        import logging
        tree = ast.parse((SOURCE_ROOT / 'scripts/feature_generator.py').read_bytes())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        body = next(n for n in main.body if isinstance(n, ast.Try)).body
        start = next(i for i,n in enumerate(body) if isinstance(n, ast.ImportFrom)
                     and n.module == 'scripts.pipeline_postflight')
        stop = next(i for i,n in enumerate(body[start:], start) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'db.db_enabled()')
        code = compile(ast.fix_missing_locations(ast.Module(body=body[start:stop], type_ignores=[])),
                       'actual-feature-publication', 'exec')
        for failed_write in (1, 2):
            with self.subTest(failed_write=failed_write), tempfile.TemporaryDirectory() as directory:
                root = Path(directory);ordinary = root / 'latest_meta.json';ordinary.write_text('ordinary')
                writer = mock.Mock(side_effect=[False] if failed_write == 1 else [True, False])
                namespace = dict(Path=Path, args=SimpleNamespace(output_dir=root), as_of='2026-10-23',
                    merged=pd.DataFrame({'label_5d_pos_300bp': [None]}), logging=logging,
                    infer_feature_columns_for_ml=lambda value: ['x'], compute_feature_signature=lambda value: 'hash',
                    run_date=datetime(2026,10,23).date(), feature_set='v1', feature_meta={},
                    DEFAULT_BARS_ADJUSTMENT='raw', DEFAULT_SPLIT_ADJUST='off', datetime=datetime,
                    timezone=timezone, per_file_meta_path=lambda path: path.with_suffix('.meta.json'),
                    FEATURE_META_NAME='latest_meta.json', _write_feature_meta=writer)
                with mode(context()), self.assertRaisesRegex(ValueError, 'metadata_write_failed'):
                    exec(code, namespace)
                self.assertEqual(ordinary.read_text(), 'ordinary')
                self.assertTrue(namespace['output_path'].is_relative_to(root / 'manual_provisional'))

    def test_actual_calendar_window_caps_provisional_query_and_leaves_normal_recipe(self):
        tree = ast.parse((SOURCE_ROOT / 'scripts/utils/calendar.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calc_daily_window')
        from datetime import date
        client = SimpleNamespace(get_calendar=lambda request: [SimpleNamespace(date=date(2026, 10, 22), close=True),
                                                               SimpleNamespace(date=date(2026, 10, 23), close=True)])
        namespace = {'TradingClient': object, 'date': date, 'datetime': datetime, 'timedelta': timedelta,
            'timezone': timezone, 'ZoneInfo': ZoneInfo, 'GetCalendarRequest': lambda **kwargs: kwargs}
        window = extracted(node, namespace)
        current = context(started=datetime(2026, 10, 23, 21, tzinfo=timezone.utc))
        # Calendar semantics use a fixed observation, independent of today's clock.
        with mock.patch.object(P, 'provisional_context', return_value=current):
            _, end, day = window(client, 10, end_date=date(2026, 10, 23), session_cutoff=True)
        self.assertEqual(day, '2026-10-23')
        self.assertEqual(end, current['started_at'])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(window(client, 10, end_date=date(2026, 10, 23), session_cutoff=True)[1],
                             '2026-10-24T00:00:00+00:00')
    def test_dashboard_source_is_explicitly_provisional_and_normal_source_unchanged(self):
        tree = ast.parse((SOURCE_ROOT / 'dashboards/data_io.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'screener_table')
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({'symbol': ['ONE'], 'score': [1.0], 'source': ['PROVISIONAL:' + '1' * 32]})
            namespace = {'pd': pd, 'Tuple': Tuple, 'Optional': Optional,
                'db': SimpleNamespace(db_enabled=lambda: False), 'DATA_DIR': Path(directory),
                '_read_csv_safe': lambda path: frame, '_mtime_iso': lambda path: 'timestamp',
                'logger': mock.Mock()}
            table = extracted(node, namespace)
            data, _, label = table()
            self.assertIn('PROVISIONAL', label)
            self.assertFalse(data['qualified_daily'].iloc[0])
            frame['source'] = 'ordinary'
            data, _, label = table()
            self.assertNotIn('PROVISIONAL', label)
            self.assertNotIn('qualified_daily', data)

    def test_cache_writes_are_suppressed_only_for_provisional_mode(self):
        tree = ast.parse((SOURCE_ROOT / 'scripts/screener.py').read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fetch_daily_bars')
        writer = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == '_write_batch_cache')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.parquet'
            frame = SimpleNamespace(empty=False, to_parquet=mock.Mock())
            namespace = {'pd': pd, 'Path': Path, 'json': json, 'provisional': context(),
                         '_cache_paths': lambda index: (path, path.with_suffix('.json')), 'LOGGER': mock.Mock()}
            save = extracted(writer, namespace)
            save(1, frame, {'rows': 1})
            frame.to_parquet.assert_not_called()
            self.assertFalse(path.with_suffix('.json').exists())
            namespace['provisional'] = None
            save(1, frame, {'rows': 1})
            frame.to_parquet.assert_called_once()
            self.assertEqual(json.loads(path.with_suffix('.json').read_text()), {'rows': 1})

    def test_actual_alert_wrapper_labels_provisional_without_changing_normal_calls(self):
        tree = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pipeline_alert')
        send = mock.Mock()
        alert = extracted(node, {'send_alert': send})
        with mock.patch.dict(os.environ, {}, clear=True):
            alert('title', {'x': 1})
        send.assert_called_once_with('title', {'x': 1})
        send.reset_mock()
        with mode(context()):
            alert('title', {'x': 1})
        self.assertTrue(send.call_args.args[0].startswith('PROVISIONAL functional test:'))
        self.assertFalse(send.call_args.args[1]['qualified_daily'])

    def test_capture_target_isolated_and_existing_capture_function_still_exercised(self):
        from scripts import production_score_capture as capture
        tree = ast.parse((SOURCE_ROOT / 'scripts/ranker_predict.py').read_bytes())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'capture_root' for t in n.targets))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current = context()
            namespace = {'BASE_DIR': base, 'provisional': current}
            extracted(assignment, namespace)
            target = namespace['capture_root']
            self.assertEqual(target, base / 'data' / 'manual_provisional' / current['id'])
            stamp = pd.Timestamp('2026-10-23T04:00:00Z')
            features = pd.DataFrame({'symbol': ['ONE'], 'timestamp': [stamp], 'feature': [1.0]})
            predictions = pd.DataFrame({'symbol': ['ONE'], 'timestamp': [stamp], 'score_5d': [0.6]})
            proof = capture.capture(target, features, predictions, snapshot_date='2026-10-23',
                model_binding={'feature_columns': ['feature']})
            self.assertTrue(Path(proof['manifest_path']).is_relative_to(target))
            self.assertFalse((base / 'data' / 'production_score_history').exists())
            self.assertFalse(proof['research_qualified'])
            namespace['provisional'] = None
            extracted(assignment, namespace)
            self.assertEqual(namespace['capture_root'], base)
