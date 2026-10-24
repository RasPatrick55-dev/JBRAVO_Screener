"""Bounded PAPER activity-response/ledger checkpoints; no clients or runtime DDL.

An interval proof does not certify account history, fees/net, recovery or entries.
The ingestor owns one atomic ledger/checkpoint/cursor transaction. Reconciliation
must independently read the committed row and verify the ledger manifestation.
"""
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from uuid import UUID
from pathlib import Path
import hashlib,json,re

ENDPOINT='https://paper-api.alpaca.markets'
SCOPE='ACTIVE_RECONCILIATION_CREATED_AT_INTERVAL_V1'
ENV_WORKFLOW='JBR_ACCOUNT_SYNC_INVOCATION_ID'
ENV_CUTOFF='JBR_ACCOUNT_SYNC_CUTOFF_UTC'
ACTIVE_LOOKBACK_DAYS=14
MAX_ROWS=200000
FIELDS=('activity_id','activity_type','transaction_time','symbol','side','qty','price','amount','order_id','description','raw')

def require(value,code):
    if not value:raise RuntimeError('ACT_CHECKPOINT_'+code)

def utc(value):
    if isinstance(value,str):
        value=value.replace('Z','+00:00')
        # Preserve the existing Python3.10 precision repair for short fractions.
        value=re.sub(r'\.(\d{1,5})(?=[+-]\d{2}:\d{2}$)',lambda m:'.'+m[1].ljust(6,'0'),value)
        value=datetime.fromisoformat(value)
    require(isinstance(value,datetime) and value.utcoffset() is not None,'TIME_INVALID')
    return value.astimezone(timezone.utc)

def workflow(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{32}',value),'WORKFLOW_INVALID')
    return str(UUID(hex=value))

def bound_cutoff(workflow_id,value,now):
    workflow(workflow_id)
    try:cutoff=utc(value)
    except (RuntimeError,ValueError,TypeError):raise RuntimeError('ACT_CHECKPOINT_CUTOFF_INVALID') from None
    require(cutoff<=utc(now),'CUTOFF_FUTURE')
    return cutoff

def source_fingerprint(endpoint,account_id):
    require(endpoint.rstrip('/')==ENDPOINT,'PAPER_SOURCE_REQUIRED')
    try:account=str(UUID(str(account_id)))
    except (ValueError,TypeError,AttributeError):raise RuntimeError('ACT_CHECKPOINT_ACCOUNT_INVALID') from None
    return hashlib.sha256((ENDPOINT+'\n'+account).encode()).hexdigest()

def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()

def recipe_fingerprint():
    root=Path(__file__).resolve().parent
    return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('activity_ingestion_checkpoint.py','fetch_account_activities.py')})

def row_value(row):
    require(set(row)==set(FIELDS),'ROW_FIELDS_INVALID')
    out={}
    for k in FIELDS:
        v=row[k]
        if k=='transaction_time':v=utc(v).isoformat()
        elif k in ('qty','price','amount') and v is not None:
            v=Decimal(str(v));require(v.is_finite(),'NUMBER_INVALID');v=format(v,'f')
            if '.' in v:v=v.rstrip('0').rstrip('.')
            if Decimal(v)==0:v='0'
        elif k=='raw':
            if isinstance(v,str):v=json.loads(v)
            require(isinstance(v,dict),'RAW_INVALID')
        elif v is not None:
            if isinstance(v,UUID):v=str(v)
            require(isinstance(v,str),'TEXT_INVALID')
        out[k]=v
    require(out['activity_id'] and out['activity_type'],'IDENTITY_INVALID')
    return out

def manifestation(rows):
    require(len(rows)<=MAX_ROWS,'ROW_LIMIT')
    values={}
    for row in rows:
        v=row_value(row);key=(v['activity_id'],v['activity_type'])
        require(key not in values,'DUPLICATE_IDENTITY');values[key]=v
    return [{'activity_id':k[0],'activity_type':k[1],'sha256':digest(v)} for k,v in sorted(values.items())]

def ledger_rows(cursor,keys):
    result=[]
    for offset in range(0,len(keys),200):
        part=keys[offset:offset+200]
        placeholders=','.join('(%s,%s)' for _ in part)
        cursor.execute('SELECT '+','.join(FIELDS)+' FROM public.alpaca_activities WHERE (activity_id,activity_type) IN ('+placeholders+')',tuple(x for key in part for x in key))
        result.extend(dict(zip(FIELDS,row)) for row in cursor.fetchall())
    return result

def window(before,required_after,upper,lookback_days):
    upper=utc(upper);after=utc(before) if before is not None else upper-timedelta(days=lookback_days)
    require(after<=upper,'CURSOR_FUTURE')
    if required_after is not None:after=min(after,utc(required_after))
    # Alpaca's filters use created_at and exclusive-looking before/after bounds.
    # Retain overlap; never treat an event/settlement timestamp as a query end.
    return after-timedelta(seconds=300),upper

def persist(connection,*,workflow_id,source,after,until,before,rows,policy_lock):
    require(connection.autocommit is False,'TRANSACTION_REQUIRED')
    checkpoint_id=workflow(workflow_id);after,until=utc(after),utc(until)
    require(after<until,'WINDOW_INVALID');require(re.fullmatch('[0-9a-f]{64}',source),'SOURCE_INVALID')
    expected=manifestation(rows);keys=[(x['activity_id'],x['activity_type']) for x in expected]
    require(len(encoded(expected).encode())<=67108864,'MANIFEST_LIMIT')
    expected_map={(x['activity_id'],x['activity_type']):x for x in expected}
    # SQL timestamps/cursors are opaque externally: record only bounded proof.
    with connection:
        policy_lock(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '1500ms'")
            cursor.execute("SET LOCAL statement_timeout = '15000ms'")
            cursor.execute("SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('jbravo:activity-ingestion'))")
            cursor.execute('SELECT source_fingerprint FROM public.activity_ingestion_checkpoints ORDER BY query_until DESC LIMIT 1')
            previous=cursor.fetchall();require(not previous or previous==[(source,)],'SOURCE_CHANGED')
            cursor.execute('SELECT last_after FROM public.reconcile_state WHERE id=2 FOR UPDATE')
            found=cursor.fetchall();require(len(found)<=1,'CURSOR_POPULATION')
            actual=found[0][0] if found else None
            require((utc(actual) if actual is not None else None)==(utc(before) if before is not None else None),'CURSOR_CHANGED')
            existing=ledger_rows(cursor,keys);prior=manifestation(existing)
            require(all(expected_map.get((x['activity_id'],x['activity_type']))==x for x in prior),'LEDGER_CONFLICT')
            present={(x['activity_id'],x['activity_type']) for x in prior}
            absent=[row for row in rows if (row['activity_id'],row['activity_type']) not in present]
            # Do not issue duplicate INSERTs against protected CRC originals.
            inserted=[]
            for offset in range(0,len(absent),200):
                part=absent[offset:offset+200]
                values=','.join('('+','.join('%s' if f!='raw' else 'CAST(%s AS JSONB)' for f in FIELDS)+')' for _ in part)
                cursor.execute('INSERT INTO public.alpaca_activities ('+','.join(FIELDS)+') VALUES '+values+' ON CONFLICT (activity_id,activity_type) DO NOTHING RETURNING activity_id,activity_type',tuple(row[f] for row in part for f in FIELDS))
                inserted.extend(cursor.fetchall())
            require(len(inserted)==len(set(inserted)) and set(inserted)<={(r['activity_id'],r['activity_type']) for r in absent},'INSERT_COUNT_INVALID')
            stored=manifestation(ledger_rows(cursor,keys));require(stored==expected,'LEDGER_READBACK')
            manifest_sha=digest(expected)
            cursor.execute('INSERT INTO public.activity_ingestion_checkpoints (checkpoint_id,source_fingerprint,recipe_sha256,query_axis,query_after,query_until,fetched_count,persisted_manifest,manifest_sha256,terminal_observed_at,cursor_before,cursor_after) VALUES (%s,%s,%s,%s,%s,%s,%s,CAST(%s AS JSONB),%s,%s,%s,%s)',(checkpoint_id,source,recipe_fingerprint(),'ALPACA_CREATED_AT',after,until,len(expected),encoded(expected),manifest_sha,datetime.now(timezone.utc),actual,until))
            cursor.execute('INSERT INTO public.reconcile_state (id,last_after,last_ran_at,updated_at) VALUES (2,%s,%s,now()) ON CONFLICT (id) DO UPDATE SET last_after=EXCLUDED.last_after,last_ran_at=EXCLUDED.last_ran_at,updated_at=EXCLUDED.updated_at WHERE public.reconcile_state.last_after IS NOT DISTINCT FROM %s RETURNING last_after',(until,datetime.now(timezone.utc),actual))
            changed=cursor.fetchall();require(len(changed)==1 and utc(changed[0][0])==until,'CURSOR_CAS')
            cursor.execute('SELECT source_fingerprint,query_after,query_until,fetched_count,persisted_manifest,manifest_sha256,cursor_after FROM public.activity_ingestion_checkpoints WHERE checkpoint_id=%s',(checkpoint_id,))
            verify=cursor.fetchall();require(len(verify)==1 and verify[0]==(source,after,until,len(expected),expected,manifest_sha,until),'CHECKPOINT_READBACK')
    # Returning/logging occurs only after the context acknowledged the commit.
    return {'checkpoint_id':checkpoint_id,'fetched':len(rows),'inserted':len(inserted),'matched_existing':len(rows)-len(inserted),'manifest_sha256':manifest_sha}

def consume(connection,*,workflow_id,source,required_after,required_until,now):
    checkpoint_id=workflow(workflow_id);required_after,required_until,now=utc(required_after),utc(required_until),utc(now)
    with connection.cursor() as cursor:
        cursor.execute('SELECT source_fingerprint,query_axis,query_after,query_until,fetched_count,persisted_manifest,manifest_sha256,cursor_after,recipe_sha256 FROM public.activity_ingestion_checkpoints WHERE checkpoint_id=%s',(checkpoint_id,))
        rows=cursor.fetchall();require(len(rows)==1,'MISSING')
        actual,axis,after,until,count,manifest,sha,cursor_after,recipe=rows[0]
        after,until=utc(after),utc(until)
        require(actual==source and axis=='ALPACA_CREATED_AT' and recipe==recipe_fingerprint(),'SOURCE_MISMATCH')
        require(after<required_after<=until==required_until<=now and utc(cursor_after)==until,'COVERAGE_INSUFFICIENT')
        require(type(count) is int and 0<=count<=MAX_ROWS and isinstance(manifest,list) and len(manifest)==count,'MANIFEST_INVALID')
        require(digest(manifest)==sha,'MANIFEST_INVALID')
        keys=[]
        for row in manifest:
            require(isinstance(row,dict) and set(row)=={'activity_id','activity_type','sha256'} and all(isinstance(v,str) and v for v in row.values()),'MANIFEST_INVALID')
            require(re.fullmatch('[0-9a-f]{64}',row['sha256']),'MANIFEST_INVALID');keys.append((row['activity_id'],row['activity_type']))
        require(len(keys)==len(set(keys)),'DUPLICATE_IDENTITY')
        require(manifestation(ledger_rows(cursor,keys))==manifest,'LEDGER_CHANGED')
        cursor.execute('SELECT last_after FROM public.reconcile_state WHERE id=2')
        current=cursor.fetchall()
        require(len(current)==1 and until<=utc(current[0][0])<=now,'CURSOR_REWOUND_OR_FUTURE')
    proof={'schema_version':1,'scope':SCOPE,'checkpoint_id':checkpoint_id,'source_fingerprint':source,'query_after':after.isoformat(),'required_after':required_after.isoformat(),'query_until':until.isoformat(),'rows':count,'manifest_sha256':sha}
    validate_proof(proof);return proof

def validate_proof(proof):
    require(isinstance(proof,dict) and set(proof)=={'schema_version','scope','checkpoint_id','source_fingerprint','query_after','required_after','query_until','rows','manifest_sha256'},'PUBLIC_PROOF_INVALID')
    require(type(proof['schema_version']) is int and proof['schema_version']==1 and proof['scope']==SCOPE,'PUBLIC_PROOF_INVALID')
    require(str(UUID(proof['checkpoint_id']))==proof['checkpoint_id'],'PUBLIC_PROOF_INVALID')
    require(all(isinstance(proof[k],str) and re.fullmatch('[0-9a-f]{64}',proof[k]) for k in ('source_fingerprint','manifest_sha256')),'PUBLIC_PROOF_INVALID')
    require(type(proof['rows']) is int and 0<=proof['rows']<=MAX_ROWS,'PUBLIC_PROOF_INVALID')
    require(utc(proof['query_after'])<utc(proof['required_after'])<=utc(proof['query_until']),'PUBLIC_PROOF_INVALID')
    require(utc(proof['query_until'])<=datetime.now(timezone.utc),'PUBLIC_PROOF_FUTURE')
    return proof

def decode_proof(text):
    require(isinstance(text,str) and len(text.encode())<=16384,'PUBLIC_PROOF_INVALID')
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,'PUBLIC_PROOF_DUPLICATE');out[k]=v
        return out
    def invalid(value):raise ValueError('invalid_constant')
    return validate_proof(json.loads(text,object_pairs_hook=unique,parse_constant=invalid))
