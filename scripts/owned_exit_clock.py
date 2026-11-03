"""Exact UTC nanosecond parsing and anonymous clock diagnostics; no I/O."""
import re
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone

class ClockError(ValueError):
    pass

def need(ok,code):
    if not ok:raise ClockError(code)

@dataclass(frozen=True)
class Instant:
    nanoseconds:int
    fraction_digits:int
    input_kind:str

EPOCH=datetime(1970,1,1,tzinfo=timezone.utc)
PATTERN=re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?')

def precise_instant(value):
    """Preserve every supplied digit, never round, truncate or assume a zone."""
    if type(value) is datetime:
        try:
            offset=value.utcoffset()
            need(value.tzinfo is not None and offset is not None,'TIME_ZONE_REQUIRED')
            need(type(offset) is timedelta and abs(offset)<timedelta(days=1)
                 and offset%timedelta(minutes=1)==timedelta(0),'TIME_OFFSET_INVALID')
            utc=value.astimezone(timezone.utc)
            base=utc.replace(microsecond=0);fraction=utc.microsecond*1000
        except ClockError:raise
        except (ValueError,OverflowError):raise ClockError('TIME_RANGE_INVALID') from None
        digits=6;kind='DATETIME'
    else:
        need(type(value) is str,'TIME_TYPE_UNSUPPORTED')
        need(len(value)<=64,'TIME_TEXT_BOUND')
        match=PATTERN.fullmatch(value)
        need(match is not None,'TIME_SYNTAX_INVALID')
        fraction_text,zone=match[2] or '',match[3]
        need(zone is not None,'TIME_ZONE_REQUIRED')
        need(zone=='Z' or int(zone[1:3])<24 and int(zone[4:6])<60,'TIME_OFFSET_INVALID')
        need(len(fraction_text)<=9,'TIME_PRECISION_UNSUPPORTED')
        try:
            base=datetime.fromisoformat(match[1]+('+00:00' if zone=='Z' else zone)).astimezone(timezone.utc)
        except ValueError:raise ClockError('TIME_CALENDAR_INVALID') from None
        except OverflowError:raise ClockError('TIME_RANGE_INVALID') from None
        fraction=int(fraction_text.ljust(9,'0')) if fraction_text else 0
        digits=len(fraction_text);kind='STRING'
    delta=base-EPOCH
    return Instant((delta.days*86400+delta.seconds)*1000000000+fraction,digits,kind)

def projection_relation(event,recorded):
    """Projection compatibility is explicitly NOT exact instant equality."""
    if event.nanoseconds==recorded.nanoseconds:return 'EXACT_INSTANT_MATCH'
    if event.nanoseconds%1000==0 and recorded.nanoseconds//1000*1000==event.nanoseconds:
        return 'MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL'
    return 'CONFLICT'

def describe(value,legacy_instant):
    kind='STRING' if type(value) is str else 'DATETIME' if type(value) is datetime else 'NULL' if value is None else 'UNSUPPORTED'
    result=dict(input_kind=kind,status='REFUSED',reason='NOT_PARSED',fraction_digits=None,
                submicrosecond_nonzero=None,timezone_state='UNRECOGNIZED',legacy_accepted=False,legacy_reason='NOT_TESTED')
    if type(value) is str and len(value)<=64:
        shape=PATTERN.fullmatch(value)
        if shape is not None:
            fraction=shape[2] or ''
            result.update(fraction_digits=len(fraction),submicrosecond_nonzero=any(c!='0' for c in fraction[6:]),
                          timezone_state='MISSING' if shape[3] is None else 'EXPLICIT_REQUIRES_VALIDATION')
    instant=None
    try:
        instant=precise_instant(value)
        result.update(status='PARSED',reason='EXACT_UTC_NANOSECONDS',fraction_digits=instant.fraction_digits,
                      submicrosecond_nonzero=instant.nanoseconds%1000!=0,
                      timezone_state='AWARE_DATETIME' if kind=='DATETIME' else 'EXPLICIT_OFFSET')
    except ClockError as error:
        result['reason']=BaseException.args.__get__(error)[0]
        if result['reason']=='TIME_ZONE_REQUIRED':result['timezone_state']='MISSING'
        elif result['reason']=='TIME_OFFSET_INVALID':result['timezone_state']='INVALID_OFFSET'
    try:
        legacy_instant(value);result.update(legacy_accepted=True,legacy_reason='ACCEPTED')
    except (ValueError,TypeError,OverflowError) as error:
        args=BaseException.args.__get__(error)
        result['legacy_reason']=args[0] if (len(args)==1 and type(args[0]) is str
            and args[0] in ('TIME_INVALID','TIME_PRECISION_UNSUPPORTED')) else 'LEGACY_REFUSED'
    return result,instant

def clock_report(event,trade,*,observed_before,legacy_instant):
    """No timestamps/IDs/hashes in result; does not qualify lot ownership.

    Existing recorded clocks are compared exactly. A raw nanosecond clock
    matching a DB microsecond truncation stays HELD: provenance for treating
    that projection as the same originating event is not supplied here.
    """
    values=dict(entry_time=trade.get('entry_time'),event_time=event.get('event_time'),
                observed_before=observed_before)
    if event.get('recorded_submitted_at') is not None:values['recorded_submitted_at']=event['recorded_submitted_at']
    envelope_state='NOT_SELECTED'
    if 'envelope_event_time_key_present' in event:
        if event['envelope_event_time_key_present'] is False:envelope_state='ABSENT'
        else:
            kinds={'string':'STRING','null':'JSON_NULL','number':'UNSUPPORTED_NUMBER','boolean':'UNSUPPORTED_BOOLEAN',
                   'array':'UNSUPPORTED_ARRAY','object':'UNSUPPORTED_OBJECT'}
            envelope_state=kinds.get(event.get('envelope_event_time_json_type'),'UNSUPPORTED')
            values['envelope_event_time']=event.get('envelope_event_time')
    fields={};parsed={}
    for name,value in values.items():fields[name],parsed[name]=describe(value,legacy_instant)
    report=dict(fields=fields,clock_qualified=False,reason='TIME_FIELD_REFUSED',first_legacy_failure=None,
                recorded_to_event='NOT_AVAILABLE',envelope_clock_state=envelope_state,
                envelope_to_db='NOT_AVAILABLE',recorded_to_envelope='NOT_AVAILABLE',ownership_proved=False)
    # Report every valid pair even if another field fails. These comparisons
    # are diagnostic evidence, never a bypass of the qualification gates.
    if parsed['event_time'] is not None and parsed.get('recorded_submitted_at') is not None:
        report['recorded_to_event']=projection_relation(parsed['event_time'],parsed['recorded_submitted_at'])
    if parsed['event_time'] is not None and parsed.get('envelope_event_time') is not None:
        report['envelope_to_db']=projection_relation(parsed['event_time'],parsed['envelope_event_time'])
    if parsed.get('recorded_submitted_at') is not None and parsed.get('envelope_event_time') is not None:
        report['recorded_to_envelope']='EXACT_INSTANT_MATCH' if parsed['recorded_submitted_at'].nanoseconds==parsed['envelope_event_time'].nanoseconds else 'CONFLICT'
    legacy={}
    for name in ('entry_time','event_time','observed_before'):
        try:legacy[name]=legacy_instant(values[name])
        except (ValueError,TypeError,OverflowError):
            report['first_legacy_failure']=name;break
    if report['first_legacy_failure'] is None:
        if not legacy['entry_time']<=legacy['event_time']<=legacy['observed_before']:
            report['first_legacy_failure']='TRAIL_CHRONOLOGY_CONFLICT'
        elif 'recorded_submitted_at' in values:
            try:
                raw=legacy_instant(values['recorded_submitted_at'])
                if raw!=legacy['event_time']:report['first_legacy_failure']='TRAIL_RECORDED_TIME_CONFLICT'
            except (ValueError,TypeError,OverflowError):report['first_legacy_failure']='recorded_submitted_at'
    if any(field['legacy_reason']=='LEGACY_REFUSED' for field in fields.values()):
        report['reason']='LEGACY_CLOCK_FAILURE_UNCLASSIFIED';return report
    if any(value is None for value in parsed.values()):return report
    entry,event_time,cutoff=(parsed[name].nanoseconds for name in ('entry_time','event_time','observed_before'))
    if not entry<=event_time<=cutoff:
        report['reason']='TRAIL_CHRONOLOGY_CONFLICT';return report
    if 'envelope_event_time' in parsed:
        if 'recorded_submitted_at' in parsed:
            if report['recorded_to_envelope']=='CONFLICT':
                report['reason']='TRAIL_ENVELOPE_CLOCK_CONFLICT';return report
        if report['envelope_to_db']=='CONFLICT':
            report['reason']='TRAIL_ENVELOPE_CLOCK_CONFLICT';return report
    if 'recorded_submitted_at' in parsed:
        relation=report['recorded_to_event']
        if relation=='MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL':
            report['reason']='CLOCK_CONVERSION_PROVENANCE_REQUIRED';return report
        if relation!='EXACT_INSTANT_MATCH':
            report['reason']='TRAIL_RECORDED_TIME_CONFLICT';return report
    if report['envelope_to_db']=='MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL':
        report['reason']='CLOCK_CONVERSION_PROVENANCE_REQUIRED';return report
    report.update(clock_qualified=True,reason='CLOCK_PREDICATES_PASSED_NOT_OWNERSHIP')
    return report

"""Interpret an explicitly paired producer clock and DB encoding; no I/O."""

def qualify_projection_clocks(event,trade,*,observed_before,legacy_instant):
    report=clock_report(event,trade,observed_before=observed_before,legacy_instant=legacy_instant)
    report.update(projection_accepted_as_encoding=False,db_and_original_same_instant=None)
    if report['clock_qualified']:
        if report['recorded_to_event']=='EXACT_INSTANT_MATCH' or report['envelope_to_db']=='EXACT_INSTANT_MATCH':
            report['db_and_original_same_instant']=True
        return report
    if report['reason']!='CLOCK_CONVERSION_PROVENANCE_REQUIRED':return report
    # Only the specific producer-envelope representation is supported. Missing
    # or contradictory originals never acquire provenance from proximity.
    if not (event.get('envelope_event_time_key_present') is True
            and event.get('envelope_event_time_json_type')=='string'
            and type(event.get('envelope_event_time')) is str
            and type(event.get('recorded_submitted_at')) is str
            and report['recorded_to_envelope']=='EXACT_INSTANT_MATCH'
            and report['recorded_to_event']=='MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL'
            and report['envelope_to_db']=='MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL'):
        return report
    entry=precise_instant(trade['entry_time']).nanoseconds
    stored=precise_instant(event['event_time']).nanoseconds
    original=precise_instant(event['recorded_submitted_at']).nanoseconds
    cutoff=precise_instant(observed_before).nanoseconds
    # Preserve the old recorded-DB chronology as well as checking the precise
    # original. A DB projection before the cutoff alone is not sufficient.
    if not entry<=stored<=cutoff:
        report['reason']='TRAIL_CHRONOLOGY_CONFLICT';return report
    if original<entry:
        report['reason']='ORIGINAL_SUBMISSION_BEFORE_ENTRY';return report
    if original>cutoff:
        report['reason']='ORIGINAL_SUBMISSION_AFTER_CUTOFF';return report
    # No tolerance or point equality: the exact encoding equation and the two
    # original clock fields have already been checked independently above.
    report.update(clock_qualified=True,reason='CLOCK_PREDICATES_PASSED_EXPLICIT_DB_PROJECTION',
                  projection_accepted_as_encoding=True,db_and_original_same_instant=False)
    return report

"""Offline historical evidence decoding and exact quantity checks.

No I/O, allocation inference, writer, policy installation or financial approval.
The link decoder follows the existing TRAIL_SUBMIT producer envelope, without
pretending a historical CLOSED trade is OPEN or requiring a full-close child.
"""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from uuid import UUID
import re


class EvidenceError(ValueError):
    pass


def require(value, reason):
    if not value:
        raise EvidenceError(reason)


def number(value):
    require(type(value) in (str, int, Decimal) and len(str(value)) <= 64, 'NUMBER_INVALID')
    try:
        result = Decimal(value)
    except (ValueError, ArithmeticError):
        raise EvidenceError('NUMBER_INVALID') from None
    require(result.is_finite() and result > 0 and -12 <= result.as_tuple().exponent <= 15
            and -12 <= result.adjusted() <= 15, 'NUMBER_INVALID')
    return result


def identity(value):
    require(type(value) is str and len(value) == 36, 'IDENTITY_INVALID')
    try:
        result = str(UUID(value))
    except ValueError:
        raise EvidenceError('IDENTITY_INVALID') from None
    require(result == value, 'IDENTITY_INVALID')
    return result


def instant(value):
    require(type(value) in (str, datetime), 'TIME_INVALID')
    try:
        if isinstance(value, str):
            require(len(value) <= 64, 'TIME_INVALID')
            match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})', value)
            require(match is not None, 'TIME_INVALID')
            fraction = match[2] or ''
            require(match[3] == 'Z' or (int(match[3][1:3]) < 24 and int(match[3][4:6]) < 60), 'TIME_INVALID')
            require(not any(c != '0' for c in fraction[6:]), 'TIME_PRECISION_UNSUPPORTED')
            value = match[1] + ('.' + fraction[:6].ljust(6, '0') if fraction else '') + ('+00:00' if match[3] == 'Z' else match[3])
            result = datetime.fromisoformat(value)
        else:
            result = value
        require(result.tzinfo is not None and result.utcoffset() is not None, 'TIME_INVALID')
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise EvidenceError('TIME_INVALID') from None


def trail_context(event, trade, *, observed_before):
    """Return a private advisory link; it is not an ownership adjudication."""
    require(type(trade.get('trade_id')) is int and trade['trade_id'] > 0
            and trade.get('status') in ('OPEN', 'CLOSED'), 'TRADE_CONTEXT_INVALID')
    require(type(event.get('event_id')) is int and event['event_id'] > 0
            and event.get('event_type') == 'TRAIL_SUBMIT'
            and event.get('raw_shape_valid') is True
            and event.get('envelope_event_type') == 'TRAIL_SUBMIT'
            and event.get('metadata_event_type') == 'TRAIL_SUBMIT'
            and event.get('symbol') == trade.get('symbol'), 'TRAIL_ENVELOPE_INVALID')
    parent = identity(trade.get('entry_order_id'))
    child = identity(event.get('order_id'))
    require(identity(event.get('parent_order_id')) == parent and child != parent
            and event.get('envelope_order_id') == child, 'TRAIL_IDENTITY_CONFLICT')
    require(number(event.get('qty')) == number(trade.get('qty')), 'TRAIL_SUBMISSION_QUANTITY_CONFLICT')
    require(type(event.get('top_parent_key_present')) is bool, 'TRAIL_OPTIONAL_PARENT_INVALID')
    if event['top_parent_key_present']:
        require(identity(event.get('top_parent_order_id')) == parent, 'TRAIL_IDENTITY_CONFLICT')
    clock = qualify_projection_clocks(event, trade, observed_before=observed_before, legacy_instant=instant)
    require(clock['clock_qualified'], clock['reason'])
    return dict(parent=parent, child=child, event_id=event['event_id'],
                status='PRODUCER_ENVELOPE_CORROBORATED_NOT_LOT_PROOF',
                clock_reason=clock['reason'],projection_accepted_as_encoding=clock['projection_accepted_as_encoding'],
                db_and_original_same_instant=clock['db_and_original_same_instant'])


def quantity_report(lots, exits, *, reserved_activity_ids=()):
    """Check an explicitly supplied candidate, never choose or split a fill.

    Lots: case, quantity. Exits: case, activity_id, quantity. These are proposed
    assignments, not authenticated inputs. Private fill IDs never leave output.
    Source/account/row bindings and all physical order owners remain required
    in the eventual reviewed observation/correction caller.
    """
    require(type(lots) is list and 1 <= len(lots) <= 6 and type(exits) is list
            and len(exits) <= 512, 'SCOPE_INVALID')
    require(type(reserved_activity_ids) in (tuple, list, set, frozenset), 'RESERVATIONS_INVALID')
    quantities = {}
    for row in lots:
        case = row.get('case')
        require(type(case) is int and 1 <= case <= 6 and case not in quantities, 'CASE_INVALID')
        quantities[case] = number(row.get('quantity'))
    assigned = {case: Decimal(0) for case in quantities}
    seen = set()
    with localcontext() as context:
        context.prec = 80
        for row in exits:
            case, key = row.get('case'), row.get('activity_id')
            require(type(case) is int and case in quantities and type(key) is str
                    and 0 < len(key) <= 256, 'FILL_SCOPE_INVALID')
            require(key not in seen and key not in reserved_activity_ids, 'FILL_REUSED_OR_RESERVED')
            seen.add(key)
            assigned[case] += number(row.get('quantity'))
        results = []
        for case in sorted(quantities):
            delta = quantities[case] - assigned[case]
            results.append(dict(case=case, entry_quantity=str(quantities[case]),
                proposed_exit_quantity=str(assigned[case]),
                quantity_status='BALANCED' if not delta else 'MISSING_EXIT_QUANTITY' if delta > 0 else 'EXCESS_EXIT_QUANTITY',
                unresolved_quantity=str(abs(delta)), ownership_proved=False,
                ledger_correction_supported=False, allocation_adopted=False))
    return dict(status='CANDIDATE_ARITHMETIC_ONLY', cases=results,
                financial_eligible=False, fees=None, net=None, financial_writes=0)


def fee_exception_report(wire, *, parse_evidence, now):
    """Use the existing exact 008/correction decoder, retain only four facts.

    The returned report excludes IDs, aliases, tokens, timestamps and hashes.
    A future authorized caller reads the existing private file on host only;
    this helper itself has no file, database or provider access.
    """
    document, _ = parse_evidence(wire, now=now)
    rows = [dict(zip(document['fields'], row)) for row in document['rows']]
    targets = [row for row in rows if row['comparison'] == 'UNSEEN']
    require(len(targets) == 4, 'FEE_EXCEPTION_SCOPE')
    window = document['window']
    result = []
    for index, row in enumerate(targets, 1):
        if window is None:
            relation = 'WINDOW_UNAVAILABLE'
        elif row['provider_created_at'] is None:
            relation = 'CLOCK_UNAVAILABLE'
        else:
            created, after, until = map(instant, (row['provider_created_at'], *window))
            relation = ('BEFORE' if created < after else 'AT_AFTER' if created == after else
                        'INSIDE' if created < until else 'AT_UNTIL' if created == until else 'AFTER')
        result.append(dict(case=index, stored_currency=row['stored_currency'],
            actual_currency='NOT_RETURNED', creation_relation=relation,
            raw_creation_type=row['raw_created_type'], alias_state=row['alias_state'],
            alias_count=len(row['aliases']), parent_state=row['parent_state'],
            currency_correction_supported=False, absence_means_zero=False))
    return dict(status='FOUR_EXCEPTION_FACTS_ONLY', cases=result,
        currency_inferred=0, fee_requests_made=0, financial_eligible=False,
        fees=None, net=None, financial_writes=0)
