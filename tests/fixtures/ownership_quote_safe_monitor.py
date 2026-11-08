"""Exact quote-safe factory from reviewed monitor006 overlay, offline fixture."""


def require(value, code):
    if not value:
        raise ValueError(code)


def quote_safe_engine(Engine):
    """Keep native stop maintenance reachable behind a queued no-quote exit.

    Native strategy decisions, queued intent and execution guards are unchanged.
    This copies only the pinned engine's existing stop-maintenance branch, using
    its own numeric/terminal helpers and durable _perform path. No quote exits,
    entry, cancellation, synthetic fill or intent rewrite is added.
    """
    core = Engine.manage.__globals__
    rules = core['rules']; quantity = core['quantity']; terminal = core['TERMINAL']
    Decimal = core['Decimal']; ROUND_FLOOR = core['ROUND_FLOOR']

    class QuoteSafeEngine(Engine):
        def manage(self, signal, records, *, now, quote, regular_open):
            payload = signal['payload']; position = payload.get('position')
            rejected = sum(r['role'] == 'stop' and r['state'] == 'REJECTED' for r in records)
            # Match pinned native partial classification exactly: queued, not
            # protective-stop breached, PARTIAL_5PCT only, and more than1share.
            # Do not cancel existing protection for a partial if native's future
            # missing-stop policy has already exhausted its three-rejection cap.
            competing = any(r['role'] in {'exit','partial'} and r.get('observation') and
                            r['observation']['status'] not in terminal for r in records)
            if position and quote is not None and rejected >= 3 and not competing:
                target = max(position['trailing_stop'] or 0, position['atr_stop'] or 0)
                breached = rules.number(quote['bid'], positive=True) <= Decimal(str(target))
                local = rules.utc(now).astimezone(rules.NY)
                queued = bool(payload['pending_exit_reasons']) and local.date().isoformat() >= payload['pending_exit_session']
                if queued and not breached and payload['pending_exit_reasons'] == ['PARTIAL_5PCT'] and quantity(position['qty']) > 1:
                    self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
            before = len(self.events)
            super().manage(signal, records, now=now, quote=quote, regular_open=regular_open)
            if quote is not None or len(self.events) == before or self.events[-1]['code'] != 'EXIT_QUOTE_UNAVAILABLE':
                return
            require(self.pipeline_orders is True, 'queued_stop_pipeline_required')
            if any(r['state'] in {'CLAIMED','UNKNOWN'} for r in records):
                self.note(signal, 'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION'); return
            position = signal['payload'].get('position')
            require(position is not None and bool(signal['payload']['pending_exit_reasons']), 'queued_stop_intent_required')
            own_qty = quantity(position['qty']); symbol = position['symbol']
            active = [r for r in records if r['role'] in {'stop','exit','partial'} and
                      r.get('observation') and r['observation']['status'] not in terminal]
            require(not any(r['role'] in {'exit','partial'} for r in active), 'queued_stop_competing_exit')
            stops = [r for r in active if r['role'] == 'stop']
            require(len(stops) <= 1, 'strategy_multiple_active_stops')
            target = max(position['trailing_stop'] or 0, position['atr_stop'] or 0)
            desired = Decimal(str(target)).quantize(Decimal('.01'), rounding=ROUND_FLOOR)
            rejected_stops = [r for r in records if r['role'] == 'stop' and r['state'] == 'REJECTED']
            require(all(r.get('operation') in {'POST','PATCH'} for r in rejected_stops), 'stop_rejection_operation_unknown')
            rejected_patches = [r for r in rejected_stops if r['operation'] == 'PATCH']
            require(all(isinstance(r.get('request'), dict) and isinstance(r['request'].get('order_id'), str)
                        and bool(r['request']['order_id']) for r in rejected_patches), 'stop_rejection_target_unknown')
            if stops:
                old = stops[0]['observation']
                remaining = quantity(old['qty']) - quantity(old.get('filled_qty', 0))
                if old['status'] in {'pending_new','pending_cancel','pending_replace','held'}:
                    self.note(signal, 'STOP_PENDING_CONFIRMATION'); return
                if quantity(old.get('filled_qty', 0)):
                    self.note(signal, 'PARTIAL_STOP_FILL_PENDING'); return
                old_price = rules.number(old['stop_price'], positive=True)
                if desired <= old_price and remaining == own_qty:
                    self.note(signal, 'REGULAR_STOP_CONFIRMED_NOT_EXTENDED_HOURS' if not regular_open else 'REGULAR_STOP_OBSERVED')
                    return
                # Reserve one attempt under native's three-rejection missing-stop
                # policy, and never repeat a rejected PATCH of this live stop.
                # Keep all historical POST/PATCH outcomes unchanged and visible.

                if rejected >= 2 or any(r['request']['order_id'] == old['id'] for r in rejected_patches):
                    self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
                sequence = max(r['sequence'] for r in records if r['role'] == 'stop') + 1
                self._perform(signal, 'stop', sequence, dict(symbol=symbol, qty=own_qty, side='sell', type='stop',
                    stop_price=float(max(old_price, desired)), time_in_force='gtc', extended_hours=False),
                    operation='PATCH', order_id=old['id'])
                return
            if rejected >= 2:
                self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
            sequence = 1 + max((r['sequence'] for r in records if r['role'] == 'stop'), default=-1)
            require(desired > 0, 'strategy_protective_stop_invalid')
            self._perform(signal, 'stop', sequence, dict(symbol=symbol, qty=own_qty, side='sell', type='stop',
                stop_price=float(desired), time_in_force='gtc', extended_hours=False))
    return QuoteSafeEngine
