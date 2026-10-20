-- UNACTIVATED implementation artifact. No grants, cases or allocations are applied.
-- Apply only to an independently attested test database, or after separate exact
-- production schema approval. No original trade/event/activity rows are rewritten.
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET LOCAL search_path = pg_catalog;
SET LOCAL lock_timeout = '1500ms';
SET LOCAL statement_timeout = '5000ms';
SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('jbravo:trade-disposition:policy'));
-- The existing administrator creates foreign keys without granting original-table
-- REFERENCES/TRIGGER privileges to the canonical owner. Never switch roles here.
DO $migration_actor$
BEGIN
    IF current_user IS DISTINCT FROM session_user OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles r WHERE r.rolname=current_user AND r.rolsuper
    ) THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_MIGRATION_ADMIN_REQUIRED';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_catalog.pg_roles r ON r.oid=c.relowner
        WHERE n.nspname='public' AND c.relname='trade_disposition_policy'
          AND c.relkind='r' AND r.rolname='jbravo_v7_owner'
    ) THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_MIGRATION_POLICY_OWNER';
    END IF;
END
$migration_actor$;
-- Persist the installation marker outside the optional overlay tables. Never
-- infer "not installed" when installed tables have subsequently disappeared.
ALTER TABLE public.trade_disposition_policy ADD COLUMN allocation_schema_version pg_catalog.int4 NOT NULL DEFAULT 0
    CHECK (allocation_schema_version IN (0,1));

CREATE TABLE public.trade_allocation_cases (
    allocation_id pg_catalog.uuid PRIMARY KEY,
    trade_id pg_catalog.int8 NOT NULL REFERENCES public.trades(trade_id),
    state pg_catalog.text NOT NULL CHECK (state IN ('ACTIVE','REVOKED')),
    allocation_revision pg_catalog.int8 NOT NULL CHECK (allocation_revision BETWEEN 1 AND 32),
    request pg_catalog.jsonb NOT NULL,
    manifest pg_catalog.jsonb NOT NULL,
    CHECK (trade_id = 110),
    CHECK (pg_catalog.octet_length(request::text) <= 8192),
    CHECK (pg_catalog.octet_length(manifest::text) <= 196608)
);
CREATE UNIQUE INDEX trade_allocation_one_active_trade ON public.trade_allocation_cases(trade_id) WHERE state='ACTIVE';
CREATE TABLE public.trade_allocation_audit (
    approval_ref pg_catalog.text PRIMARY KEY,
    allocation_id pg_catalog.uuid NOT NULL REFERENCES public.trade_allocation_cases(allocation_id),
    request_digest pg_catalog.text NOT NULL CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    request pg_catalog.jsonb NOT NULL,
    manifest pg_catalog.jsonb NOT NULL,
    result pg_catalog.jsonb NOT NULL,
    policy_revision pg_catalog.int8 NOT NULL UNIQUE CHECK (policy_revision > 0),
    CHECK (pg_catalog.octet_length(approval_ref) BETWEEN 1 AND 2048),
    CHECK (pg_catalog.octet_length(request::text) <= 8192),
    CHECK (pg_catalog.octet_length(manifest::text) <= 196608),
    CHECK (pg_catalog.octet_length(result::text) <= 8192)
);
CREATE TABLE public.trade_allocation_order_claims (
    order_id pg_catalog.text PRIMARY KEY,
    allocation_id pg_catalog.uuid NOT NULL REFERENCES public.trade_allocation_cases(allocation_id),
    UNIQUE(order_id,allocation_id),
    CHECK (pg_catalog.octet_length(order_id) BETWEEN 1 AND 128)
);
CREATE TABLE public.trade_allocation_fill_claims (
    activity_id pg_catalog.text NOT NULL,
    activity_type pg_catalog.text NOT NULL CHECK (activity_type='FILL'),
    order_id pg_catalog.text NOT NULL,
    allocation_id pg_catalog.uuid NOT NULL REFERENCES public.trade_allocation_cases(allocation_id),
    quantity pg_catalog.numeric NOT NULL CHECK (quantity > 0 AND quantity::text NOT IN ('NaN','Infinity','-Infinity')),
    PRIMARY KEY(activity_id,activity_type),
    FOREIGN KEY(order_id,allocation_id) REFERENCES public.trade_allocation_order_claims(order_id,allocation_id)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY(activity_id,activity_type) REFERENCES public.alpaca_activities(activity_id,activity_type),
    CHECK (pg_catalog.octet_length(activity_id) BETWEEN 1 AND 256)
);

CREATE FUNCTION public.trade_allocation_immutable() RETURNS pg_catalog.trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $body$
BEGIN
    RAISE EXCEPTION 'TRADE_ALLOCATION_IMMUTABLE';
END
$body$;
CREATE TRIGGER trade_allocation_audit_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON public.trade_allocation_audit
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_immutable();
CREATE TRIGGER trade_allocation_fill_claims_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON public.trade_allocation_fill_claims
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_immutable();
CREATE TRIGGER trade_allocation_order_claims_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON public.trade_allocation_order_claims
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_immutable();
CREATE TRIGGER trade_allocation_cases_no_truncate BEFORE TRUNCATE ON public.trade_allocation_cases
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_immutable();

CREATE FUNCTION public.trade_allocation_lock_policy() RETURNS pg_catalog.trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $body$
BEGIN
    IF pg_catalog.current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_ISOLATION_INVALID';
    END IF;
    PERFORM pg_catalog.set_config('lock_timeout','1500ms',true);
    PERFORM pg_catalog.set_config('statement_timeout','5000ms',true);
    PERFORM pg_catalog.set_config('TimeZone','UTC',true);
    PERFORM pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'));
    RETURN NULL;
END
$body$;
CREATE TRIGGER trade_allocation_trade_statement BEFORE INSERT OR UPDATE OR DELETE ON public.trades
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_lock_policy();
CREATE TRIGGER trade_allocation_activity_statement BEFORE INSERT OR UPDATE OR DELETE ON public.alpaca_activities
FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_lock_policy();

CREATE FUNCTION public.trade_allocation_case_guard() RETURNS pg_catalog.trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $body$
BEGIN
    IF pg_catalog.current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_ISOLATION_INVALID';
    END IF;
    PERFORM pg_catalog.set_config('lock_timeout','1500ms',true);
    PERFORM pg_catalog.set_config('statement_timeout','5000ms',true);
    PERFORM pg_catalog.set_config('TimeZone','UTC',true);
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('jbravo:trade-disposition:policy'));
    IF TG_OP='INSERT' THEN
        IF NEW.state<>'ACTIVE' OR NEW.allocation_revision<>1 THEN
            RAISE EXCEPTION 'TRADE_ALLOCATION_CASE_INVALID';
        END IF;
    ELSIF TG_OP='UPDATE' THEN
        IF NEW.allocation_id IS DISTINCT FROM OLD.allocation_id
          OR NEW.trade_id IS DISTINCT FROM OLD.trade_id OR NEW.manifest IS DISTINCT FROM OLD.manifest
          OR NEW.allocation_revision<>OLD.allocation_revision+1 OR NEW.state=OLD.state
          OR (NEW.state='ACTIVE')<>(NEW.allocation_revision % 2=1) THEN
            RAISE EXCEPTION 'TRADE_ALLOCATION_CASE_INVALID';
        END IF;
    ELSE
        RAISE EXCEPTION 'TRADE_ALLOCATION_IMMUTABLE';
    END IF;
    IF NEW.request->>'allocation_id' IS DISTINCT FROM NEW.allocation_id::text
      OR NEW.request->>'trade_id' IS DISTINCT FROM NEW.trade_id::text
      OR NEW.request->>'action' IS DISTINCT FROM (CASE WHEN NEW.state='ACTIVE' THEN 'APPLY' ELSE 'REVOKE' END) THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_CASE_INVALID';
    END IF;
    RETURN NEW;
END
$body$;
CREATE TRIGGER trade_allocation_cases_guard BEFORE INSERT OR UPDATE OR DELETE ON public.trade_allocation_cases
FOR EACH ROW EXECUTE PROCEDURE public.trade_allocation_case_guard();

CREATE FUNCTION public.trade_allocation_guard_original() RETURNS pg_catalog.trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $body$
DECLARE before_row pg_catalog.jsonb; after_row pg_catalog.jsonb;
BEGIN
    IF TG_OP<>'INSERT' THEN before_row=pg_catalog.to_jsonb(OLD); END IF;
    IF TG_OP<>'DELETE' THEN after_row=pg_catalog.to_jsonb(NEW); END IF;
    IF TG_TABLE_NAME='trades' THEN
        IF EXISTS (SELECT 1 FROM public.trade_allocation_cases c WHERE c.state='ACTIVE' AND
            (c.trade_id::text=before_row->>'trade_id' OR c.trade_id::text=after_row->>'trade_id'))
          OR EXISTS (SELECT 1 FROM public.trade_allocation_order_claims o
            JOIN public.trade_allocation_cases c ON c.allocation_id=o.allocation_id AND c.state IN ('ACTIVE','REVOKED')
            WHERE (o.order_id IN (before_row->>'entry_order_id',before_row->>'exit_order_id')
                   AND c.trade_id::text IS DISTINCT FROM before_row->>'trade_id')
               OR (o.order_id IN (after_row->>'entry_order_id',after_row->>'exit_order_id')
                   AND c.trade_id::text IS DISTINCT FROM after_row->>'trade_id')) THEN
            RAISE EXCEPTION 'TRADE_ALLOCATION_ORIGINAL_PROTECTED';
        END IF;
    ELSIF TG_TABLE_NAME='alpaca_activities' THEN
        -- Preserve the existing producer's INSERT ... ON CONFLICT DO NOTHING.
        -- Only an identical native business row whose complete existing row is
        -- still the audited snapshot may reach PostgreSQL's normal conflict path.
        IF TG_OP='INSERT' THEN
          IF EXISTS (
            SELECT 1 FROM public.alpaca_activities activity
            JOIN public.trade_allocation_fill_claims claim ON claim.activity_id=activity.activity_id
              AND claim.activity_type=activity.activity_type
            JOIN public.trade_allocation_cases c ON c.allocation_id=claim.allocation_id
              AND c.state IN ('ACTIVE','REVOKED')
            JOIN public.trade_allocation_audit audit ON audit.allocation_id=c.allocation_id
              AND audit.approval_ref=c.request->>'approval_ref' AND audit.request=c.request AND audit.manifest=c.manifest
              AND audit.result->>'state'=c.state AND audit.result->>'allocation_revision'=c.allocation_revision::text
            CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(c.manifest->'fills') bound(value)
            WHERE activity.activity_id=NEW.activity_id AND activity.activity_type=NEW.activity_type
              AND bound.value->'snapshot'=pg_catalog.to_jsonb(activity)
              AND (after_row-'activity_pk'-'created_at')=(pg_catalog.to_jsonb(activity)-'activity_pk'-'created_at')
        ) THEN
            RETURN NEW;
          END IF;
        END IF;
        IF EXISTS (SELECT 1 FROM public.trade_allocation_fill_claims f
            JOIN public.trade_allocation_cases c ON c.allocation_id=f.allocation_id AND c.state IN ('ACTIVE','REVOKED')
            WHERE (f.activity_id=before_row->>'activity_id' AND f.activity_type=before_row->>'activity_type')
               OR (f.activity_id=after_row->>'activity_id' AND f.activity_type=after_row->>'activity_type'))
          OR EXISTS (SELECT 1 FROM public.trade_allocation_order_claims o
            JOIN public.trade_allocation_cases c ON c.allocation_id=o.allocation_id AND c.state IN ('ACTIVE','REVOKED')
            WHERE (before_row->>'activity_type'='FILL' AND o.order_id=before_row->>'order_id')
               OR (after_row->>'activity_type'='FILL' AND o.order_id=after_row->>'order_id')) THEN
            RAISE EXCEPTION 'TRADE_ALLOCATION_ORIGINAL_PROTECTED';
        END IF;
    ELSE
        RAISE EXCEPTION 'TRADE_ALLOCATION_TRIGGER_TARGET';
    END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END
$body$;
CREATE TRIGGER trade_allocation_trade_guard BEFORE INSERT OR UPDATE OR DELETE ON public.trades
FOR EACH ROW EXECUTE PROCEDURE public.trade_allocation_guard_original();
CREATE TRIGGER trade_allocation_activity_guard BEFORE INSERT OR UPDATE OR DELETE ON public.alpaca_activities
FOR EACH ROW EXECUTE PROCEDURE public.trade_allocation_guard_original();
CREATE FUNCTION public.trade_allocation_marker_immutable() RETURNS pg_catalog.trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $body$
BEGIN
    IF NEW.allocation_schema_version IS DISTINCT FROM OLD.allocation_schema_version THEN
        RAISE EXCEPTION 'TRADE_ALLOCATION_SCHEMA_MARKER_IMMUTABLE';
    END IF;
    RETURN NEW;
END
$body$;
-- Transfer only the new sidecars/functions, atomically before publishing marker1.
-- Original trades/activities/policy ownership and every original ACL are untouched.
ALTER TABLE public.trade_allocation_cases OWNER TO jbravo_v7_owner;
ALTER TABLE public.trade_allocation_audit OWNER TO jbravo_v7_owner;
ALTER TABLE public.trade_allocation_order_claims OWNER TO jbravo_v7_owner;
ALTER TABLE public.trade_allocation_fill_claims OWNER TO jbravo_v7_owner;
ALTER FUNCTION public.trade_allocation_immutable() OWNER TO jbravo_v7_owner;
ALTER FUNCTION public.trade_allocation_lock_policy() OWNER TO jbravo_v7_owner;
ALTER FUNCTION public.trade_allocation_case_guard() OWNER TO jbravo_v7_owner;
ALTER FUNCTION public.trade_allocation_guard_original() OWNER TO jbravo_v7_owner;
ALTER FUNCTION public.trade_allocation_marker_immutable() OWNER TO jbravo_v7_owner;
-- Deliberately no grants, SECURITY DEFINER, native-claim activation, or historical updates.
UPDATE public.trade_disposition_policy SET allocation_schema_version=1 WHERE singleton=1 AND allocation_schema_version=0;
CREATE TRIGGER trade_allocation_marker_guard BEFORE UPDATE OF allocation_schema_version ON public.trade_disposition_policy
FOR EACH ROW EXECUTE PROCEDURE public.trade_allocation_marker_immutable();
COMMIT;
