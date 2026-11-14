"""Explicit PAPER-only context for the existing scheduled primary pipeline.

The shell launcher loads configured credentials; this module never discovers
them or changes an endpoint. It declares paper context only after checking the
canonical paper endpoint and rejecting an explicitly conflicting declaration.
"""
import json
import os
import sys


PAPER_ENDPOINT = 'https://paper-api.alpaca.markets'


def configure_paper_context(environ):
    if environ.get('APCA_API_BASE_URL') != PAPER_ENDPOINT:
        raise ValueError('primary_paper_endpoint_required')
    if environ.get('JBR_EXEC_PAPER') not in (None, 'true'):
        raise ValueError('primary_paper_context_conflict')
    environ['JBR_EXEC_PAPER'] = 'true'


def main(argv=None):
    try:
        configure_paper_context(os.environ)
    except ValueError as failure:
        print(json.dumps({'status': 'failed', 'reason': str(failure)}), flush=True)
        return 1
    # Keep the primary argument, single-flight, readiness and policy contracts.
    # This import is deferred so refused contexts cannot initialize the pipeline.
    from scripts import candidate007_worker_fence as generation_fence
    def work():
        from scripts.primary_pipeline import main as primary_main
        return primary_main(argv)
    return generation_fence.run_guarded(generation_fence.DIRECTORY, work)


if __name__ == '__main__':
    raise SystemExit(main())
