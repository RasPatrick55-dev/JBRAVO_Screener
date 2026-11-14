"""Pinned PythonAnywhere-only entry; generic primary_paper_pipeline remains portable.

The release must bind its PA task to this wrapper. Missing or blocked generation
locks refuse before the underlying workflow is imported. This is not an optional
flag or an environment-controlled safety switch.
"""
from scripts import candidate007_worker_fence as generation_fence


def main(argv=None):
    def work():
        from scripts.primary_paper_pipeline import main as workflow
        return workflow(argv)
    return generation_fence.run_guarded(generation_fence.DIRECTORY, work)


if __name__ == '__main__':
    raise SystemExit(main())
