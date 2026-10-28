"""Small, nonsecret phase receipts; never serialize arbitrary exception messages."""
import time


class StrategyBudgetError(RuntimeError):
    def __init__(self, code, phase):
        self.code, self.phase = code, phase
        super().__init__(code)


class StrategyInputError(ValueError):
    def __init__(self, code):
        self.code, self.phase = code, 'PREPARED_INPUT_VALIDATION'
        super().__init__(code)


def checkpoint(phase, *, deadline=None, stop_file=None):
    if stop_file and stop_file.exists():
        raise StrategyBudgetError('STOP_FILE_PRESENT', phase)
    if deadline is not None and time.monotonic() >= deadline:
        raise StrategyBudgetError('WALL_CLOCK_BUDGET_EXCEEDED', phase)


def diagnostic(error, phase, started, *, deadline=None):
    code = 'VALIDATION_OR_RUNTIME_FAILURE'
    if isinstance(error, (StrategyBudgetError, StrategyInputError)):
        code, phase = error.code, error.phase
    elif isinstance(error, RuntimeError) and str(error) == 'Research stopped by deadline or stop file':
        code = 'PORTFOLIO_DEADLINE_OR_STOP_FILE'
    return dict(error_type=type(error).__name__, code=code, phase=phase,
                elapsed_seconds=round(time.monotonic()-started, 3),
                remaining_seconds=None if deadline is None else round(max(0., deadline-time.monotonic()), 3))
