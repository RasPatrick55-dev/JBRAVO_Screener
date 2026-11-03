"""Rerun all 17 existing CRC110 tests against the candidate modules.

The one fixture adaptation supplies the new nullable group-reference SELECT
column for legacy rows. No runtime validation, source pins or trigger fixture
bodies are relaxed. Original fixture file remains unchanged.
"""
import sys,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).resolve().parent))
SOURCE=HERE/'tests/test_trade_allocation.py'
spec=importlib.util.spec_from_file_location('owned_exit_legacy_crc_fixture',SOURCE)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
LegacyAllocationTests=fixture.AllocationTests
