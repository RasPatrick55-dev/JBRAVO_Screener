from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_HEADER = (
    "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,"
    "entry_price,adv20,atrp,source,sma9,ema20,sma180,rsi14,passed_gates,gate_fail_reason"
)

BANNED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("live_alpaca_endpoint", re.compile(r"https?://api\.alpaca\.markets", re.IGNORECASE)),
    (
        "live_trading_enabled_claim",
        re.compile(
            r"\blive[- ]trading\b.{0,60}\b(enabled|supported|default|active)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "production_trading_enabled_claim",
        re.compile(
            r"\bproduction\b.{0,30}\btrading\b.{0,30}\b(enabled|supported|active)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "csv_source_of_truth_claim",
        re.compile(r"\bcsvs?\s+(are|is)\s+the\s+source\s+of\s+truth\b", re.IGNORECASE),
    ),
    (
        "latest_candidates_source_of_truth_claim",
        re.compile(
            r"(latest_candidates|top_candidates)\.csv.{0,80}source\s+of\s+truth",
            re.IGNORECASE,
        ),
    ),
    (
        "latest_candidates_authoritative_claim",
        re.compile(
            r"(latest_candidates|top_candidates)\.csv.{0,80}\b(is|are)\s+(?!not\b)(authoritative|canonical)\b",
            re.IGNORECASE,
        ),
    ),
)

_IGNORE_CONTEXT_TOKENS: tuple[str, ...] = (
    "paper-only",
    "paper only",
    "not supported",
    "do not use",
    "never use",
    "non-authoritative",
    "debug export",
)

HELP_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "python -m scripts.run_pipeline --help",
        (sys.executable, "-m", "scripts.run_pipeline", "--help"),
    ),
    (
        "python -m scripts.execute_trades --help",
        (sys.executable, "-m", "scripts.execute_trades", "--help"),
    ),
    (
        "python -m scripts.screener --help",
        (sys.executable, "-m", "scripts.screener", "--help"),
    ),
)


def _canonical_header() -> str:
    try:
        from scripts.fallback_candidates import CANONICAL_COLUMNS

        header = ",".join(str(column) for column in CANONICAL_COLUMNS)
        return header or DEFAULT_CANONICAL_HEADER
    except Exception:
        return DEFAULT_CANONICAL_HEADER


def _iter_markdown_docs(docs_dir: Path) -> Iterable[Path]:
    for path in sorted(docs_dir.rglob("*.md")):
        relative = path.relative_to(docs_dir)
        if relative.parts and relative.parts[0].lower() == "archive":
            continue
        yield path


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _line_text(text: str, line_number: int) -> str:
    lines = text.splitlines()
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def _scan_banned_phrases(base_dir: Path, docs_dir: Path) -> list[str]:
    def _ignore_match(text: str, start: int, end: int, line_number: int, label: str) -> bool:
        line_lower = _line_text(text, line_number).lower()
        if f"docs-check: allow-{label}" in line_lower:
            return True
        if any(token in line_lower for token in _IGNORE_CONTEXT_TOKENS):
            return True
        lo = max(0, start - 160)
        hi = min(len(text), end + 160)
        context = text[lo:hi].lower()
        return any(token in context for token in _IGNORE_CONTEXT_TOKENS)

    findings: list[str] = []
    for path in _iter_markdown_docs(docs_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            findings.append(f"[DOC_READ] {path.relative_to(base_dir)}: {exc}")
            continue
        for label, pattern in BANNED_PATTERNS:
            for match in pattern.finditer(text):
                line_number = _line_number(text, match.start())
                if _ignore_match(text, match.start(), match.end(), line_number, label):
                    continue
                snippet = _line_text(text, line_number).strip()
                if len(snippet) > 140:
                    snippet = snippet[:137] + "..."
                findings.append(
                    f"[BANNED] {path.relative_to(base_dir)}:{line_number} "
                    f"pattern={label} snippet={snippet}"
                )
    return findings


def _header_locations(base_dir: Path, reference_dir: Path, header: str) -> list[str]:
    locations: list[str] = []
    for path in sorted(reference_dir.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        start = 0
        while True:
            idx = text.find(header, start)
            if idx < 0:
                break
            line = _line_number(text, idx)
            locations.append(f"{path.relative_to(base_dir)}:{line}")
            start = idx + 1
    return locations


def _extract_help_text(raw_output: str) -> str:
    lines = raw_output.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for idx, line in enumerate(lines):
        if line.strip().startswith("usage:"):
            trimmed = [entry.rstrip() for entry in lines[idx:]]
            return "\n".join(trimmed).strip()
    return "\n".join(entry.rstrip() for entry in lines).strip()


def _capture_cli_help(base_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    sections: list[tuple[str, str]] = []
    failures: list[str] = []
    for display, command in HELP_COMMANDS:
        try:
            result = subprocess.run(
                command,
                cwd=base_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            failures.append(f"[CLI] command_failed command={display} error={exc}")
            continue
        combined = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
        cleaned = _extract_help_text(combined)
        sections.append((display, cleaned or "<no output>"))
        if result.returncode != 0:
            failures.append(f"[CLI] nonzero_exit command={display} rc={result.returncode}")
    return sections, failures


def _write_cli_reference(target: Path, sections: Sequence[tuple[str, str]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# CLI Reference",
        "",
        "Generated by `python -m scripts.docs_consistency_check` from live `--help` output.",
        "",
    ]
    for display, payload in sections:
        lines.append(f"## `{display}`")
        lines.append("")
        lines.append("```text")
        lines.append(payload.strip())
        lines.append("```")
        lines.append("")
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_findings(path: Path, findings: Sequence[str], notes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "FAIL" if findings else "PASS"
    lines = [f"DOCS_CONSISTENCY status={status}"]
    for note in notes:
        lines.append(f"[INFO] {note}")
    if findings:
        lines.append("Findings:")
        for finding in findings:
            lines.append(f"- {finding}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_docs_consistency(
    *,
    base_dir: Path | str = BASE_DIR,
    docs_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    cli_reference_path: Path | str | None = None,
    fail_on_missing_docs: bool = True,
) -> tuple[list[str], Path]:
    base = Path(base_dir).resolve()
    docs_root = Path(docs_dir).resolve() if docs_dir else (base / "docs")
    reports_root = Path(reports_dir).resolve() if reports_dir else (base / "reports")
    cli_path = (
        Path(cli_reference_path).resolve()
        if cli_reference_path
        else (docs_root / "reference" / "cli_reference.md")
    )
    findings_path = reports_root / "docs_findings.txt"

    findings: list[str] = []
    notes: list[str] = []

    if not docs_root.exists():
        message = f"[DOCS] missing docs directory: {docs_root}"
        notes.append(message)
        if fail_on_missing_docs:
            findings.append(message)
        _write_findings(findings_path, findings, notes)
        return findings, findings_path

    notes.append(f"docs_root={docs_root}")

    banned_findings = _scan_banned_phrases(base, docs_root)
    findings.extend(banned_findings)
    notes.append(f"banned_pattern_hits={len(banned_findings)}")

    cli_sections, cli_failures = _capture_cli_help(base)
    _write_cli_reference(cli_path, cli_sections)
    notes.append(f"cli_reference={cli_path}")
    findings.extend(cli_failures)

    reference_dir = docs_root / "reference"
    if not reference_dir.exists():
        findings.append(f"[DOCS] missing reference directory: {reference_dir}")
    else:
        header = _canonical_header()
        locations = _header_locations(base, reference_dir, header)
        notes.append(f"canonical_header_occurrences={len(locations)}")
        if len(locations) != 1:
            location_text = ", ".join(locations) if locations else "none"
            findings.append(
                "[HEADER] canonical candidate header must appear exactly once in docs/reference "
                f"(found {len(locations)}: {location_text})"
            )

    _write_findings(findings_path, findings, notes)
    return findings, findings_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check docs consistency and regenerate CLI reference docs")
    parser.add_argument("--base", default=str(BASE_DIR), help="Repository base directory")
    parser.add_argument("--docs-dir", default=None, help="Optional docs directory override")
    parser.add_argument("--reports-dir", default=None, help="Optional reports directory override")
    parser.add_argument(
        "--cli-reference-path",
        default=None,
        help="Optional path for generated cli_reference.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    findings, findings_path = run_docs_consistency(
        base_dir=args.base,
        docs_dir=args.docs_dir,
        reports_dir=args.reports_dir,
        cli_reference_path=args.cli_reference_path,
        fail_on_missing_docs=True,
    )
    print(f"DOCS_FINDINGS path={findings_path}")
    if findings:
        for finding in findings:
            print(f"ERROR {finding}")
        return 1
    print("DOCS_CONSISTENCY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
