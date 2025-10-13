from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.env import load_env


pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    keys = [
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_BASE_URL",
        "ALPACA_API_BASE_URL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def _write_env(tmp_path: Path, contents: str) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(contents, encoding="utf-8")


def test_load_env_strips_crlf_and_spaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_env(
        tmp_path,
        "APCA_API_KEY_ID= key123   \r\n"
        "APCA_API_SECRET_KEY= secret456\r\n",
    )

    summary = load_env()

    assert os.environ["APCA_API_KEY_ID"] == "key123"
    assert os.environ["APCA_API_SECRET_KEY"] == "secret456"
    assert summary["APCA_API_KEY_ID"] == {"present": True, "len": 6}
    assert summary["APCA_API_SECRET_KEY"] == {"present": True, "len": 9}


def test_load_env_handles_quoted_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_env(
        tmp_path,
        'APCA_API_KEY_ID="quoted-key"\nAPCA_API_SECRET_KEY="quoted-secret"\n',
    )

    summary = load_env()

    assert os.environ["APCA_API_KEY_ID"] == "quoted-key"
    assert os.environ["APCA_API_SECRET_KEY"] == "quoted-secret"
    assert summary["APCA_API_KEY_ID"]["len"] == len("quoted-key")
    assert summary["APCA_API_SECRET_KEY"]["len"] == len("quoted-secret")


def test_load_env_trims_v2_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_env(
        tmp_path,
        "APCA_API_KEY_ID=abc\n"
        "APCA_API_SECRET_KEY=def\n"
        "APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2/\n",
    )

    summary = load_env()

    assert os.environ["APCA_API_BASE_URL"] == "https://paper-api.alpaca.markets"
    assert summary["APCA_API_BASE_URL"] == {
        "present": True,
        "len": len("https://paper-api.alpaca.markets"),
    }
