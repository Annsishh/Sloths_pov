"""Runtime guards: ground-truth path refusal in DataLoader, mock-fallback warning in the LLM factory."""

import logging
import subprocess
import sys
import warnings

import pytest

from analytics.loader import DataLoader
from analytics.path_guard import ForbiddenDataPathError
from controller.llm_client import MockLLMClient, create_llm_client


@pytest.mark.parametrize("parts", [
    ("ground_truth",),
    ("scenario", "ground_truth"),
    ("Ground_Truth",),
    ("scenario", "ground_truth", "nested"),
    ("my_ground_truth_copy",),
])
def test_loader_refuses_ground_truth_directories(tmp_path, parts):
    d = tmp_path.joinpath(*parts)
    d.mkdir(parents=True)
    with pytest.raises(ForbiddenDataPathError):
        DataLoader(data_dir=d)


def test_loader_refuses_dotdot_traversal_into_ground_truth(tmp_path):
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "investigator_data").mkdir()
    with pytest.raises(ForbiddenDataPathError):
        DataLoader(data_dir=tmp_path / "investigator_data" / ".." / "ground_truth")


def test_loader_refuses_link_into_ground_truth(tmp_path):
    """A directory link with an innocent name must be resolved and refused (symlink, or junction on Windows)."""
    target = tmp_path / "ground_truth"
    target.mkdir()
    link = tmp_path / "innocent_name"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("symlinks not permitted here")
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if made.returncode != 0:
            pytest.skip("cannot create a directory link on this machine")
    with pytest.raises(ForbiddenDataPathError):
        DataLoader(data_dir=link)


def test_loader_still_accepts_investigator_directories(tmp_path):
    d = tmp_path / "investigator_data"
    d.mkdir()
    assert DataLoader(data_dir=d).data_dir == d


def test_error_is_a_permission_error(tmp_path):
    d = tmp_path / "ground_truth"
    d.mkdir()
    with pytest.raises(PermissionError):
        DataLoader(data_dir=d)


@pytest.fixture
def no_keys(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)


def test_fallback_to_mock_warns_loudly(no_keys, caplog):
    with caplog.at_level(logging.WARNING, logger="controller.llm_client"):
        with pytest.warns(RuntimeWarning, match="NO LLM API KEY"):
            client = create_llm_client()
    assert isinstance(client, MockLLMClient)
    assert any("NOT from a real model" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_explicit_mock_does_not_warn(no_keys, caplog):
    with caplog.at_level(logging.WARNING, logger="controller.llm_client"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            client = create_llm_client("mock")
    assert isinstance(client, MockLLMClient)
    assert not caplog.records


def test_real_client_when_key_present(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client = create_llm_client()
    assert not isinstance(client, MockLLMClient)
