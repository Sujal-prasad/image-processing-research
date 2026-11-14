"""Preflight checks: missing secret, read-only token, wrong account, token that cannot write, all good.
Uses a fake Hugging Face API, so no network or real token is needed."""
import types

import pytest

from pedger_kd import preflight
from pedger_kd.jobs import make_worker_id


class FakeApi:
    def __init__(self, role="write", user="owner", repo_error=None, upload_error=None):
        self.role, self.user, self.repo_error, self.upload_error = role, user, repo_error, upload_error
        self.uploads, self.created = [], []

    def whoami(self):
        return {"name": self.user, "orgs": [], "auth": {"accessToken": {"role": self.role}}}

    def repo_info(self, repo, repo_type):
        if self.repo_error:
            raise self.repo_error

    def create_repo(self, repo, **kw):
        self.created.append(repo)

    def upload_file(self, **kw):
        if self.upload_error:
            raise self.upload_error
        self.uploads.append(kw["path_in_repo"])


def RepositoryNotFoundError():
    return type("RepositoryNotFoundError", (Exception,), {})()


@pytest.fixture
def fake(monkeypatch):
    import huggingface_hub

    def use(api):
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        return api
    return use


def test_missing_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ok, _, msg = preflight.hf_check("hf:owner/store")
    assert not ok and "Secrets" in msg


def test_read_only_token_is_rejected(fake):
    api = fake(FakeApi(role="read"))
    ok, user, msg = preflight.hf_check("hf:owner/store")
    assert not ok and "READ-only" in msg and not api.uploads


def test_token_from_another_account(fake):
    fake(FakeApi(user="teammate", repo_error=RepositoryNotFoundError()))
    ok, user, msg = preflight.hf_check("hf:owner/store")
    assert not ok and user == "teammate" and "'owner'" in msg


def test_token_that_reads_but_cannot_write(fake):
    """e.g. a fine-grained token: not labelled 'read', but without write permission on this repo."""
    fake(FakeApi(role="fineGrained", upload_error=PermissionError("403")))
    ok, _, msg = preflight.hf_check("hf:owner/store")
    assert not ok and "NOT WRITE" in msg


def test_write_token_passes_and_proves_write(fake):
    api = fake(FakeApi(role="write"))
    ok, _, msg = preflight.hf_check("hf:owner/store")
    assert ok and "read + write" in msg and api.uploads == [preflight.WRITE_PROBE]


def test_missing_repo_is_created_by_its_owner(fake):
    api = fake(FakeApi(role="write", repo_error=RepositoryNotFoundError()))
    ok, _, msg = preflight.hf_check("hf:owner/store")
    assert ok and api.created == ["owner/store"] and "created" in msg


def test_write_test_can_be_skipped(fake):
    api = fake(FakeApi(role="write"))
    ok, _, msg = preflight.hf_check("hf:owner/store", write_test=False)
    assert ok and not api.uploads


def test_worker_ids_unique_even_with_same_label():
    """Two Kaggle accounts with the same ACCOUNT label must never share a worker id (the lock compares ids)."""
    ids = {make_worker_id("account-1", "gpu0") for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("account-1@") for i in ids)
    assert make_worker_id("", "gpu0").startswith("worker@")
