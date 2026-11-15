"""Code uploads never replace newer code with an older copy (a stale notebook session did exactly that)."""
import json
import shutil

from pedger_kd.sync import STAMP_FILE, LocalRemote, push_code, read_stamp


def make_repo(path, content, stamp=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "requirements.txt").write_text(content)
    if stamp:
        (path / STAMP_FILE).write_text(json.dumps({"stamp": stamp, "git": "x"}))
    return path


def test_newer_copy_is_pushed_older_is_refused(tmp_path):
    remote = LocalRemote(tmp_path / "remote")
    new = make_repo(tmp_path / "dev", "fixed", stamp="2026-11-15T10:00:00+00:00")
    old = make_repo(tmp_path / "stale_session", "broken", stamp="2026-11-14T09:00:00+00:00")
    assert push_code(remote, repo_root=new)
    assert (remote.root / "code/requirements.txt").read_text() == "fixed"
    assert not push_code(remote, repo_root=old)                   # older: refused
    assert not push_code(remote, repo_root=new)                   # same version: nothing to do
    assert (remote.root / "code/requirements.txt").read_text() == "fixed"


def test_unstamped_copy_is_refused_unless_forced(tmp_path):
    remote = LocalRemote(tmp_path / "remote")
    unstamped = make_repo(tmp_path / "old_download", "broken")
    assert not push_code(remote, repo_root=unstamped)
    assert not (remote.root / "code/requirements.txt").exists()
    assert push_code(remote, repo_root=unstamped, force=True)


def test_only_code_is_uploaded_and_stray_files_are_removed(tmp_path):
    remote = LocalRemote(tmp_path / "remote")
    stray = remote.root / "code" / "docs" / "Proposal.docx"            # left by an older upload
    stray.parent.mkdir(parents=True)
    stray.write_text("private")
    dev = make_repo(tmp_path / "dev", "v1", stamp="2026-11-15T10:00:00+00:00")
    (dev / "docs").mkdir()
    (dev / "docs" / "Research_Report.docx").write_text("private")
    (dev / "pedger_kd").mkdir()
    (dev / "pedger_kd" / "x.py").write_text("print(1)")
    (dev / "pedger_kd" / "~$lock.docx").write_text("lock")
    (dev / "notes.txt").write_text("private")
    assert push_code(remote, repo_root=dev)
    assert sorted(remote.list("code")) == ["code/CODE_STAMP.json", "code/pedger_kd/x.py", "code/requirements.txt"]


def test_stamp_marks_the_editing_machine_copy_as_newest(tmp_path):
    remote = LocalRemote(tmp_path / "remote")
    dev = make_repo(tmp_path / "dev", "v1", stamp="2020-01-01T00:00:00+00:00")
    assert push_code(remote, repo_root=dev)
    (dev / "requirements.txt").write_text("v2")
    assert push_code(remote, repo_root=dev, stamp=True)           # fresh stamp -> newer -> pushed
    assert (remote.root / "code/requirements.txt").read_text() == "v2"
    assert read_stamp(dev)["stamp"] > "2020-01-01T00:00:00+00:00"
    # the notebook round trip: a session downloads the code, then tries to push it back -> refused
    session = tmp_path / "session"
    shutil.copytree(remote.root / "code", session)
    assert not push_code(remote, repo_root=session)
