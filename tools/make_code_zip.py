"""Zip the code for upload as a Kaggle Dataset (no data, no checkpoints, no third_party).

    python tools/make_code_zip.py            -> dist/pedger-mtkd-code.zip
Upload it on kaggle.com -> Datasets -> New Dataset (private), name it pedger-mtkd-code.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pedger_kd", "data", "teachers", "eval", "bench", "results", "configs", "queues", "notebooks",
           "tests", "tools", "README.md", "requirements.txt", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pt", ".pth", ".npy", ".tmp"}


def main():
    out = ROOT / "dist" / "pedger-mtkd-code.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            paths = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for q in paths:
                rel = q.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or q.suffix in SKIP_SUFFIX:
                    continue
                z.write(q, Path("pedger-mtkd") / rel)
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
