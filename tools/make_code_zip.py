"""Zip the code for upload as a Kaggle Dataset (no data, no checkpoints, no third_party, no documents).

    python tools/make_code_zip.py            -> dist/pedger-mtkd-code.zip
Upload it on kaggle.com -> Datasets -> New Dataset (private), name it pedger-mtkd-code.
The files are the same allowlist that `python -m pedger_kd.sync push-code` uploads (sync.CODE_INCLUDE).
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT))
    from pedger_kd.sync import code_files, write_stamp
    print("code stamp:", write_stamp(ROOT)["stamp"])   # newer than HF -> the notebooks may push this copy
    out = ROOT / "dist" / "pedger-mtkd-code.zip"
    out.parent.mkdir(exist_ok=True)
    files = code_files(ROOT)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, q in files.items():
            z.write(q, Path("pedger-mtkd") / rel)
    print(f"wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
