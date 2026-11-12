"""Cut a teacher cache for one split out of another (e.g. train from trainval), by image name.
Teachers run once on trainval; the dev-split ablations (data.split=train) reuse those maps.

    python teachers/subset_cache.py --store ws/data/bsds --teacher-root ws/teachers/bsds \
        --from trainval --to train --teachers sauge muge memo ensemble_mean ensemble_confidence
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--from", dest="src", default="trainval")
    p.add_argument("--to", dest="dst", default="train")
    p.add_argument("--teachers", nargs="+", required=True)
    args = p.parse_args()
    want = json.load(open(Path(args.store) / args.dst / "names.json"))
    out = Path(args.teacher_root) / args.dst
    out.mkdir(parents=True, exist_ok=True)
    for t in args.teachers:
        src = Path(args.teacher_root) / args.src
        meta = json.load(open(src / f"{t}.json"))
        index = {n: i for i, n in enumerate(meta["names"])}
        missing = [n for n in want if n not in index]
        if missing:
            raise ValueError(f"{t}: {len(missing)} {args.dst} images missing from {args.src} cache")
        arr = np.load(src / f"{t}.npy", mmap_mode="r")
        np.save(out / f"{t}.npy", np.stack([arr[index[n]] for n in want]))
        json.dump({**meta, "names": want, "subset_of": args.src}, open(out / f"{t}.json", "w"))
        print(f"{t}: {args.src} -> {args.dst} ({len(want)} images)")


if __name__ == "__main__":
    main()
