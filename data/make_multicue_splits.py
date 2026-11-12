"""MultiCue protocol: 100 scenes, 80 train / 20 test, 3 random splits with fixed seeds.

Input: a list file with every MultiCue image (format of data/prepare_list.py).
The scene id is taken from the image file name with --scene-regex (default: the
leading digits, e.g. 0001_left_0667.png -> 0001). Check the downloaded naming
first and adjust the regex if needed.

    python data/make_multicue_splits.py --list multicue_boundaries_all.lst --out lists/multicue_boundary
-> lists/multicue_boundary/split{0,1,2}_{train,test}.lst
"""
import argparse
import random
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scene-regex", default=r"^(\d+)")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    lines = [l.rstrip("\n") for l in open(args.list) if l.strip()]
    rx = re.compile(args.scene_regex)
    scene_of = {}
    for l in lines:
        m = rx.search(Path(l.split()[0]).name)
        if not m:
            raise ValueError(f"scene regex did not match {l.split()[0]}")
        scene_of[l] = m.group(1)
    scenes = sorted(set(scene_of.values()))
    print(f"{len(lines)} images, {len(scenes)} scenes")
    if len(scenes) <= args.n_train:
        raise ValueError("not enough scenes for the requested split")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(args.seeds):
        rng = random.Random(seed)
        sc = scenes[:]
        rng.shuffle(sc)
        train = set(sc[: args.n_train])
        for name, keep in (("train", lambda s: s in train), ("test", lambda s: s not in train)):
            sel = [l for l in lines if keep(scene_of[l])]
            (out / f"split{i}_{name}.lst").write_text("\n".join(sel) + "\n")
        print(f"split{i} (seed {seed}): {len(train)} train scenes, {len(scenes) - len(train)} test scenes")


if __name__ == "__main__":
    main()
