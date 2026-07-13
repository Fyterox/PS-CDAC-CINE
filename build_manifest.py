"""
build_manifest.py — generic manifest builder.

Writes a manifest.csv with columns: path,label,generator,split

For the real DFFD layout (<generator>/<split>), use build_manifest_dffd.py instead.
This script handles two other common layouts:

(A) --mode presplit    (reuse an existing split — preferred when one exists)
    root/
      train/real/*.png            train/faceapp/*.png  train/stargan/*.png ...
      val/real/*.png              val/faceapp/*.png    ...
      test/real/*.png             test/faceapp/*.png   ...
    Any subfolder named 'real' -> label 0; everything else -> label 1, with the
    folder name kept as the generator tag (use --map to remap odd folder names).

(B) --mode flat        (build a fresh stratified split from class folders)
    root/
      real/*.png  faceapp/*.png  stargan/*.png  pggan/*.png  stylegan/*.png
    Produces a stratified train/val/test split (by generator) using --val-frac
    and --test-frac.

    Caution: a random per-image split can put the same identity in both train and
    test, which inflates recall. Prefer an official identity-disjoint split when
    the dataset provides one.

Examples
--------
  python build_manifest.py --mode presplit --root /data/DFFD --out manifest.csv
  python build_manifest.py --mode flat --root /data/DFFD --out manifest.csv \
         --val-frac 0.1 --test-frac 0.1
"""
import argparse, os, glob, random
import pandas as pd
from common import GENERATORS

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(folder, "**", "*" + ext), recursive=True)
    return sorted(files)


def gen_tag(folder_name, remap):
    name = folder_name.lower()
    name = remap.get(name, name)
    return name


def build_presplit(root, remap):
    rows = []
    for split in ("train", "val", "test"):
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            print(f"[warn] missing split folder: {sdir}")
            continue
        for gdir in sorted(os.listdir(sdir)):
            full = os.path.join(sdir, gdir)
            if not os.path.isdir(full):
                continue
            g = gen_tag(gdir, remap)
            label = 0 if g == "real" else 1
            for p in list_images(full):
                rows.append((p, label, g, split))
    return rows


def build_flat(root, remap, val_frac, test_frac, seed):
    random.seed(seed)
    rows = []
    for gdir in sorted(os.listdir(root)):
        full = os.path.join(root, gdir)
        if not os.path.isdir(full):
            continue
        g = gen_tag(gdir, remap)
        label = 0 if g == "real" else 1
        imgs = list_images(full)
        random.shuffle(imgs)
        n = len(imgs)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        for i, p in enumerate(imgs):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            rows.append((p, label, g, split))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["presplit", "flat"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--map", nargs="*", default=[],
                    help="remap folder names, e.g. --map ffhq=real gan=stylegan")
    args = ap.parse_args()

    remap = {}
    for kv in args.map:
        k, v = kv.split("=")
        remap[k.lower()] = v.lower()

    if args.mode == "presplit":
        rows = build_presplit(args.root, remap)
    else:
        rows = build_flat(args.root, remap, args.val_frac, args.test_frac, args.seed)

    df = pd.DataFrame(rows, columns=["path", "label", "generator", "split"])
    # Sanity: warn about unknown generator tags.
    unknown = set(df["generator"].unique()) - set(GENERATORS)
    if unknown:
        print(f"[warn] generator tags not in vocab {GENERATORS}: {unknown}. "
              f"Use --map to fix, else per-generator report will just show them raw.")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
    print(df.groupby(["split", "generator"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
