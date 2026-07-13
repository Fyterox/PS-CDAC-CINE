"""
build_manifest_dffd.py — manifest for the REAL DFFD layout.

DFFD ships as  <generator>/{train,validation,test}/  — generator first, split
second. (The generic build_manifest.py expects the opposite nesting, which is why
this dedicated script exists.)

The train/validation/test folders ARE the official identity-disjoint split, so we
read them directly rather than making a random one. That keeps results comparable
to the baseline and avoids the same face leaking across train and test.

Usage:
    python build_manifest_dffd.py --root /content/DFFD --out /content/PS1/manifest.csv
"""
import argparse, os, csv
from collections import Counter

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# folder -> (generator tag, label)   label 0 = real, 1 = fake
MAP = {
    "ffhq":            ("real",     0),
    "faceapp":         ("faceapp",  1),
    "stargan":         ("stargan",  1),
    "pggan_v1":        ("pggan",    1),   # v1 + v2 both aggregate to 'pggan'
    "pggan_v2":        ("pggan",    1),
    "stylegan_ffhq":   ("stylegan", 1),   # both stylegan variants -> 'stylegan'
    "stylegan_celeba": ("stylegan", 1),
}
SPLITMAP = {"train": "train", "validation": "val", "test": "test"}
GENS = ["real", "faceapp", "stargan", "pggan", "stylegan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/DFFD")
    ap.add_argument("--out", default="/content/PS1/manifest.csv")
    args = ap.parse_args()

    rows = []
    for folder, (gen, label) in MAP.items():
        for split_dir, split in SPLITMAP.items():
            d = os.path.join(args.root, folder, split_dir)
            if not os.path.isdir(d):
                print(f"[skip] missing {d}")
                continue
            # NOTE: *_mask folders (faceapp/train_mask etc.) are manipulation
            # masks, not classifier inputs — they are never listed here.
            for fn in os.listdir(d):
                if fn.lower().endswith(EXT):
                    rows.append((os.path.join(d, fn), label, gen, split))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "split"])
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {args.out}\n")
    c = Counter((r[3], r[2]) for r in rows)
    print(f"{'split':<7}" + "".join(f"{g:>10}" for g in GENS))
    for s in ["train", "val", "test"]:
        print(f"{s:<7}" + "".join(f"{c.get((s,g),0):>10}" for g in GENS))
    print("\nSanity: every column in the 'test' row must be non-zero, or that "
          "generator's recall can't be computed.")


if __name__ == "__main__":
    main()
