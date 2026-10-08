"""
Build a binary land/sea mask from an equirectangular Earth texture.

Classification rule (user's rule): blue-dominant pixels are sea,
everything else is land. This is crude but effective for standard
Earth albedo textures where oceans are distinctly blue.
"""
import numpy as np
from PIL import Image


def load_texture(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def classify_land(arr, blue_threshold=60):
    """
    Return a boolean array, True = land.
    A pixel is sea if B is the dominant channel AND B > threshold.
    """
    arr = arr.astype(np.int16)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_sea = (B > R + 10) & (B > G + 10) & (B > blue_threshold)
    return ~is_sea


def stats(mask):
    total = mask.size
    land = int(mask.sum())
    return {"total": total, "land": land, "sea": total - land,
            "land_fraction": land / total}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        tex_path = sys.argv[1]
    else:
        candidates = sorted(Path("output/textures").glob("texture_*.png")) \
                   + sorted(Path("output/textures").glob("texture_*.jpg"))
        if not candidates:
            print("No texture found. Run io/glb_inspect.py first.")
            raise SystemExit(1)
        tex_path = str(candidates[0])

    print(f"Loading {tex_path}")
    arr = load_texture(tex_path)
    print(f"  shape: {arr.shape}")

    mask = classify_land(arr)
    s = stats(mask)
    print(f"  land pixels: {s['land']:>10,}")
    print(f"  sea pixels : {s['sea']:>10,}")
    print(f"  land fraction (image): {s['land_fraction']:.4f}")

    # Save mask for later
    from pathlib import Path
    out = Path("data/ne_land_mask.npy")
    np.save(out, mask)
    print(f"  saved mask to {out}")

    # Visualise
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(arr); axes[0].set_title("Original texture")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Land (white) / Sea (black)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig("output/land_mask_preview.png", dpi=120)
    print("  saved preview to output/land_mask_preview.png")
