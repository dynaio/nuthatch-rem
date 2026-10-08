"""
Download Natural Earth land polygons and rasterise them onto an
equirectangular grid (default 4096x2048, same as texture_1).

Requires internet access on first run. Result is cached as
data/ne_land_mask.npy.
"""
import os
import numpy as np
from pathlib import Path


def build_mask(width=4096, height=2048, cache="data/ne_land_mask.npy"):
    if os.path.exists(cache):
        print(f"  loading cached mask from {cache}")
        return np.load(cache)

    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    print("  fetching Natural Earth 1:110m land polygons ...")
    url = ("https://naciscdn.org/naturalearth/110m/physical/"
           "ne_110m_land.zip")
    gdf = gpd.read_file(url)

    print(f"  loaded {len(gdf)} land features")

    transform = from_bounds(-180, -90, 180, 90, width, height)
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mask)
    print(f"  saved mask to {cache}")
    return mask


if __name__ == "__main__":
    mask = build_mask()
    H, W = mask.shape
    land_frac = mask.mean()
    print(f"  shape           : {mask.shape}")
    print(f"  land fraction   : {land_frac:.4f}")
    print(f"  target          : 0.2920")
    print(f"  difference      : {abs(land_frac - 0.292):.4f}")

    # Also save a texture-matched mask if texture is available
    tex_path = Path("output/textures/texture_1.png")
    if tex_path.exists():
        from PIL import Image
        img = Image.open(tex_path)
        tex_w, tex_h = img.size
        if (tex_w, tex_h) != (W, H):
            print(f"  resizing mask to texture size {tex_w}x{tex_h} for overlay")
            import numpy as np
            from PIL import Image as I
            m_img = I.fromarray((mask * 255).astype(np.uint8))
            m_img = m_img.resize((tex_w, tex_h), I.NEAREST)
            mask_tex = np.array(m_img) > 127
            np.save("output/ne_land_mask_tex.npy", mask_tex)
            print("  saved output/ne_land_mask_tex.npy")

    # Preview
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.imshow(mask, cmap="gray")
    plt.title(f"Natural Earth land mask — land fraction = {land_frac:.4f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output/ne_land_mask.png", dpi=120)
    print("  saved preview output/ne_land_mask.png")
