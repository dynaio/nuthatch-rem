"""
Minimal GLB parser. Extracts embedded images and mesh metadata.
GLB spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
"""
import json
import struct
from pathlib import Path

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK  = 0x004E4942


def parse_glb(path):
    with open(path, "rb") as f:
        data = f.read()

    magic, version, length = struct.unpack("<4sII", data[:12])
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: magic={magic!r}")
    if version != 2:
        raise ValueError(f"Unsupported glTF version: {version}")

    offset = 12
    gltf = None
    blob = None
    while offset < length:
        chunk_len, chunk_type = struct.unpack("<II", data[offset:offset + 8])
        chunk_data = data[offset + 8:offset + 8 + chunk_len]
        if chunk_type == JSON_CHUNK:
            gltf = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == BIN_CHUNK:
            blob = chunk_data
        offset += 8 + chunk_len

    return gltf, blob


def extract_images(gltf, blob, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = gltf.get("images", [])
    buffer_views = gltf.get("bufferViews", [])

    saved = []
    for i, img in enumerate(images):
        if "bufferView" not in img:
            continue
        bv = buffer_views[img["bufferView"]]
        start = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        img_bytes = blob[start:start + length]

        mime = img.get("mimeType", "image/png")
        ext = "png" if "png" in mime else "jpg"
        out_path = out_dir / f"texture_{i}.{ext}"
        out_path.write_bytes(img_bytes)
        saved.append(out_path)

    return saved


def summarise(gltf):
    print(f"  meshes         : {len(gltf.get('meshes', []))}")
    print(f"  materials      : {len(gltf.get('materials', []))}")
    print(f"  textures       : {len(gltf.get('textures', []))}")
    print(f"  images         : {len(gltf.get('images', []))}")
    print(f"  nodes          : {len(gltf.get('nodes', []))}")
    print(f"  accessors      : {len(gltf.get('accessors', []))}")

    for i, mat in enumerate(gltf.get("materials", [])):
        pbr = mat.get("pbrMetallicRoughness", {})
        base = pbr.get("baseColorTexture")
        print(f"  material[{i}]  : name={mat.get('name')!r} "
              f"baseColorTexture={'yes' if base else 'no'}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "3D-Model/earth.glb"
    print(f"Parsing {path}")
    gltf, blob = parse_glb(path)
    print(f"  binary blob size: {len(blob):,} bytes")
    summarise(gltf)

    imgs = extract_images(gltf, blob, "output/textures")
    print(f"  extracted {len(imgs)} images to output/textures/")
    for p in imgs:
        print(f"    {p}  ({p.stat().st_size:,} bytes)")
