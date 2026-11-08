#!/usr/bin/env python3
"""Create a portable local copy of NVIDIA's TableTrolley_B01 asset."""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath

from pxr import Sdf, Usd


ISAAC_ASSET_ROOT = Path(
    "/mnt/shangbin407/isaacsim_assets/Assets/Isaac/5.0/NVIDIA/Assets/DigitalTwin"
)
SOURCE_DIR = ISAAC_ASSET_ROOT / "Assets/Warehouse/Equipment/Carts/TableTrolley_B"
SOURCE_USD = SOURCE_DIR / "TableTrolley_B01_01.usd"
SOURCE_THUMB = SOURCE_DIR / ".thumbs/256x256/TableTrolley_B01_01.usd.png"

ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
DEST_ROOT = ASSEMBLE_ROOT / "assets-new/object/TableTrolley_B"
DEST_USD = DEST_ROOT / SOURCE_USD.name

TEXTURE_PATTERN = re.compile(r'texture_2d\(\s*"([^"]+)"')


def _materials_relative_path(authored_path: str) -> Path:
    parts = PurePosixPath(authored_path).parts
    try:
        index = parts.index("Materials")
    except ValueError as error:
        raise RuntimeError(f"Unexpected external asset path: {authored_path}") from error
    return Path(*parts[index:])


def _copy_mdl_and_textures(relative_mdl: Path) -> int:
    source_mdl = ISAAC_ASSET_ROOT / relative_mdl
    destination_mdl = DEST_ROOT / relative_mdl
    if not source_mdl.is_file():
        raise FileNotFoundError(source_mdl)
    destination_mdl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mdl, destination_mdl)

    texture_count = 0
    source_text = source_mdl.read_text(encoding="utf-8")
    for authored_texture in sorted(set(TEXTURE_PATTERN.findall(source_text))):
        if not authored_texture:
            continue
        source_texture = (source_mdl.parent / authored_texture).resolve()
        destination_texture = destination_mdl.parent / authored_texture
        if not source_texture.is_file():
            raise FileNotFoundError(source_texture)
        destination_texture.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_texture, destination_texture)
        texture_count += 1
    return texture_count


def migrate() -> None:
    if not SOURCE_USD.is_file():
        raise FileNotFoundError(SOURCE_USD)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_USD, DEST_USD)

    source_stage = Usd.Stage.Open(str(SOURCE_USD), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Failed to open {SOURCE_USD}")

    mdl_paths: set[Path] = set()
    authored_assets: list[tuple[str, str, Path]] = []
    for prim in Usd.PrimRange(source_stage.GetPseudoRoot()):
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            relative_path = _materials_relative_path(value.path)
            mdl_paths.add(relative_path)
            authored_assets.append(
                (str(prim.GetPath()), attribute.GetName(), relative_path)
            )

    texture_count = sum(_copy_mdl_and_textures(path) for path in sorted(mdl_paths))

    destination_stage = Usd.Stage.Open(str(DEST_USD), Usd.Stage.LoadAll)
    if destination_stage is None:
        raise RuntimeError(f"Failed to open copied asset {DEST_USD}")
    for prim_path, attribute_name, relative_path in authored_assets:
        attribute = destination_stage.GetPrimAtPath(prim_path).GetAttribute(
            attribute_name
        )
        if not attribute:
            raise RuntimeError(f"Missing copied attribute {prim_path}.{attribute_name}")
        attribute.Set(Sdf.AssetPath(f"./{relative_path.as_posix()}"))
    destination_stage.GetRootLayer().Save()

    if SOURCE_THUMB.is_file():
        destination_thumb = DEST_ROOT / ".thumbs/256x256" / SOURCE_THUMB.name
        destination_thumb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_THUMB, destination_thumb)

    print(f"Copied {DEST_USD}")
    print(f"MDL files={len(mdl_paths)}, textures={texture_count}")


if __name__ == "__main__":
    migrate()
