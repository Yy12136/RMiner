#!/usr/bin/env python3
"""Build the task-specific 45 mm diameter, 1.45 m long roof bolt."""

from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
OBJECT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object"
SOURCE = OBJECT_ROOT / "M20_Bolt_Chain_D45_L1500.usda"
OUTPUT = OBJECT_ROOT / "M20_Bolt_Chain_D45_L1450.usda"

SOURCE_LENGTH = 1.50
TARGET_LENGTH = 1.45
AXIAL_SCALE = TARGET_LENGTH / SOURCE_LENGTH


def build() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_stage = Usd.Stage.Open(str(SOURCE), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Failed to open {SOURCE}")
    if not source_stage.GetRootLayer().Export(str(OUTPUT)):
        raise RuntimeError(f"Failed to export {OUTPUT}")

    stage = Usd.Stage.Open(str(OUTPUT), Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError("Roof-bolt asset has no default prim")

    collider = UsdGeom.Cylinder(
        stage.GetPrimAtPath(root.GetPath().AppendChild("ContinuousCollision"))
    )
    if not collider:
        raise RuntimeError("Continuous roof-bolt collider is missing")
    collider.GetHeightAttr().Set(TARGET_LENGTH)

    segment_count = 0
    for child in root.GetChildren():
        if not child.GetName().startswith("Segment_"):
            continue
        xform = UsdGeom.Xformable(child)
        translate_op = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        scale_op = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if translate_op is None or scale_op is None:
            raise RuntimeError(f"Segment transform is incomplete: {child.GetPath()}")
        translation = translate_op.Get()
        scale = scale_op.Get()
        translate_op.Set((translation[0], translation[1], translation[2] * AXIAL_SCALE))
        scale_op.Set((scale[0], scale[1], scale[2] * AXIAL_SCALE))
        segment_count += 1

    if segment_count != 11:
        raise RuntimeError(f"Expected 11 visual segments, found {segment_count}")
    root.CreateAttribute("task:length", Sdf.ValueTypeNames.Double).Set(TARGET_LENGTH)
    root.CreateAttribute("task:diameter", Sdf.ValueTypeNames.Double).Set(0.045)
    stage.GetRootLayer().documentation = (
        "Eleven localized M20 bolt assets joined along local Z. Overall visual "
        "envelope is approximately 45 mm diameter by 1450 mm long."
    )
    stage.GetRootLayer().Save()
    print(f"Built {OUTPUT}")
    print(
        f"target diameter=0.045 m, length={TARGET_LENGTH:.3f} m, "
        f"axial scale={AXIAL_SCALE:.9f}, segments={segment_count}"
    )


if __name__ == "__main__":
    build()
