#!/usr/bin/env python3
"""Build a portable, tabletop-ready rod-and-rope USD.

Run this with Isaac Sim's Python interpreter.  The source assembly produced by
Isaac Sim contains a session-only ``metrics:`` sublayer and references two
neighbouring assets.  This script copies those dependencies, removes the
session layer, and authors a stable horizontal rest pose without changing the
26-link rope topology.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_ROOT = ASSEMBLE_ROOT / "assets-new" / "samples"
OBJECT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object"
SOURCE_ASSEMBLY = SAMPLES_ROOT / "chaganshengzi001.usd"
OUTPUT_ASSEMBLY = OBJECT_ROOT / "chaganshengzi_tabletop.usd"

ROD_RADIUS_CLEARANCE = 0.0125
ROD_TOP_Z = 0.25029654499407916
ROPE_SCALE = 0.013
LINK_00_LOCAL_X = -19.101123595505616


def _copy_dependencies() -> None:
    OBJECT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("chagan.usd", "rope_40cm.usda"):
        source = SAMPLES_ROOT / name
        destination = OBJECT_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def _strip_session_metadata(layer: Sdf.Layer) -> None:
    layer.subLayerPaths = [
        path for path in layer.subLayerPaths if not path.startswith("metrics:")
    ]
    custom_data = dict(layer.customLayerData)
    custom_data.pop("metricsAssembler", None)
    omni_layer = dict(custom_data.get("omni_layer", {}))
    omni_layer["authoring_layer"] = f"./{OUTPUT_ASSEMBLY.name}"
    custom_data["omni_layer"] = omni_layer
    layer.customLayerData = custom_data
    layer.Save()


def _set_tabletop_rest_pose(stage: Usd.Stage) -> None:
    # Lay the rod's authored +Z axis along +X.  A small +Z translation keeps
    # both the rod and the thinner rope capsules clear of the tabletop.
    rod_prim = stage.GetPrimAtPath("/World/SmoothConnector")
    if not rod_prim:
        raise RuntimeError("Missing /World/SmoothConnector")
    rod_xform = UsdGeom.Xformable(rod_prim)
    rod_xform.ClearXformOpOrder()
    rod_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ROD_RADIUS_CLEARANCE))
    rod_xform.AddOrientOp().Set(
        Gf.Quatf(
            math.cos(math.pi / 4.0),
            Gf.Vec3f(0.0, math.sin(math.pi / 4.0), 0.0),
        )
    )

    # Keep all rope links in their original straight +X arrangement.  Move
    # Link_00 to the rotated rod tip so the full assembly lies in the XY plane.
    rope_prim = stage.GetPrimAtPath("/World/rope_40cm")
    if not rope_prim:
        raise RuntimeError("Missing /World/rope_40cm")
    rope_xform = UsdGeom.Xformable(rope_prim)
    translate_op = next(
        op
        for op in rope_xform.GetOrderedXformOps()
        if op.GetOpName() == "xformOp:translate"
    )
    rope_root_x = ROD_TOP_Z - LINK_00_LOCAL_X * ROPE_SCALE
    translate_op.Set(Gf.Vec3d(rope_root_x, 0.0, ROD_RADIUS_CLEARANCE))

    fixed_prim = stage.GetPrimAtPath(
        "/World/rope_40cm/Links/Link_00/FixedJoint"
    )
    fixed_joint = UsdPhysics.FixedJoint(fixed_prim)
    if not fixed_joint:
        raise RuntimeError("Missing rod-to-rope FixedJoint")
    fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, ROD_TOP_Z))
    fixed_joint.GetLocalRot0Attr().Set(
        Gf.Quatf(
            math.cos(math.pi / 4.0),
            Gf.Vec3f(0.0, -math.sin(math.pi / 4.0), 0.0),
        )
    )
    fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.GetLocalRot1Attr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )

    # The simulator task owns the global physics scene.  A nested scene inside
    # a reusable object asset can otherwise change gravity/solver settings.
    stage.RemovePrim("/World/PhysicsScene")
    stage.GetRootLayer().Save()


def _compact_output(stage: Usd.Stage) -> None:
    """Rewrite the crate so removed session-layer tokens are not retained."""

    temporary = OUTPUT_ASSEMBLY.with_name(
        f".{OUTPUT_ASSEMBLY.stem}.tmp{OUTPUT_ASSEMBLY.suffix}"
    )
    if temporary.exists():
        temporary.unlink()
    if not stage.GetRootLayer().Export(str(temporary)):
        raise RuntimeError(f"Failed to export compact USD: {temporary}")
    temporary.replace(OUTPUT_ASSEMBLY)


def main() -> None:
    if not SOURCE_ASSEMBLY.is_file():
        raise FileNotFoundError(SOURCE_ASSEMBLY)
    _copy_dependencies()
    shutil.copy2(SOURCE_ASSEMBLY, OUTPUT_ASSEMBLY)

    layer = Sdf.Layer.FindOrOpen(str(OUTPUT_ASSEMBLY))
    if layer is None:
        raise RuntimeError(f"Failed to open {OUTPUT_ASSEMBLY}")
    _strip_session_metadata(layer)

    stage = Usd.Stage.Open(str(OUTPUT_ASSEMBLY), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to compose {OUTPUT_ASSEMBLY}")
    _set_tabletop_rest_pose(stage)
    _compact_output(stage)
    print(f"Built {OUTPUT_ASSEMBLY}")


if __name__ == "__main__":
    main()
