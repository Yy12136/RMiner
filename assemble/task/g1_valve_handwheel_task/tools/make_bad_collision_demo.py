#!/usr/bin/env python3
"""Create demo assets whose collision geometry is deliberately wrong.

Both variants keep the original SAM3D visual mesh completely unchanged. The
hand-authored, open compound collision is replaced by a single naive
convex-hull collision over the whole mesh. Because a convex hull fills every
concave feature, it seals the task-relevant openings:

  * three-way ball valve  -> the three threaded pipe ports are sealed
  * blue valve handwheel  -> the central stem bore and the three spoke
                             windows are sealed

Loading a bad variant into the g1_valve_handwheel_task makes the intended
interaction fail (opening access / rotation about the valve stem), which is
the failure mode the construction workflow is meant to detect and repair.

Run with the Isaac Sim interpreter:

  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV "$ISAACSIM_ROOT/python.sh" make_bad_collision_demo.py  # ISAACSIM_ROOT defaults to ~/isaacsim
"""

from __future__ import annotations
import os

from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics, UsdShade

TASK_DIR = Path(
    os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[3])) + "/assets-new/object/valve_handwheel_task"
)
OUT_DIR = TASK_DIR / "collision_demo"

ASSETS = [
    {
        "source": TASK_DIR / "three_way_ball_valve_task.usd",
        "root": "/ThreeWayBallValve",
        "variants": [
            {
                "mode": "convexhull",
                "suffix": "bad_collision_convexhull",
                "defect": "convex_hull_seals_three_pipe_ports",
                "interaction": "opening access / pipe insertion into the ports",
            },
            {
                "mode": "missing",
                "suffix": "bad_collision_missing",
                "defect": "collision_missing_on_valve_body",
                "interaction": "any contact with the valve body",
            },
        ],
    },
    {
        "source": TASK_DIR / "blue_valve_handwheel_task.usd",
        "root": "/ValveHandwheel",
        "variants": [
            {
                "mode": "convexhull",
                "suffix": "bad_collision_convexhull",
                "defect": "convex_hull_seals_stem_bore_and_spoke_windows",
                "interaction": "rotation about the valve stem / grasping the spokes",
            },
            {
                "mode": "missing",
                "suffix": "bad_collision_missing",
                "defect": "collision_missing_on_handwheel",
                "interaction": "grasping and rotating the handwheel",
            },
        ],
    },
]


def _find_physics_material(stage: Usd.Stage, root: str):
    scope = stage.GetPrimAtPath(f"{root}/PhysicsMaterials")
    if scope:
        for prim in scope.GetChildren():
            material = UsdShade.Material(prim)
            if material:
                return material
    material = UsdShade.Material(
        stage.GetPrimAtPath(f"{root}/Looks/PhysicsMaterial")
    )
    return material if material else None


def _build_variant(spec: dict, variant: dict) -> None:
    stage = Usd.Stage.Open(str(spec["source"]))
    if stage is None:
        raise RuntimeError(f"Failed to open {spec['source']}")

    root = spec["root"]
    collision_path = Sdf.Path(f"{root}/Collision")
    if stage.GetPrimAtPath(collision_path):
        stage.RemovePrim(collision_path)

    visual = stage.GetPrimAtPath(f"{root}/Visual")
    if not visual:
        raise RuntimeError(f"Missing visual mesh {root}/Visual")

    mode = variant["mode"]
    if mode == "convexhull":
        UsdPhysics.CollisionAPI.Apply(visual).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(visual).CreateApproximationAttr(
            "convexHull"
        )
        material = _find_physics_material(stage, root)
        if material:
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(
                material, materialPurpose="physics"
            )
    elif mode == "missing":
        pass
    else:
        raise ValueError(f"Unknown mode {mode}")

    root_prim = stage.GetPrimAtPath(root)
    root_prim.CreateAttribute(
        "demo:collisionDefect", Sdf.ValueTypeNames.String
    ).Set(variant["defect"])
    root_prim.CreateAttribute(
        "demo:blockedInteraction", Sdf.ValueTypeNames.String
    ).Set(variant["interaction"])
    root_prim.CreateAttribute(
        "demo:note", Sdf.ValueTypeNames.String
    ).Set(
        "Intentional collision defect for demonstration. The visual mesh is "
        "unchanged; only the collision geometry is wrong."
    )

    output = OUT_DIR / f"{spec['source'].stem.replace('_task', '')}_task_{variant['suffix']}.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    print(f"wrote {output}")
    print(f"    defect: {variant['defect']}")
    print(f"    blocked interaction: {variant['interaction']}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in ASSETS:
        for variant in spec["variants"]:
            _build_variant(spec, variant)


if __name__ == "__main__":
    main()
