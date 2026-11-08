#!/usr/bin/env python3
"""Build a G1 + BrainCo Revo2 composed USD for Isaac Sim.

Run with Isaac Sim's ``python.sh``.  The generated layer keeps upstream Unitree
assets immutable: it references the G1 base model, disables its Inspire hands,
references the official BrainCo Revo2 USDs, removes their standalone
articulation-root marker, and fixes each Revo2 base to the matching G1 wrist.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdPhysics


HERE = Path(__file__).resolve().parent
ASSEMBLE = HERE.parent.parent
SIM_ROOT = ASSEMBLE / "third_party/unitree_sim_isaaclab"
G1_USD = SIM_ROOT / "assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd"
REVO_DIR = HERE / "assets/brainco-description/revo2_system/usd"
OUTPUT = HERE / "assets/g1_29dof_brainco_revo2/g1_29dof_with_brainco_revo2.usda"


def disable_existing_inspire_hand(stage: Usd.Stage, side: str) -> None:
    robot = Sdf.Path("/World/G1/g1_29dof_with_hand_rev_1_0")
    for suffix in (
        f"{side}_hand_base_link",
        f"joints/{side}_hand_palm_joint",
    ):
        stage.OverridePrim(robot.AppendPath(suffix)).SetActive(False)
    letter = "L" if side == "left" else "R"
    for finger in ("index", "middle", "pinky", "ring", "thumb"):
        stage.OverridePrim(robot.AppendPath(f"joints/{letter}_{finger}_proximal_joint")).SetActive(False)
        stage.OverridePrim(robot.AppendPath(f"joints/{letter}_{finger}_intermediate_joint")).SetActive(False)
    stage.OverridePrim(robot.AppendPath(f"joints/{letter}_thumb_proximal_yaw_joint")).SetActive(False)
    stage.OverridePrim(robot.AppendPath(f"joints/{letter}_thumb_distal_joint")).SetActive(False)


def mount_revo(stage: Usd.Stage, side: str) -> None:
    hand_path = Sdf.Path(f"/World/Revo2_{side.title()}")
    hand = stage.DefinePrim(hand_path, "Xform")
    hand.GetReferences().AddReference(str(REVO_DIR / f"revo2_{side}.usd"), "/revo2_single")
    # The Revo2 hand is normally standalone.  It becomes part of G1 through
    # the fixed wrist joint below, so do not retain an independent articulation root.
    stage.OverridePrim(hand_path.AppendPath("root_joint")).RemoveAPI(UsdPhysics.ArticulationRootAPI)

    joint = UsdPhysics.FixedJoint.Define(stage, f"/World/Joints/{side}_revo2_mount")
    joint.GetBody0Rel().SetTargets([Sdf.Path(f"/World/G1/g1_29dof_with_hand_rev_1_0/{side}_wrist_yaw_link")])
    joint.GetBody1Rel().SetTargets([hand_path.AppendPath(f"{side}_hand_base_link")])
    if side == "left":
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0415, 0.003, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.7071055, Gf.Vec3f(0.0, 0.0, 0.7071081)))
    else:
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0415, -0.003, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(-0.000002559715, Gf.Vec3f(-0.7071081, 0.7071055, -0.0000025597242)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))


def main() -> None:
    if not G1_USD.is_file() or not (REVO_DIR / "revo2_left.usd").is_file():
        raise FileNotFoundError("Required G1 or Revo2 asset is missing")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    stage.SetMetadata("upAxis", "Z")
    world = UsdGeom.Xform.Define(stage, "/World")
    del world
    robot = stage.DefinePrim("/World/G1", "Xform")
    robot.GetReferences().AddReference(str(G1_USD), "/g1_29dof_with_hand_rev_1_0")
    for side in ("left", "right"):
        disable_existing_inspire_hand(stage, side)
        mount_revo(stage, side)
    stage.GetRootLayer().Save()
    print(OUTPUT)


if __name__ == "__main__":
    from pxr import UsdGeom
    main()
