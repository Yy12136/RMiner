#!/usr/bin/env python3
"""Load, simulate briefly, and render the tracked Galbot hybrid USD."""

import os
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 960,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 2,
    }
)

import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, UsdGeom, UsdLux


HERE = Path(__file__).resolve().parent
USD = HERE / "Tracked_Galbot_UpperBody_SolidBoom.usda"
STATIC_ONLY = os.environ.get("GALBOT_STATIC_PREVIEW", "0") == "1"
PREVIEW = HERE / (
    "Tracked_Galbot_UpperBody_SolidBoom.initial.png"
    if STATIC_ONLY
    else "Tracked_Galbot_UpperBody_SolidBoom.png"
)


def wait_for_stage() -> None:
    for _ in range(600):
        APP.update()
        if not is_stage_loading():
            for _ in range(24):
                APP.update()
            return
    raise TimeoutError("Hybrid USD stage did not finish loading")


def main() -> None:
    if not omni.usd.get_context().open_stage(str(USD)):
        raise RuntimeError(f"Could not open {USD}")
    wait_for_stage()
    stage = omni.usd.get_context().get_stage()

    dome = UsdLux.DomeLight.Define(stage, "/Assembly/ValidationLights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.86, 0.96))
    key = UsdLux.DistantLight.Define(stage, "/Assembly/ValidationLights/Key")
    key.CreateIntensityAttr(2600.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(42.0, -35.0, -30.0))

    if not STATIC_ONLY:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(90):
            APP.update()
        timeline.pause()

    bbox = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    ).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedBox()
    print(f"bounds_after_sim={bbox.GetMin()}..{bbox.GetMax()}")

    viewport = get_active_viewport()
    viewport.set_texture_resolution((1280, 960))
    set_camera_view(
        eye=[2.8, -3.7, 2.45],
        target=[-0.38, 0.0, 0.92],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(72):
        APP.update()
    if PREVIEW.exists():
        PREVIEW.unlink()
    capture_viewport_to_file(viewport, str(PREVIEW), is_hdr=False)
    for _ in range(300):
        APP.update()
        if PREVIEW.exists() and PREVIEW.stat().st_size > 0:
            print(f"preview={PREVIEW}")
            return
    raise TimeoutError(f"Could not capture {PREVIEW}")


try:
    main()
finally:
    APP.close(wait_for_replicator=False, skip_cleanup=True)
