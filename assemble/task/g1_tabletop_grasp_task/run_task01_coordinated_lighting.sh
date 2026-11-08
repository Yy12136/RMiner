#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if pgrep -af "[s]im_main.py|[s]upervisor.py|[i]saac-sim.sh|[k]it/kit" >/dev/null; then
  echo "Another Isaac/Task01 simulator process is already running." >&2
  exit 75
fi

export G1_DUST_MODE=off
export G1_FALLING_DEBRIS=0
export G1_HEADLAMP=on
export G1_HEADLAMP_INTENSITY=2500000
export G1_HEADLAMP_AMBIENT_INTENSITY=65
export G1_HEADLAMP_NUREC_EMISSIVE_SCALE=0.114
export G1_HEADLAMP_CAMERA_EXPOSURE=0
export G1_AUX_LIGHTS=on
export G1_AUX_LIGHT_PROFILE=distinct
export G1_AUX_LIGHT_SEED=260920
export G1_AUX_LIGHT_EPISODE=0
export G1_AUX_LIGHT_WORLD_INTENSITIES=20000000,0,18000000
export G1_AUX_LIGHT_CONE_ANGLE_DEG=30
export G1_AUX_LIGHT_CONE_SOFTNESS=0.60
export G1_AUX_LIGHT_FOCUS=0.30
export G1_AUX_LIGHT_RADIUS_M=0.12
# The rod-and-rope task object is centered near (-2.86, 3.36, 0.85) m.
# Aim the two lights to opposite nearby sides rather than stacking all beams
# on that one point.
export G1_AUX_LIGHT_WORLD_TARGETS='-3.28,3.12,0.84;-2.86,3.36,0.84;-2.45,3.60,0.84'
export G1_HEADLAMP_CONE_ANGLE_DEG=68
export G1_HEADLAMP_CONE_SOFTNESS=0.82
export G1_HEADLAMP_FOCUS=0.20
export G1_AUX_LIGHT_COLOR_TEMP_K=3800
export G1_LIGHT_FLICKER=coordinated
export G1_LIGHT_FLICKER_SEED=260920
export G1_LIGHT_FLICKER_DISABLED_ROLES=aux1
export G1_LIGHT_FLICKER_LIGHT_PATHS=/World/TrackedBase/G1UpperBody/d435_link/Task01Headlamp,/World/Task01AuxLights/AuxLight0,/World/Task01AuxLights/AuxLight1,/World/Task01AuxLights/AuxLight2
export G1_LIGHT_FLICKER_FORCE_CAMERA_REFRESH=0

exec "$TASK_ROOT/benchmark.sh" sim "$@"
