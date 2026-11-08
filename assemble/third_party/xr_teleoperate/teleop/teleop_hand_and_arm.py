import time
import argparse
import json
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController, R1_A5_ArmController, R1_A7_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK, R1_A5_ArmIK, R1_A7_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

MOBILE_BASE_STATE_NAMES = [
    "base_forward_position_m",
    "base_lateral_position_m",
    "base_yaw_delta_rad",
    "lift_height_m",
    "left_track_velocity_mps",
    "right_track_velocity_mps",
    "lift_velocity_mps",
]
MOBILE_BASE_ACTION_NAMES = [
    "left_track_velocity_target_mps",
    "right_track_velocity_target_mps",
    "lift_velocity_target_mps",
]
HAND_CART_STATE_NAMES = [
    "cart_world_x_m",
    "cart_world_y_m",
    "cart_world_z_m",
    "cart_world_qw",
    "cart_world_qx",
    "cart_world_qy",
    "cart_world_qz",
]
SINGLE_OBJECT_POSE_STATE_NAMES = [
    "object_world_x_m", "object_world_y_m", "object_world_z_m",
    "object_world_qw", "object_world_qx", "object_world_qy", "object_world_qz",
]
LEIGUAN_ROD_POSE_STATE_NAMES = [
    "leiguan_world_x_m", "leiguan_world_y_m", "leiguan_world_z_m",
    "leiguan_world_qw", "leiguan_world_qx", "leiguan_world_qy", "leiguan_world_qz",
    "rod_world_x_m", "rod_world_y_m", "rod_world_z_m",
    "rod_world_qw", "rod_world_qx", "rod_world_qy", "rod_world_qz",
]


def extract_mobile_base_sample(
    sim_state,
    expected_task_id,
    max_age_seconds=0.5,
    include_hand_cart_state=False,
):
    """Validate and order the optional tracked-base sample from rt/sim_state."""
    if not isinstance(sim_state, dict):
        return None
    runtime_state = sim_state.get("runtime_state")
    if not isinstance(runtime_state, dict):
        return None
    if sim_state.get("task_name") != expected_task_id:
        return None
    if runtime_state.get("task_id") != expected_task_id:
        return None
    if runtime_state.get("schema") != "tracked_base_v1":
        return None
    try:
        timestamp_ns = int(runtime_state["timestamp_ns"])
        age_seconds = (time.time_ns() - timestamp_ns) / 1.0e9
        if age_seconds < -0.1 or age_seconds > max_age_seconds:
            return None
        state_names = list(MOBILE_BASE_STATE_NAMES)
        if include_hand_cart_state:
            state_names.extend(HAND_CART_STATE_NAMES)
        state_values = np.asarray(
            [runtime_state["state"][name] for name in state_names], dtype=np.float32
        )
        action_values = np.asarray(
            [runtime_state["action"][name] for name in MOBILE_BASE_ACTION_NAMES],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(state_values).all() or not np.isfinite(action_values).all():
        return None
    return state_values, action_values


def object_pose_state_names(schema):
    if schema == "single":
        return list(SINGLE_OBJECT_POSE_STATE_NAMES)
    if schema == "leiguan-rod":
        return list(LEIGUAN_ROD_POSE_STATE_NAMES)
    raise ValueError(f"Unsupported object-pose schema: {schema}")


def extract_object_pose_sample(
    sim_state, expected_task_id, schema="leiguan-rod", max_age_seconds=0.5
):
    """Return a fresh task-object XYZ/WXYZ pose from simulator telemetry."""
    if not isinstance(sim_state, dict):
        return None
    runtime_state = sim_state.get("runtime_state")
    if not isinstance(runtime_state, dict):
        return None
    if sim_state.get("task_name") != expected_task_id:
        return None
    if runtime_state.get("task_id") != expected_task_id:
        return None
    try:
        timestamp_ns = int(runtime_state["timestamp_ns"])
        age_seconds = (time.time_ns() - timestamp_ns) / 1.0e9
        if schema == "single":
            pose = np.asarray(runtime_state["object_root_pose_wxyz"], dtype=np.float32)
        elif schema == "leiguan-rod":
            objects = runtime_state.get("objects", {})
            leiguan = np.asarray(objects["leiguan"]["root_pose_wxyz"], dtype=np.float32)
            rod = np.asarray(objects["rod"]["root_pose_wxyz"], dtype=np.float32)
            pose = np.concatenate((leiguan, rod))
        else:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if age_seconds < -0.1 or age_seconds > max_age_seconds:
        return None
    if pose.shape != (len(object_pose_state_names(schema)),):
        return None
    if not np.isfinite(pose).all():
        return None
    return pose


def extract_replay_snapshot(sim_state, expected_task_id, max_age_seconds=0.5):
    """Freeze one fresh simulator reset state for deterministic replay."""
    if not isinstance(sim_state, dict):
        return None
    if sim_state.get("task_name") != expected_task_id:
        return None
    raw_init_state = sim_state.get("init_state")
    if isinstance(raw_init_state, str):
        try:
            parsed_init_state = json.loads(raw_init_state)
        except (TypeError, ValueError):
            return None
    else:
        parsed_init_state = raw_init_state
    if not isinstance(parsed_init_state, dict):
        return None
    runtime_state = sim_state.get("runtime_state")
    if not isinstance(runtime_state, dict):
        return None
    if runtime_state.get("task_id") != expected_task_id:
        return None
    if not isinstance(runtime_state.get("replay_initial_state"), dict):
        return None
    try:
        simulator_timestamp_ns = int(runtime_state["timestamp_ns"])
        age_seconds = (time.time_ns() - simulator_timestamp_ns) / 1.0e9
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if age_seconds < -0.1 or age_seconds > max_age_seconds:
        return None
    # JSON round-trip makes the snapshot independent of the DDS subscriber's
    # mutable dictionary and rejects values that cannot be persisted.
    snapshot_state = json.loads(
        json.dumps(sim_state, ensure_ascii=False, allow_nan=False)
    )
    snapshot_state.pop("_timestamp", None)
    snapshot_state["init_state"] = parsed_init_state
    return {
        "schema": "unitree_lerobot_episode_replay_v1",
        "task_name": expected_task_id,
        "captured_at_ns": time.time_ns(),
        "simulator_timestamp_ns": simulator_timestamp_ns,
        "sim_state": snapshot_state,
    }

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

QUEST_BASE_COMMAND_TOPIC = "rt/hand_cart_base/cmd"


def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def publish_quest_base_command(forward_axis, publisher, task_id, sequence):
    """Publish a task-scoped, dead-man base command from Quest buttons."""

    if publisher is None:
        return sequence
    payload = {
        "schema": "hand_cart_base_command_v1",
        "task_id": str(task_id),
        "sequence": int(sequence),
        "timestamp_ns": time.time_ns(),
        "forward_axis": float(np.clip(forward_axis, -1.0, 1.0)),
    }
    publisher.Write(
        String_(data=json.dumps(payload, separators=(",", ":"), allow_nan=False))
    )
    return sequence + 1

def home_arms_and_reset_all(arm_controller, publisher):
    """Match the configured task-home pose before a full simulator reset.

    The arm controller publishes its last target continuously at 250 Hz.  A
    simulator-only reset is therefore immediately overwritten unless that DDS
    target is changed to the same home target used by the q exit path.
    """
    logger_mp.info("Returning both arms to task home before the full simulator reset...")
    arm_controller.ctrl_dual_arm_go_home()
    publish_reset_category(2, publisher)

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
LEROBOT_MODE   = False  # Use explicit accept/discard episode controls
ACCEPT_REQUEST = False
DISCARD_REQUEST = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, ACCEPT_REQUEST, DISCARD_REQUEST
    if key == 'q':
        START = False
        STOP = True
    elif LEROBOT_MODE and key == 's':
        DISCARD_REQUEST = True
    elif not LEROBOT_MODE and key == 'r':
        START = True
    elif not LEROBOT_MODE and key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2', 'R1_A5', 'R1_A7'], default='G1_29', help='Select arm controller')
    parser.add_argument('--active-arm', type=str, choices=['both', 'left', 'right'], default='both', help='Select which arm and matching end effector follow XR input; inactive side holds its pose')
    parser.add_argument('--arm-home-q', type=float, nargs=14, default=None, metavar='Q', help='Optional 14-value G1_29 task-home pose: left 7 joints followed by right 7 joints')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    parser.add_argument('--lerobot-v21', action='store_true', help='Record an accepted/discardable LeRobot v2.1 dataset')
    parser.add_argument('--lerobot-v30', action='store_true', help='Record an accepted/discardable LeRobot Dataset v3.0 dataset')
    parser.add_argument('--lerobot-dataset-dir', type=str, default=None, help='LeRobot dataset root directory')
    parser.add_argument('--instruction', type=str, default='Insert the detonator lead wire into the detonator opening.', help='Natural-language task instruction stored in the dataset')
    parser.add_argument('--success-marker-file', type=str, default=None, help='Pause control and recording while this simulator SUCCESS marker exists')
    parser.add_argument('--lerobot-target-episodes', type=int, default=0, help='Display accepted/saved episode progress toward this target (0 disables the target)')
    parser.add_argument('--lerobot-resume', action=argparse.BooleanOptionalAction, default=True, help='Resume and append to an existing LeRobot v3.0 dataset (default: enabled)')
    parser.add_argument('--lerobot-mobile-base', action='store_true', help='Append opt-in tracked-base/lift state and action telemetry to LeRobot v3.0 samples')
    parser.add_argument('--lerobot-mobile-base-task', type=str, default='Isaac-HandCartPush-G129-Dex1-Joint', help='Expected simulator task id for --lerobot-mobile-base telemetry validation')
    parser.add_argument('--lerobot-hand-cart-state', action='store_true', help='Append the hand-cart world XYZ and WXYZ quaternion to observation.state')
    parser.add_argument('--lerobot-object-pose-task', type=str, default=None, help='Append fresh task-object world XYZ/WXYZ telemetry from this simulator task to observation.state')
    parser.add_argument('--lerobot-object-pose-schema', choices=['single', 'leiguan-rod'], default='leiguan-rod', help='Simulator object-pose telemetry layout used by --lerobot-object-pose-task')
    parser.add_argument('--quest-base-buttons', action='store_true', help='Use left Quest Y/X buttons for tracked-base forward/reverse on the hand-cart task')
    parser.add_argument('--lerobot-replay-snapshot-task', type=str, default=None, help='Require and store a fresh simulator reset snapshot for this task at episode start')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    lerobot_mode = args.lerobot_v21 or args.lerobot_v30
    LEROBOT_MODE = lerobot_mode
    record_enabled = args.record or lerobot_mode
    if args.lerobot_v21 and args.lerobot_v30:
        parser.error("--lerobot-v21 and --lerobot-v30 cannot be enabled together.")
    if args.record and lerobot_mode:
        parser.error("--record and LeRobot recording are separate formats; select only one.")
    if lerobot_mode and (args.arm != "G1_29" or args.ee not in ("dex1", "dex1_internal")):
        parser.error("LeRobot recording currently requires --arm G1_29 with --ee dex1 or dex1_internal.")
    if lerobot_mode and not args.lerobot_dataset_dir:
        parser.error("--lerobot-dataset-dir is required with LeRobot recording.")
    if args.lerobot_target_episodes < 0:
        parser.error("--lerobot-target-episodes must be non-negative.")
    if args.lerobot_mobile_base and not (args.sim and args.lerobot_v30):
        parser.error("--lerobot-mobile-base requires both --sim and --lerobot-v30.")
    if args.lerobot_hand_cart_state and not args.lerobot_mobile_base:
        parser.error("--lerobot-hand-cart-state requires --lerobot-mobile-base.")
    if args.lerobot_object_pose_task and not (args.sim and args.lerobot_v30):
        parser.error("--lerobot-object-pose-task requires --sim and --lerobot-v30.")
    if args.quest_base_buttons and not (
        args.sim and args.input_mode == "controller" and args.lerobot_mobile_base
    ):
        parser.error(
            "--quest-base-buttons requires --sim, --input-mode=controller, "
            "and --lerobot-mobile-base."
        )
    if args.lerobot_replay_snapshot_task and not (args.sim and args.lerobot_v30):
        parser.error("--lerobot-replay-snapshot-task requires --sim and --lerobot-v30.")

    if args.ee == "dex1_internal" and args.motion:
        parser.error("--ee dex1_internal does not currently support --motion.")
    if args.arm_home_q is not None:
        if args.arm != "G1_29":
            parser.error("--arm-home-q currently supports only --arm G1_29.")
        if args.ee == "dex1_internal":
            parser.error("--arm-home-q is not yet supported with --ee dex1_internal.")
        if not np.isfinite(np.asarray(args.arm_home_q, dtype=np.float64)).all():
            parser.error("--arm-home-q values must be finite.")

    quest_base_publisher = None
    quest_base_sequence = 0
    lerobot_recorder = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw"
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived

        if args.ee == "dex1_internal":
            if args.arm != "G1_29":
                raise ValueError("dex1_internal is only supported with --arm G1_29.")
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array,
                                                              dual_gripper_action_array, motion_mode=args.motion, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
            else:
                arm_ctrl = G1_29_ArmController(
                    motion_mode=args.motion,
                    simulation_mode=args.sim,
                    home_q=args.arm_home_q,
                )
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A5":
            arm_ik = R1_A5_ArmIK()
            arm_ctrl = R1_A5_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A7":
            arm_ik = R1_A7_ArmIK()
            arm_ctrl = R1_A7_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            if args.quest_base_buttons:
                quest_base_publisher = ChannelPublisher(
                    QUEST_BASE_COMMAND_TOPIC, String_
                )
                quest_base_publisher.Init()
                logger_mp.info(
                    "Quest base buttons enabled: left [Y] drives forward, "
                    "left [X] drives backward."
                )
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        if lerobot_mode:
            if args.lerobot_v30:
                from teleop.utils.lerobot_v30_writer import LeRobotV30Writer as LeRobotWriter
                lerobot_version = "v3.0"
            else:
                from teleop.utils.lerobot_v21_writer import LeRobotV21Writer as LeRobotWriter
                lerobot_version = "v2.1"
            arm_joint_names = [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]
            state_action_names = arm_joint_names + ["left_gripper_joint", "right_gripper_joint"]
            state_names = list(state_action_names)
            action_names = list(state_action_names)
            if args.lerobot_mobile_base:
                state_names.extend(MOBILE_BASE_STATE_NAMES)
                action_names.extend(MOBILE_BASE_ACTION_NAMES)
            if args.lerobot_hand_cart_state:
                state_names.extend(HAND_CART_STATE_NAMES)
            if args.lerobot_object_pose_task:
                state_names.extend(object_pose_state_names(args.lerobot_object_pose_schema))
            writer_options = {}
            if args.lerobot_v30:
                writer_options["resume"] = args.lerobot_resume
            lerobot_recorder = LeRobotWriter(
                root=args.lerobot_dataset_dir,
                fps=args.frequency,
                instruction=args.instruction,
                state_names=state_names,
                action_names=action_names,
                camera_keys=[
                    "observation.images.head",
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                ],
                **writer_options,
            )

        logger_mp.info("----------------------------------------------------------------")
        if lerobot_mode:
            logger_mp.info("🟢  Press right controller [B] to START teleoperation and a new episode.")
            logger_mp.info("✅  Press right controller [A] to SAVE the successful episode and reset all.")
            logger_mp.info("🗑️  Press [s] to DISCARD the episode and reset all.")
            logger_mp.info(f"LeRobot Dataset {lerobot_version}: {args.lerobot_dataset_dir}")
            logger_mp.info(f"Instruction: {args.instruction}")
            if args.lerobot_v30:
                logger_mp.info(f"Resume existing dataset: {args.lerobot_resume}")
            if args.lerobot_mobile_base:
                logger_mp.info(
                    "Mobile-base recording enabled: "
                    f"state={len(state_names)} dims, action={len(action_names)} dims, "
                    f"task={args.lerobot_mobile_base_task}"
                )
            if args.lerobot_target_episodes:
                logger_mp.info(
                    f"📊 Saved episodes: {lerobot_recorder.total_episodes}/{args.lerobot_target_episodes}"
                )
            else:
                logger_mp.info(f"📊 Saved episodes: {lerobot_recorder.total_episodes}")
        elif args.record:
            logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP and not lerobot_mode: # legacy mode waits here; collection mode handles idle/reset below.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)

        if lerobot_mode:
            logger_mp.info("----------------Waiting for right controller [B]---------------")
        else:
            logger_mp.info("---------------------🚀start Tracking🚀-------------------------")

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        last_camera_warning_time = 0.0
        last_mobile_base_warning_time = 0.0
        last_object_pose_warning_time = 0.0
        previous_right_a_button = False
        previous_right_b_button = False
        success_pause_logged = False
        inactive_left_arm_q = None
        inactive_right_arm_q = None
        pending_replay_metadata = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if record_enabled or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if record_enabled:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if record_enabled:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # LeRobot collection uses explicit episode acceptance.  Stop recording
            # before publishing reset so reset motion can never enter the dataset.
            if lerobot_mode:
                # Controller buttons are level signals, so only act on the rising
                # edge.  XR data is read while idle as well, allowing B to start
                # the very first episode without touching the keyboard.
                tele_data = tv_wrapper.get_tele_data()
                right_a_button = bool(tele_data.right_ctrl_aButton)
                right_b_button = bool(tele_data.right_ctrl_bButton)
                right_a_pressed = right_a_button and not previous_right_a_button
                right_b_pressed = right_b_button and not previous_right_b_button
                previous_right_a_button = right_a_button
                previous_right_b_button = right_b_button

                if right_b_pressed:
                    if not START:
                        can_start = True
                        if args.lerobot_replay_snapshot_task:
                            pending_replay_metadata = extract_replay_snapshot(
                                sim_state_subscriber.read_data(),
                                args.lerobot_replay_snapshot_task,
                            )
                            if pending_replay_metadata is None:
                                can_start = False
                                logger_mp.warning(
                                    "Right controller [B] ignored: waiting for a "
                                    "fresh, replayable simulator reset snapshot."
                                )
                            else:
                                replay_state = pending_replay_metadata["sim_state"][
                                    "runtime_state"
                                ]
                                logger_mp.info(
                                    "Replay snapshot captured: "
                                    f"reset={replay_state.get('reset_count')}, "
                                    f"object={replay_state.get('active_object')}, "
                                    f"bin={replay_state.get('active_bin')}, "
                                    f"color={replay_state.get('active_bin_color')}."
                                )
                        if can_start:
                            # Capture the inactive Dex1 at its actual position
                            # before enabling XR motion. This prevents a disabled
                            # gripper from jumping on the first frame.
                            if args.ee in ("dex1", "dex1_internal"):
                                with dual_gripper_data_lock:
                                    frozen_left_q = float(dual_gripper_state_array[0])
                                    frozen_right_q = float(dual_gripper_state_array[1])
                                if args.active_arm == "right":
                                    with left_gripper_value.get_lock():
                                        left_gripper_value.value = float(
                                            np.interp(frozen_left_q, [0.0, 5.4], [5.0, 7.0])
                                        )
                                elif args.active_arm == "left":
                                    with right_gripper_value.get_lock():
                                        right_gripper_value.value = float(
                                            np.interp(frozen_right_q, [0.0, 5.4], [5.0, 7.0])
                                        )
                            START = True
                    else:
                        logger_mp.warning("Right controller [B] ignored: an episode is already active.")
                if right_a_pressed:
                    if START and lerobot_recorder.is_recording:
                        ACCEPT_REQUEST = True
                    else:
                        logger_mp.warning("Right controller [A] ignored: no episode is being recorded.")

                if ACCEPT_REQUEST:
                    ACCEPT_REQUEST = False
                    START = False
                    if args.quest_base_buttons:
                        quest_base_sequence = publish_quest_base_command(
                            0.0,
                            quest_base_publisher,
                            args.lerobot_mobile_base_task,
                            quest_base_sequence,
                        )
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                    if lerobot_recorder.is_recording:
                        episode_index, frame_count = lerobot_recorder.save_episode()
                        logger_mp.info(f"✅ Saved LeRobot episode {episode_index} ({frame_count} frames).")
                        saved_count = lerobot_recorder.total_episodes
                        if args.lerobot_target_episodes:
                            logger_mp.info(
                                f"📊 Saved episodes: {saved_count}/{args.lerobot_target_episodes}"
                            )
                            if saved_count >= args.lerobot_target_episodes:
                                logger_mp.info("🎉 Saved episode target reached.")
                        else:
                            logger_mp.info(f"📊 Saved episodes: {saved_count}")
                        if args.sim:
                            home_arms_and_reset_all(arm_ctrl, reset_pose_publisher)
                    else:
                        logger_mp.warning("Right controller [A] ignored: no episode is being recorded.")
                if DISCARD_REQUEST:
                    DISCARD_REQUEST = False
                    START = False
                    if args.quest_base_buttons:
                        quest_base_sequence = publish_quest_base_command(
                            0.0,
                            quest_base_publisher,
                            args.lerobot_mobile_base_task,
                            quest_base_sequence,
                        )
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                    frame_count = lerobot_recorder.discard_episode()
                    logger_mp.info(f"🗑️ Discarded pending episode ({frame_count} frames).")
                    if args.sim:
                        home_arms_and_reset_all(arm_ctrl, reset_pose_publisher)
                if START and not lerobot_recorder.is_recording:
                    if args.lerobot_v30:
                        lerobot_recorder.start_episode(
                            replay_metadata=pending_replay_metadata
                        )
                    else:
                        lerobot_recorder.start_episode()
                    pending_replay_metadata = None
                    logger_mp.info(
                        f"🔴 Recording LeRobot episode {lerobot_recorder.total_episodes}; press right [A] to save or terminal [s] to discard."
                    )
                success_marker_present = bool(
                    args.success_marker_file
                    and os.path.isfile(args.success_marker_file)
                )
                if args.quest_base_buttons:
                    # Vuer reports Quest X/Y as left aButton/bButton.  Button
                    # layout makes upper Y forward and lower X reverse.  Both
                    # pressed, idle, or SUCCESS always produces a stop command.
                    quest_forward_axis = 0.0
                    if START and not success_marker_present:
                        quest_forward_axis = float(
                            bool(tele_data.left_ctrl_bButton)
                            - bool(tele_data.left_ctrl_aButton)
                        )
                    quest_base_sequence = publish_quest_base_command(
                        quest_forward_axis,
                        quest_base_publisher,
                        args.lerobot_mobile_base_task,
                        quest_base_sequence,
                    )
                if not START:
                    READY = True
                    inactive_left_arm_q = None
                    inactive_right_arm_q = None
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                    current_time = time.time()
                    time.sleep(max(0, (1 / args.frequency) - (current_time - start_time)))
                    continue
                if success_marker_present:
                    if not success_pause_logged:
                        logger_mp.info(
                            "SUCCESS: robot control and episode recording are paused; "
                            "press right [A] to save or terminal [s] to discard/reset."
                        )
                        success_pause_logged = True
                    with xr_motion_data_ready.get_lock():
                        xr_motion_data_ready.value = False
                    current_time = time.time()
                    time.sleep(
                        max(0, (1 / args.frequency) - (current_time - start_time))
                    )
                    continue
                success_pause_logged = False

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            if not lerobot_mode:
                tele_data = tv_wrapper.get_tele_data()
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                if args.active_arm in ("both", "left"):
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                if args.active_arm in ("both", "right"):
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                if args.active_arm in ("both", "left"):
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_hand_pinchValue
                if args.active_arm in ("both", "right"):
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            arm_half = len(current_lr_arm_q) // 2
            if args.active_arm == "right":
                if inactive_left_arm_q is None:
                    inactive_left_arm_q = current_lr_arm_q[:arm_half].copy()
                    logger_mp.info("Left arm and left end effector locked; right-only XR control is active.")
                sol_q = np.asarray(sol_q).copy()
                sol_tauff = np.asarray(sol_tauff).copy()
                sol_q[:arm_half] = inactive_left_arm_q
                sol_tauff[:arm_half] = 0.0
            elif args.active_arm == "left":
                if inactive_right_arm_q is None:
                    inactive_right_arm_q = current_lr_arm_q[arm_half:].copy()
                    logger_mp.info("Right arm and right end effector locked; left-only XR control is active.")
                sol_q = np.asarray(sol_q).copy()
                sol_tauff = np.asarray(sol_tauff).copy()
                sol_q[arm_half:] = inactive_right_arm_q
                sol_tauff[arm_half:] = 0.0
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if record_enabled:
                if args.record:
                    READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action (split into left/right halves by the arm's own DOF, so it works for any variant: H1/G1_23/R1_A5 = 4/5 per arm, G1_29/R1_A7 = 7)
                half = len(current_lr_arm_q) // 2
                left_arm_state,  right_arm_state  = current_lr_arm_q[:half], current_lr_arm_q[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                if lerobot_mode and lerobot_recorder.is_recording:
                    synchronized_images = None
                    if (head_img is not None and head_img.bgr is not None
                            and left_wrist_img is not None and left_wrist_img.bgr is not None
                            and right_wrist_img is not None and right_wrist_img.bgr is not None):
                        synchronized_images = {
                            "observation.images.head": head_img.bgr,
                            "observation.images.left_wrist": left_wrist_img.bgr,
                            "observation.images.right_wrist": right_wrist_img.bgr,
                        }
                    mobile_base_sample = None
                    if args.lerobot_mobile_base:
                        mobile_base_sample = extract_mobile_base_sample(
                            sim_state_subscriber.read_data(),
                            args.lerobot_mobile_base_task,
                            include_hand_cart_state=args.lerobot_hand_cart_state,
                        )
                        if (
                            mobile_base_sample is None
                            and time.time() - last_mobile_base_warning_time > 2.0
                        ):
                            logger_mp.warning(
                                "Waiting for fresh mobile-base telemetry from the expected simulator task; sample skipped."
                            )
                            last_mobile_base_warning_time = time.time()
                    object_pose_sample = None
                    if args.lerobot_object_pose_task:
                        object_pose_sample = extract_object_pose_sample(
                            sim_state_subscriber.read_data(),
                            args.lerobot_object_pose_task,
                            args.lerobot_object_pose_schema,
                        )
                        if (
                            object_pose_sample is None
                            and time.time() - last_object_pose_warning_time > 2.0
                        ):
                            logger_mp.warning(
                                "Waiting for fresh object-pose telemetry from "
                                "the expected simulator task; sample skipped."
                            )
                            last_object_pose_warning_time = time.time()
                    if synchronized_images is not None and (
                        not args.lerobot_mobile_base or mobile_base_sample is not None
                    ) and (
                        not args.lerobot_object_pose_task
                        or object_pose_sample is not None
                    ):
                        state_vector = np.concatenate((
                            np.asarray(current_lr_arm_q, dtype=np.float32),
                            np.asarray(left_ee_state + right_ee_state, dtype=np.float32),
                        ))
                        action_vector = np.concatenate((
                            np.asarray(sol_q, dtype=np.float32),
                            np.asarray(left_hand_action + right_hand_action, dtype=np.float32),
                        ))
                        if mobile_base_sample is not None:
                            mobile_state, mobile_action = mobile_base_sample
                            state_vector = np.concatenate((state_vector, mobile_state))
                            action_vector = np.concatenate((action_vector, mobile_action))
                        if object_pose_sample is not None:
                            state_vector = np.concatenate(
                                (state_vector, object_pose_sample)
                            )
                        lerobot_recorder.add_frame(state_vector, action_vector, synchronized_images)
                    elif (
                        synchronized_images is None
                        and time.time() - last_camera_warning_time > 2.0
                    ):
                        logger_mp.warning("Waiting for all three synchronized camera frames; sample skipped.")
                        last_camera_warning_time = time.time()

                if args.record and RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if quest_base_publisher is not None:
                publish_quest_base_command(
                    0.0,
                    quest_base_publisher,
                    args.lerobot_mobile_base_task,
                    quest_base_sequence,
                )
                quest_base_publisher.Close()
        except Exception as e:
            logger_mp.error(f"Failed to stop Quest base command publisher: {e}")

        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        try:
            if lerobot_mode and lerobot_recorder is not None:
                lerobot_recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close LeRobot recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
