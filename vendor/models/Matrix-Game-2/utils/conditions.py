
import torch
import random
import math


def Custom_constant_action_universal(
    num_frames, action="turn_left", turn_degrees=360.0
):
    """Build one deterministic universal-scene action for the whole video."""
    if num_frames < 2:
        raise ValueError("Custom action requires at least 2 video frames.")
    if turn_degrees < 0:
        raise ValueError("turn_degrees must be non-negative.")

    keyboard_map = {
        "turn_left": [0, 0, 0, 0],
        "turn_right": [0, 0, 0, 0],
        "forward": [1, 0, 0, 0],
        "backward": [0, 1, 0, 0],
        "strafe_left": [0, 0, 1, 0],
        "strafe_right": [0, 0, 0, 1],
        "forward_turn_left": [1, 0, 0, 0],
        "forward_turn_right": [1, 0, 0, 0],
        "backward_turn_left": [0, 1, 0, 0],
        "backward_turn_right": [0, 1, 0, 0],
        "idle": [0, 0, 0, 0],
        "stop": [0, 0, 0, 0],
    }
    if action not in keyboard_map:
        raise ValueError(f"Unsupported custom action: {action}")

    # Match Matrix-Game-3's nominal yaw calibration. Matrix-Game-2 consumes
    # mouse values directly and does not expose an explicit camera pose.
    degrees_per_action_unit = 15.0
    num_transitions = num_frames - 1
    keyboard_condition = torch.zeros((num_frames, 4), dtype=torch.float32)
    mouse_condition = torch.zeros((num_frames, 2), dtype=torch.float32)
    keyboard_condition[:num_transitions] = torch.tensor(
        keyboard_map[action], dtype=torch.float32
    )

    yaw_action = 0.0
    if action.endswith("turn_left") or action == "turn_left":
        yaw_action = -turn_degrees / (
            degrees_per_action_unit * num_transitions
        )
    elif action.endswith("turn_right") or action == "turn_right":
        yaw_action = turn_degrees / (
            degrees_per_action_unit * num_transitions
        )
    mouse_condition[:num_transitions, 1] = yaw_action

    return {
        "keyboard_condition": keyboard_condition,
        "mouse_condition": mouse_condition,
        "phase_info": [{
            "name": action,
            "frames": num_transitions,
            "angle": yaw_action * degrees_per_action_unit * num_transitions,
            "yaw_action": yaw_action,
        }],
    }


def Custom_sequence_actions_universal(num_frames, sequence):
    """Build a timed universal action sequence.

    Syntax:
      turn_left:degrees:seconds
      turn_right:degrees:seconds
      idle:seconds

    Phase seconds are relative duration weights because Matrix-Game-2 emits
    video in fixed latent blocks. The launch script reports exact playback
    duration after rounding to a legal latent-frame count.
    """
    if not sequence:
        raise ValueError("action_sequence must not be empty.")

    phase_specs = []
    for raw_phase in sequence.split(","):
        parts = [part.strip() for part in raw_phase.strip().split(":")]
        name = parts[0]
        if name in ("idle", "stop"):
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid idle phase: {raw_phase}. Use idle:seconds."
                )
            angle = 0.0
            keyboard = [0, 0, 0, 0]
            duration_text = parts[1]
            normalized_name = "idle"
        elif name in ("turn_left", "turn_right"):
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid rotation phase: {raw_phase}. Use "
                    "turn_left:degrees:seconds."
                )
            try:
                degrees = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid angle in phase: {raw_phase}") from exc
            if degrees <= 0:
                raise ValueError(f"Rotation angle must be positive: {raw_phase}")
            angle = -degrees if name == "turn_left" else degrees
            keyboard = [0, 0, 0, 0]
            duration_text = parts[2]
            normalized_name = name
        else:
            raise ValueError(
                f"Invalid action phase: {raw_phase}. Use turn_left, "
                "turn_right, or idle."
            )

        try:
            duration = float(duration_text)
        except ValueError as exc:
            raise ValueError(f"Invalid duration in phase: {raw_phase}") from exc
        if duration <= 0:
            raise ValueError(f"Phase duration must be positive: {raw_phase}")
        phase_specs.append(
            (normalized_name, angle, duration, keyboard)
        )

    num_transitions = num_frames - 1
    total_duration = sum(duration for _, _, duration, _ in phase_specs)
    raw_counts = [
        num_transitions * duration / total_duration
        for _, _, duration, _ in phase_specs
    ]
    phase_counts = [math.floor(count) for count in raw_counts]
    remainder = num_transitions - sum(phase_counts)
    remainder_order = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - phase_counts[idx],
        reverse=True,
    )
    for idx in remainder_order[:remainder]:
        phase_counts[idx] += 1
    if any(count == 0 for count in phase_counts):
        raise ValueError("Video is too short for the requested action sequence.")

    degrees_per_action_unit = 15.0
    keyboard_condition = torch.zeros((num_frames, 4), dtype=torch.float32)
    mouse_condition = torch.zeros((num_frames, 2), dtype=torch.float32)
    phase_info = []
    cursor = 0
    for (name, angle, _, keyboard), count in zip(phase_specs, phase_counts):
        yaw_action = angle / (
            degrees_per_action_unit * count
        ) if angle else 0.0
        keyboard_condition[cursor:cursor + count] = torch.tensor(
            keyboard, dtype=torch.float32
        )
        mouse_condition[cursor:cursor + count, 1] = yaw_action
        phase_info.append({
            "name": name,
            "frames": count,
            "angle": angle,
            "yaw_action": yaw_action,
        })
        cursor += count

    assert cursor == num_transitions
    return {
        "keyboard_condition": keyboard_condition,
        "mouse_condition": mouse_condition,
        "phase_info": phase_info,
    }

def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse == True:
        mouse_condition = torch.zeros((num_frames, 2))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        if mouse == True:
            m = data[rd]['mouse_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            if mouse == True:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            if mouse == True:
                mouse_condition[current_frame:current_frame+rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse == True:
        return {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition
            }
    return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr" 
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)


def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
    }
    
    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "jump",
        "slide",
        "leftside",
        "rightside",
        "turnleft",
        "turnright",
        "nomove"
    ]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "nomove": 0, "jump": 1, "slide": 2, "turnleft": 3,
        "turnright": 4, "leftside": 5, "rightside": 6
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)
