import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import ikpy.chain
import cv2
import ikpy.utils.plot as plot_utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import pandas as pd

from typing import Tuple, Any, Optional
from process_hdf5_with_upsample import upsample_frames

# Load the configuration from the config.json file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config', 'config.json')

with open(config_path, 'r') as config_file:
    config = json.load(config_file)
config = config["data_process_config"]

# Extract configuration values
START_QPOS = config["start_qpos"] # Initial joint positions for the robot (values specific to your robot's configuration)
PI = np.pi
ACTIVE_MASK = config["active_joint_mask"]

# load piper urdf
urdf_path = os.path.join(current_dir, "assets", "piper_description.urdf")
my_chain = ikpy.chain.Chain.from_urdf_file(
    urdf_file= urdf_path, 
    active_links_mask=ACTIVE_MASK
)

print(f"Number of joints in the chain: {len(my_chain)}")
print(f"Joints in the chain {my_chain}")

def video_to_array(video_path: str) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_array = np.empty((frame_count, height, width, 3), dtype=np.uint8)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                frames_array = frames_array[:i]
                frame_count = i
                break
            frames_array[i] = frame
        return frames_array, frame_count
    finally:
        cap.release()

def cartesian_to_joints(position, quaternion, initial_joint_angles=None, **kwargs):
    """
    Convert Cartesian coordinates to robot joint angles using inverse kinematics.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    if initial_joint_angles is None:
        initial_joint_angles = [0] * len(my_chain)
    joint_angles = my_chain.inverse_kinematics(
        position,
        rotation_matrix,
        orientation_mode='all',
        initial_position=initial_joint_angles
    )
    return joint_angles

def process_single_file(csv_file, input_dir, output_dir, episode_index, qpos_mode, camera_names):
    
    error_logs = []
    df = pd.read_csv(os.path.join(input_dir, csv_file))
    traj_name = csv_file.split("poses_")[1].split(".csv")[0]
    mp4_file = os.path.join(input_dir, f"{traj_name}.mp4")
    cam_file = os.path.join(input_dir, f"{traj_name}_Camera.mp4")
    if os.path.exists(mp4_file):
        frames, frame_count = video_to_array(mp4_file)
    elif os.path.exists(cam_file) :
        frames, frame_count = video_to_array(cam_file)
    else:
        raise NotImplementedError("No exist file")
    
    xyz = df.iloc[:, 1:4].values
    euler_angles = df.iloc[:, 4:7].values
    Inflation_deflation_status = df.iloc[:, 7].values
    rotations = R.from_euler('xyz', euler_angles, degrees=False)
    quaternions = rotations.as_quat()
    
    initial_joint_angles = None
    end_poses = []
    new_joint_angles = []
    new_poses = []
    valid_indices = []
    data_size = len(xyz)
    if data_size != frame_count:
        msg = f"Warning: Number of frames ({frame_count}) does not match data points ({data_size}) in {csv_file}"
        error_logs.append(msg)
        print(msg)
        if data_size > frame_count:
            frames = upsample_frames(frames, data_size)

    
    for i in range(data_size):
        x, y, z = xyz[i]
        qx, qy, qz, qw = quaternions[i]
        roll, pitch, yaw = euler_angles[i]
        end_pose = [x, y, z, qx, qy, qz, qw]
        end_poses.append(end_pose)
        direction, quaternion = [x, y, z], [qx, qy, qz, qw]
        seven_dof_pose = [x, y, z, roll, pitch, yaw, Inflation_deflation_status[i]]#0.0215
        new_poses.append(seven_dof_pose)
        
        try:
            if i == 0:
                initial_joint_angles = np.array(START_QPOS)
                full_joint_angles = cartesian_to_joints(direction, quaternion, initial_joint_angles)
            else:
                full_joint_angles = cartesian_to_joints(direction, quaternion, initial_joint_angles)
            if initial_joint_angles is not None:
                error = np.linalg.norm(np.array(full_joint_angles) - np.array(initial_joint_angles))
                threshold = 0.5 
                
                # Warn if the error between current and previous joint angles is large
                if error > threshold:
                    log_msg = f"Warning: Large joint update error at frame {i}: error={error}"
                    print(log_msg)
                    error_logs.append(log_msg)
            initial_joint_angles = full_joint_angles
            six_dof_joint_angles = full_joint_angles[1:7]
            # minus the real gap betweein gripper markers, 0.02m here
            joint_angles = np.append(six_dof_joint_angles, Inflation_deflation_status[i])
            new_joint_angles.append(joint_angles)
            valid_indices.append(i)
        except Exception as e:
            error_msg = f"Exception at frame {i}: {str(e)}"
            print(error_msg)
            error_logs.append(error_msg)
            continue
    
    frames = frames[valid_indices]

    
    output_file = os.path.join(output_dir, f"episode_{episode_index}.hdf5")
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['compress'] = False
        f_out.attrs['sim'] = False
        obs = f_out.create_group('observations')
        images = obs.create_group('images')
        images.create_dataset(name='front', dtype='uint8', data=frames)
        if qpos_mode == 'joint':
            obs.create_dataset('qpos', data=new_joint_angles)
            f_out.create_dataset('action', data=new_joint_angles)
        else:
            obs.create_dataset('qpos', data=new_poses)
            f_out.create_dataset('action', data=new_poses)
        obs.create_dataset('end_pose', data=np.array(end_poses))
        print(f"Data saved to {output_file}")
    
    if error_logs:
        log_file = os.path.join(output_dir, "../logs", f"episode_{episode_index}_error.log")
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        with open(log_file, 'w') as lf:
            for log in error_logs:
                lf.write(log + "\n")
        print(f"Error logs saved to {log_file}")
    
    print(f"Processed file: {csv_file}")


def process_file_wrapper(args):
    return process_single_file(*args)

def read_csv_ik_save_hdf5_parallel(input_dir, output_dir, use_multiprocessing=True, qpos_mode="joint", camera_names=["cam_right"]):
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    if use_multiprocessing:
        pool = Pool(cpu_count())
        tasks = [(csv_file, input_dir, output_dir, idx, qpos_mode, camera_names) for idx, csv_file in enumerate(csv_files)]
        for _ in tqdm(pool.imap_unordered(process_file_wrapper, tasks), total=len(tasks)):
            pass
        pool.close()
        pool.join()
    else:
        idx = 0
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            process_single_file(csv_file, input_dir, output_dir, idx, qpos_mode, camera_names)
            idx += 1

if __name__ == "__main__":
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    qpos_mode = config.get("qpos_mode", "joint") # How to store the robot pose? 'joint' for joint state, 'cartesian' for 6D pose
    camera_names = config.get("camera_names", ["cam_right"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    use_multiprocessing = config.get("use_multiprocessing", True)
    read_csv_ik_save_hdf5_parallel(input_dir, output_dir, use_multiprocessing=use_multiprocessing, qpos_mode=qpos_mode, camera_names=camera_names)

    print("Processing completed.")
