import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_hdf5_data(hdf5_file, output_dir):
    # 打开 HDF5 文件
    with h5py.File(hdf5_file, 'r') as f:
        # 获取图像数据
        images = f['observations/images/front'][:]
        # 获取关节角度数据 (假设 qpos 是关节角度数据)
        qpos = f['observations/qpos'][:]
        # 获取末端位姿数据
        end_pose = f['observations/end_pose'][:]

        # 读取图像数据并显示
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(5):
            axes[i].imshow(images[i])
            axes[i].axis('off')
            axes[i].set_title(f"Frame {i}")
        plt.tight_layout()
        # 保存图像
        plt.savefig(f"{output_dir}/images_visualization_{os.path.basename(hdf5_file)}.png")
        plt.close()

        # 关节角度可视化：展示前几个关节角度
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(qpos[:10, 0], label='Joint 1')
        ax.plot(qpos[:10, 1], label='Joint 2')
        ax.plot(qpos[:10, 2], label='Joint 3')
        ax.plot(qpos[:10, 3], label='Joint 4')
        ax.plot(qpos[:10, 4], label='Joint 5')
        ax.plot(qpos[:10, 5], label='Joint 6')
        ax.set_title('Joint Angles')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (rad)')
        ax.legend()
        # 保存关节角度图
        plt.savefig(f"{output_dir}/joint_angles_visualization_{os.path.basename(hdf5_file)}.png")
        plt.close()

        # 末端位姿可视化
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(end_pose[:10, 0], label='X')
        ax.plot(end_pose[:10, 1], label='Y')
        ax.plot(end_pose[:10, 2], label='Z')
        ax.set_title('End Pose (Position)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (m)')
        ax.legend()
        # 保存末端位姿图
        plt.savefig(f"{output_dir}/end_pose_visualization_{os.path.basename(hdf5_file)}.png")
        plt.close()

def visualize_all_hdf5_files(input_dir, output_dir):
    # 获取文件夹中的所有 .hdf5 文件
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 对每个 HDF5 文件进行可视化
    for hdf5_file in hdf5_files:
        hdf5_file_path = os.path.join(input_dir, hdf5_file)
        print(f"Visualizing {hdf5_file_path}...")
        visualize_hdf5_data(hdf5_file_path, output_dir)

if __name__ == "__main__":
    input_dir = './Softacman/softacman_datacollect/hdf5_data/insert1024'  # HDF5 文件所在的目录
    output_dir = './Softacman/visualize_hdf5_data/insert1024'  # 可视化结果保存的目录

    # 执行可视化操作
    visualize_all_hdf5_files(input_dir, output_dir)

    print("Visualization completed for all files.")
