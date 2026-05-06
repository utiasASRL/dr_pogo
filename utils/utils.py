import os
import yaml
import numpy as np
import pandas as pd
from pyboreas.utils import odometry
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt

T_applanix_dmu = np.array([[9.99945620e-01, -1.03283308e-02, -1.44307407e-03, 0],
                [-1.03287236e-02, -9.99946622e-01, -2.64973148e-04, 0],
                [-1.44026031e-03,  2.79863852e-04, -9.99998924e-01, 0],
                [0, 0, 0, 1]])

kDataPaths = [
    "/media/ced/Extreme Pro/data/boreas/rss/ba",
    "/media/ced/Extreme Pro/data/boreas/rss/test",
    "/media/ced/Extreme Pro/data/boreas/original_train",
    "/media/ced/Extreme Pro/data/boreas/rt_radar",
    "/home/ced/Documents/data/boreas/rss/test",
    "/home/ced/Documents/data/boreas/original_train",
    "/home/ced/Documents/data/boreas/for_tbv/rss/boreas",
    "/home/ced/Documents/data/boreas/for_tbv/original_train/boreas",

]

kUpgradeTime = 1632182400
kRatioAfterUpgrade = 0.04381 / 0.0596 

def getOutputDataDir():
    # Fetch the sequence ID from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    if opts['data']['multi_sequence']:
        raise ValueError("This script is not designed for multi-sequence data.")
    data_dir = opts['data']['data_path']
    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]
    sequence_id = data_dir.split('/')[-1]

    # Get the data path
    data_dir = os.path.join("output", sequence_id)
    return data_dir

def isMultiSequence():
    # Fetch the multi_sequence flag from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['data']['multi_sequence']

def getOutputDataDirs():
    # Get the folders in the output directory that start with 'boreas-'
    output_path = "output"
    seq_dirs = [os.path.join(output_path, d) for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith('boreas-')]
    return seq_dirs



def getDataDir(seq_id=None):
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    if seq_id is None:
        # Fetch the sequence ID from the DRO config file
        if opts['data']['multi_sequence']:
            raise ValueError("This script is not designed for multi-sequence data.")
        data_dir = opts['data']['data_path']
        if data_dir.endswith('/'):
            data_dir = data_dir[:-1]
        # Get the sequence ID
        sequence_id = data_dir.split('/')[:-1]
        # Combine to get the full data path
        data_dir = '/'.join(sequence_id)
        return data_dir
    else:
        temp_paths = kDataPaths.copy()
        path_from_config = opts['data']['data_path']
        if not isMultiSequence():
            # Add the output paths to the search paths
            if path_from_config.endswith('/'):
                path_from_config = path_from_config[:-1]
            path_from_config = '/'.join(path_from_config.split('/')[:-1])
        temp_paths.append(path_from_config)
            
        for path in temp_paths:
            if os.path.exists(path):
                if os.path.exists(os.path.join(path, seq_id)):
                    return path
        raise ValueError(f"Data path for sequence ID {seq_id} not found in predefined paths.")



def affineToPoseAndScale(affine_matrix, pix_res, img_shape):
    # Transform to convert the opencv frame to the local map frame
    T_cv_local_map = np.array([[0, 1, 0, pix_res*img_shape[1]/2],
                               [-1, 0, 0, pix_res*img_shape[0]/2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    T_local_map_cv = np.linalg.inv(T_cv_local_map)

    # Get the scale from the affine matrix
    scale = np.linalg.norm(affine_matrix[0, :2])
    # Get the rotation from the affine matrix
    rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
    pose = np.eye(4)
    pose[0, 0] = np.cos(rotation)
    pose[0, 1] = -np.sin(rotation)
    pose[1, 0] = np.sin(rotation)
    pose[1, 1] = np.cos(rotation)
    pose[0, 3] = affine_matrix[0, 2] * pix_res
    pose[1, 3] = affine_matrix[1, 2] * pix_res

    # Convert the pose to the local map frame
    pose = T_local_map_cv @ pose @ T_cv_local_map
    return pose, scale

def poseToAffine(pose, pix_res, img_shape):
    # Transform to convert the opencv frame to the local map frame
    T_cv_local_map = np.array([[0, 1, 0, pix_res*img_shape[1]/2],
                               [-1, 0, 0, pix_res*img_shape[0]/2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    T_local_map_cv = np.linalg.inv(T_cv_local_map)

    # Convert the pose to the cv frame
    pose_cv = T_cv_local_map @ pose @ T_local_map_cv

    # Get the affine matrix from the pose
    affine_matrix = np.eye(2, 3)
    affine_matrix[0, 0] = pose_cv[0, 0]
    affine_matrix[0, 1] = pose_cv[0, 1]
    affine_matrix[0, 2] = pose_cv[0, 3] / pix_res
    affine_matrix[1, 0] = pose_cv[1, 0]
    affine_matrix[1, 1] = pose_cv[1, 1]
    affine_matrix[1, 2] = pose_cv[1, 3] / pix_res
    return affine_matrix



def getPixelResolution():
    # Fetch the pixel resolution from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['direct']['local_map_res']

def getMaxLocalMapRange():
    # Fetch the max local map range from the DRO config file
    with open(os.path.join("dro", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)
    return opts['direct']['max_local_map_range']


def poseToXYTheta(pose):
    # Convert the pose to (x, y, theta)
    x = pose[0, 3]
    y = pose[1, 3]
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([x, y]), theta


def XYThetaToPose(xy, theta):
    # Convert (x, y, theta) to pose
    pose = np.eye(4)
    pose[0, 0] = np.cos(theta)
    pose[0, 1] = -np.sin(theta)
    pose[1, 0] = np.sin(theta)
    pose[1, 1] = np.cos(theta)
    pose[0, 3] = xy[0]
    pose[1, 3] = xy[1]
    return pose



def getGTRadarPosesAndTimes(seq_id):
    # Get the data path
    data_path = os.path.join(getDataDir(seq_id), seq_id)

    # Get the gps poses
    gt_path = os.path.join(data_path, "applanix", "gps_post_process.csv")
    T_radar_applanix = np.loadtxt(os.path.join(data_path, "calib", "T_radar_lidar.txt")) @ np.linalg.inv(np.loadtxt(os.path.join(data_path, "calib", "T_applanix_lidar.txt")))
    gt_poses, gt_times = odometry.read_traj_file_gt(gt_path, T_radar_applanix, 2)
    gt_data_raw = pd.read_csv(gt_path)
    gt_times = gt_data_raw.iloc[:, 0].to_numpy()
    poses = []
    for pose in gt_poses:
        poses.append(np.linalg.inv(pose))
    gt_poses = np.array(poses)
    return gt_poses, gt_times

def getPogoPosesAndTimes(seq_id, ouput_path='output', file_name="pose_graph_traj.txt", delimiter=' '):
    # Get the results path
    data_path = os.path.join(ouput_path, seq_id, file_name)
    if not os.path.exists(data_path):
        print(f"Skipping sequence {seq_id} due to missing results.")
        return None, None
    data_raw = pd.read_csv(data_path, delimiter=delimiter)
    times = data_raw.iloc[:, 0].to_numpy()
    poses = []
    for i in range(len(times)):
        pose = np.eye(4)
        pose[:2, 3] = data_raw.iloc[i, 1:3].to_numpy()
        pose[:2, :2] = np.array([[np.cos(data_raw.iloc[i, 3]), -np.sin(data_raw.iloc[i, 3])],
                                  [np.sin(data_raw.iloc[i, 3]), np.cos(data_raw.iloc[i, 3])]])
        poses.append(pose)
    poses = np.array(poses)
    return poses, times

def getDroPosesAndTimes(seq_id, ouput_path='output'):
    # Get the results path
    data_path = os.path.join(ouput_path, seq_id, "odometry_2d", seq_id + ".txt")
    if not os.path.exists(data_path):
        print(f"Skipping sequence {seq_id} due to missing results.")
        return None, None
    data_raw = pd.read_csv(data_path, delimiter=' ')
    times = data_raw.iloc[:, 0].to_numpy()
    poses = []
    for i in range(len(times)):
        pose = np.eye(4)
        pose[:2, 3] = data_raw.iloc[i, 1:3].to_numpy()
        pose[:2, :2] = np.array([[np.cos(data_raw.iloc[i, 3]), -np.sin(data_raw.iloc[i, 3])],
                                  [np.sin(data_raw.iloc[i, 3]), np.cos(data_raw.iloc[i, 3])]])
        poses.append(pose)
    poses = np.array(poses)
    return poses, times

def getInterpolatedPose(poses, times, query_time):
    # Interpolate the pose at the query time
    if query_time < times[0] or query_time > times[-1]:
        print("Query time out of range:", query_time, times[0], times[-1])
        if query_time < times[0]:
            return poses[0]
        else:
            return poses[-1]
    # Find the right interval using searchsorted
    idx = np.searchsorted(times, query_time)
    if idx == 0:
        i = 0
    elif idx >= len(times):
        return poses[-1]
    else:
        i = idx - 1
    t0, t1 = times[i], times[i+1]
    t = (query_time - t0) / (t1 - t0)
    pose1 = poses[i]
    pose2 = poses[i+1]
    # Interpolate the translation
    delta_pos = (1 - t) * pose1[:3, 3] + t * pose2[:3, 3]
    # Interpolate the rotation using slerp
    rotations = R.from_matrix([pose1[:3, :3], pose2[:3, :3]])
    r = Slerp([0, 1], rotations)(t)
    delta_rot = r.as_matrix()
    # Combine the translation and rotation into a pose
    delta_pose = np.eye(4)
    delta_pose[:3, :3] = delta_rot
    delta_pose[:3, 3] = delta_pos
    return delta_pose

def getInterpolatedTrajectory(poses, times, query_times):
    interpolated_poses = np.zeros((len(query_times), 4, 4))
    for i, query_time in enumerate(query_times):
        interpolated_pose = getInterpolatedPose(poses, times, query_time)
        interpolated_poses[i] = interpolated_pose
    return interpolated_poses

def readFastLio2DTraj(path, seq_id):
    # Read the calibration
    raw_data_path = os.path.join(getDataDir(seq_id), seq_id)
    T_radar_lidar = np.loadtxt(os.path.join(raw_data_path, "calib", "T_radar_lidar.txt"))
    T_applanix_lidar = np.loadtxt(os.path.join(raw_data_path, "calib", "T_applanix_lidar.txt"))
    T_dmu_radar = np.linalg.inv(T_applanix_dmu) @ T_applanix_lidar @ np.linalg.inv(T_radar_lidar)


    data = np.loadtxt(path, delimiter=',')

    pos = data[:, 1:4]

    # Project points onto the plane so that z = 0
    poses = np.zeros((len(pos), 4, 4))
    for i in range(len(pos)):
        poses[i, :3, 3] = pos[i]
        poses[i, 3, 3] = 1
        poses[i, :3, :3] = R.from_quat(data[i, 4:8]).as_matrix()

        poses[i, :,:] = T_applanix_dmu @ poses[i, :,:] @ T_dmu_radar

    # Project the poses onto the plane
    poses = align3DPosesTo2D(poses)

    #plt.figure()
    #plt.plot(poses[:, 0, 3], poses[:, 1, 3])
    #plt.plot(pos[:, 0], pos[:, 1], '--')
    #plt.axis('equal')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Projected 2D Poses' + seq_id)
    #plt.show()

    return poses, data[:,0]

def readNavtechSLAM2DTraj(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    poses = np.zeros((len(data), 4, 4))
    poses[:, 0, 0] = data[:,11]
    poses[:, 0, 1] = data[:,12]
    poses[:, 0, 2] = data[:,13]
    poses[:, 0, 3] = data[:,14]
    poses[:, 1, 0] = data[:,15]
    poses[:, 1, 1] = data[:,16]
    poses[:, 1, 2] = data[:,17]
    poses[:, 1, 3] = data[:,18]
    poses[:, 2, 0] = data[:,19]
    poses[:, 2, 1] = data[:,20]
    poses[:, 2, 2] = data[:,21]
    poses[:, 2, 3] = data[:,22]
    poses[:, 3, 0] = data[:,23]
    poses[:, 3, 1] = data[:,24]
    poses[:, 3, 2] = data[:,25]
    poses[:, 3, 3] = data[:,26]

    T_flip = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

    poses = T_flip.reshape(1,4,4) @ poses

    if data[0,0] > kUpgradeTime:
        poses[:, :3, 3] *= kRatioAfterUpgrade
    # Project the poses onto the plane
    #poses = align3DPosesTo2D(poses)

    #plt.figure()
    #plt.plot(poses[:, 0], poses[:, 1])
    #plt.axis('equal')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Projected 2D Poses')
    #plt.show()

    return poses, data[:,0]

def read2Fast2Lamaa2DTraj(path, seq_id):
    raw_data_path = os.path.join(getDataDir(seq_id), seq_id)
    T_radar_lidar = np.loadtxt(os.path.join(raw_data_path, "calib", "T_radar_lidar.txt"))
    T_applanix_lidar = np.loadtxt(os.path.join(raw_data_path, "calib", "T_applanix_lidar.txt"))
    T_dmu_radar = np.linalg.inv(T_applanix_dmu) @ T_applanix_lidar @ np.linalg.inv(T_radar_lidar)


    data = np.loadtxt(path, delimiter=' ')

    poses = np.zeros((len(data), 4, 4))
    for i in range(len(data)):
        temp_T = np.eye(4)
        temp_T[:3, :] = data[i,1:].reshape(3, 4)
        poses[i, :, :] = T_applanix_dmu @ np.linalg.inv(temp_T) @ T_dmu_radar


    # Project the poses onto the plane
    poses = align3DPosesTo2D(poses)

    #plt.figure()
    #plt.plot(poses[:, 0], poses[:, 1])
    #plt.axis('equal')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Projected 2D Poses')
    #plt.show()

    return poses, data[:,0]*1e-6

def readTBV2DTraj(path):
    # paramers path 
    param_path = os.path.join(path, "pars.txt")
    # Read line by line to find the sequence ID
    with open(param_path, 'r') as f:
        lines = f.readlines()
    seq_id = None
    for line in lines:
        if line.startswith("sequence, "):
            seq_id = line.split(",")[1].strip()
            break
    if seq_id is None:
        raise ValueError("Sequence ID not found in pars.txt")
    
    graph_path = os.path.join(path, "graph.txt")
    # Read the graph file
    with open(graph_path, 'r') as f:
        lines = f.readlines()
    times = []
    poses = []
    for line in lines:
        if line.strip() == "":
            continue
        parts = line.split()
        times.append(int(parts[-1]))
        pose = np.eye(4)
        pose[:3, :] = np.array(parts[:-1], dtype=float).reshape(3, 4)
        poses.append(pose)
    
    poses = np.array(poses)

    if times[0]*1e-9 > kUpgradeTime:
        poses[:, :3, 3] *= kRatioAfterUpgrade

    #plt.figure()
    #plt.plot(poses[:, 0, 3], poses[:, 1, 3])
    #plt.axis('equal')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title('Projected 2D Poses')
    #plt.show()

    return poses, np.array(times)*1e-9, seq_id

    


def align3DPosesTo2D(poses):
    # First get the plane equation
    pts_3d = poses[:, :3, 3]
    centroid = np.mean(pts_3d, axis=0)
    centered_pts = pts_3d - centroid
    _,_, Vt = np.linalg.svd(centered_pts)
    normal = Vt[-1]
    # Quick and dirty fix
    if normal[2] < 0:
        normal = -normal
    d = -centroid @ normal

    projected_pts = pts_3d - np.dot(centered_pts, normal)[:, np.newaxis] * normal

    # Compute the rotation to align the normal to [0,0,1]
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    rotation_angle = np.dot(normal, z_axis)
    if np.allclose(rotation_axis, 0):
        R_align = np.eye(3) if rotation_angle > 0 else -np.eye(3)
    else:
        R_align = R.from_rotvec(rotation_axis / np.linalg.norm(rotation_axis) * np.arccos(rotation_angle)).as_matrix()

    projected_pts = projected_pts @ R_align.T
    projected_poses = np.zeros_like(poses)
    projected_poses[:, :3, 3] = projected_pts
    projected_poses[:, 3, 3] = 1
    projected_poses[:, :3, :3] = R_align @ poses[:, :3, :3]
    return projected_poses

def nameToTime(name):
    # Extract the timestamp from the filename
    time_str = name.split('.')[0]
    return float(time_str)*1e-6
def timeToName(time):
    # Convert the timestamp to the filename
    time_int = int(time)
    return str(time_int).zfill(16) + '.png'

def align2DTrajectories(gt_poses, gt_times, est_poses, est_times):
    gt_interp_poses = getInterpolatedTrajectory(gt_poses, gt_times, est_times)

    gt_xy = gt_interp_poses[:, :2, 3]
    est_xy = est_poses[:, :2, 3]
    gt_centroid = np.mean(gt_xy, axis=0)
    est_centroid = np.mean(est_xy, axis=0)

    # Center the trajectories
    gt_xy_centered = gt_xy - gt_centroid
    est_xy_centered = est_xy - est_centroid

    # Compute the covariance matrix
    H = gt_xy_centered.T @ est_xy_centered
    # Compute the SVD
    U, S, Vt = np.linalg.svd(H)
    # Compute the rotation
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        print("Reflection detected, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Compute the translation
    t = est_centroid - R @ gt_centroid
    # Apply the transformation to the est_xy
    est_xy_aligned = est_xy @ R - R.T@t

    return est_xy_aligned, R, t



def get2dATE(gt_poses, est_poses, save_fig=False, est_colour='b', est_label='Estimated', gt_colour='orange', gt_label='Ground Truth', path=None):
    if gt_poses is None or est_poses is None:
        print("Invalid input poses.")
        return None

    if len(gt_poses) != len(est_poses):
        print("Ground truth and estimated poses must have the same length.")
        return None

    # Align the trajectories (find the R and t that best aligns the trajectories)
    gt_xy = gt_poses[:, :2, 3]
    est_xy = est_poses[:, :2, 3]
    gt_centroid = np.mean(gt_xy, axis=0)
    est_centroid = np.mean(est_xy, axis=0)

    # Center the trajectories
    gt_xy_centered = gt_xy - gt_centroid
    est_xy_centered = est_xy - est_centroid

    # Compute the covariance matrix
    H = gt_xy_centered.T @ est_xy_centered

    # Compute the SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("Reflection detected, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation
    t = est_centroid - R @ gt_centroid

    # Apply the transformation to the est_xy
    est_xy_aligned = est_xy @ R - R.T@t

    ate = np.sqrt(np.mean(np.sum((gt_xy - est_xy_aligned)**2, axis=1)))

    #print("2D Absolute Trajectory Error (RMSE ATE):", ate)

    if save_fig:
        if path is None:
            path = "output/ate_2d_trajectory.pdf"
        plt.figure(figsize=(6, 6))
        plt.plot(est_xy_aligned[:, 0], est_xy_aligned[:, 1], label=est_label, color=est_colour, linewidth=0.5)
        plt.plot(gt_xy[:, 0], gt_xy[:, 1], label=gt_label, color=gt_colour, linewidth=0.5, linestyle='--')
        plt.legend(loc='upper left')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.savefig(path)
        plt.close()

    return ate

def getSeqType(seq_id):

    seqs = {
        'boreas-2024-12-05-14-12': 'Commercial',
        'boreas-2024-12-23-16-27': 'Commercial',
        'boreas-2024-12-23-16-44': 'Commercial',
        'boreas-2024-12-23-17-01': 'Commercial',
        'boreas-2024-12-23-17-18': 'Commercial',

        'boreas-2024-12-03-10-24': 'Glenshield',
        'boreas-2024-12-03-12-54': 'Glenshield',
        'boreas-2025-01-08-10-59': 'Glenshield',
        'boreas-2025-01-08-11-22': 'Glenshield',
        'boreas-2025-01-08-11-44': 'Glenshield',
        'boreas-2025-01-08-12-28': 'Glenshield',
        'boreas-2024-12-05-14-25': 'Glenshield',

        'boreas-2024-12-03-13-13': 'Highway',
        'boreas-2024-12-03-13-34': 'Highway',
        'boreas-2024-12-10-12-07': 'Highway',
        'boreas-2024-12-10-12-24': 'Highway',
        'boreas-2024-12-10-12-38': 'Highway',
        'boreas-2024-12-10-12-56': 'Highway',

        'boreas-2024-12-04-14-28': 'Tunnel',
        'boreas-2024-12-04-14-34': 'Tunnel',
        'boreas-2024-12-04-14-38': 'Tunnel',
        'boreas-2024-12-04-14-44': 'Tunnel',
        'boreas-2024-12-04-14-50': 'Tunnel',
        'boreas-2024-12-04-14-59': 'Tunnel',

        'boreas-2024-12-04-11-45': 'Skyway',
        'boreas-2024-12-04-11-56': 'Skyway',
        'boreas-2024-12-04-12-08': 'Skyway',
        'boreas-2024-12-04-12-19': 'Skyway',
        'boreas-2024-12-04-12-34': 'Skyway',


        'boreas-2020-11-26-13-58': 'Original_train',
        'boreas-2020-12-18-13-44': 'Original_train',
        'boreas-2021-01-26-11-22': 'Original_train',
        'boreas-2021-02-02-14-07': 'Original_train',
        'boreas-2021-03-02-13-38': 'Original_train',
        'boreas-2021-03-30-14-23': 'Original_train',
        'boreas-2021-04-20-14-11': 'Original_train',
        'boreas-2021-05-13-16-11': 'Original_train',
        'boreas-2021-07-20-17-33': 'Original_train',
        'boreas-2021-09-02-11-42': 'Original_train',
        'boreas-2021-10-15-12-35': 'Original_train',
        'boreas-2021-11-14-09-47': 'Original_train',
        'boreas-2021-11-23-14-27': 'Original_train',

        'boreas-2025-07-18-10-00': 'Forest',
        'boreas-2025-07-18-10-33': 'Forest',
        'boreas-2025-07-18-11-00': 'Forest',
        'boreas-2025-07-18-11-25': 'Forest',
        'boreas-2025-07-18-11-53': 'Forest',

        'boreas-2025-07-18-14-55': 'Farm',
        'boreas-2025-07-18-15-12': 'Farm',
        'boreas-2025-07-18-15-30': 'Farm',
        'boreas-2025-07-18-15-48': 'Farm',
        'boreas-2025-07-18-16-05': 'Farm',

        'boreas-2025-08-06-06-33': 'Urban',
        'boreas-2025-08-06-07-05': 'Urban',
        'boreas-2025-08-06-07-41': 'Urban',
        'boreas-2025-08-06-08-35': 'Urban',
        'boreas-2025-08-06-10-48': 'Urban',
    }
    if seq_id in seqs:
        return seqs[seq_id]
    else:
        return 'Unknown'
