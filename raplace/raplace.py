'''
Introduction:
Simple Python Code for the Paper :
RaPlace：Place Recognition for Imaging Radar using Radon Transform and Mutable Threshold
'''

# Create Time: Aug 11th, 2023

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import math
import os
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon
from scipy import ndimage
import scipy.io as sio  # when the python file deal with .mat document
import pandas as pd
from utils import utils
import yaml
import time


def main(seq_dir):


    # Get the maximum local map range
    max_local_map_range = utils.getMaxLocalMapRange()


    # Load the RaPlace config file
    with open(os.path.join("raplace", "config.yaml"), 'r') as f:
        opts = yaml.safe_load(f)

    down_shape = 0.6

    time_start = time.time()

    # Generate Radon Transforms (not all the data, only a subset based on a distance threshold)
    data_sinofft, data_rowkeys, data_names, times, odom_poses, distances = generateRadon(seq_dir, down_shape, real_time=True, max_img_size=opts['max_img_size'])

    num_queries = len(data_names)


    # Structure to store the loop candidates
    loop_records = []

    for query_idx in range(num_queries - 1):
        print("\rProcessing local map", query_idx + 1, " / ", num_queries - 1, end='          ')

        # Get the query data
        query_sinofft = data_sinofft[query_idx]
        query_sinofft = (query_sinofft - np.mean(query_sinofft)) / np.std(query_sinofft)


        # Get the data that is far enough from the query data in time, but close enough spatially
        mask = (times - times[query_idx]) <  -opts['min_time_diff']
        max_relative_travelled_dist = (np.abs(distances - distances[query_idx]) * opts['max_odom_drift']) + max_local_map_range
        odom_dist = np.sqrt((odom_poses[:,0,3] - odom_poses[query_idx,0,3])**2 + (odom_poses[:,1,3] - odom_poses[query_idx,1,3])**2 + (odom_poses[:,2,3] - odom_poses[query_idx,2,3])**2)
        mask = np.logical_and(mask, odom_dist < max_relative_travelled_dist)
        if sum(mask) == 0:
            print(f"No valid candidates for query {query_idx}. Skipping...")
            continue
        can_sinofft = [data_sinofft[i] for i in range(len(data_sinofft)) if mask[i] and i != query_idx]
        can_idx = [i for i in range(len(data_sinofft)) if mask[i] and i != query_idx]

        tmpval = 0
        maxval = 0
        rotval = 0
        candnum = 0

        all_scores = []            # keep every (score, idx)


        for cands, tmp_sinofft in enumerate(can_sinofft):
            _, score = fast_dft(query_sinofft, tmp_sinofft)
            all_scores.append((score, cands))
            if score > maxval:
                maxval  = score
                candnum = cands


        nearest_idx = can_idx[candnum]
        _, tmpval = fast_dft(query_sinofft, query_sinofft)
        min_dist = abs(tmpval - maxval) 

        # Write the loop closure candidates to a CSV file
        loop_records.append({
            'time_i': times[query_idx],
            'time_j': times[nearest_idx],
            'scan_i_name': data_names[query_idx],
            'scan_j_name': data_names[nearest_idx],
            'score': float(maxval),
            'min_dist': float(min_dist)
        })    


    time_end = time.time()
    np.savetxt(seq_dir + '/raplace_time.txt', np.array([round(time_end - time_start, 2), num_queries]), fmt='%.2f', header='Total time (s), Number of queries')

    df = pd.DataFrame(loop_records)
    df.to_csv(seq_dir + '/raplace_loops.csv', index=False)
    print(f"Wrote {len(df)} loop candidates to raplace_loops.csv")




# Calculate the Euclidean distance between two two-dimensional postures
def dist_btn_pose(pose1, pose2):
    dist = math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist

# Radar Similarity Appraisal using Discrete Fourier Transform
# resulting in a one-dimensional array of cross-correlation
def fast_dft(Mq, Mi):
    Fq = np.fft.fft(Mq, axis=0)  # fft along theta axis
    Fn = np.fft.fft(Mi, axis=0)
    corrmap_2d = np.fft.ifft(Fq * np.conj(Fn), axis=0)

    corrmap = np.sum(corrmap_2d, axis=-1)
    maxval = np.max(corrmap)
    return corrmap, maxval

# Another Radon Transform Method
# steps is a constant, usually steps = 180
def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, steps), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s*180/steps, reshape=False).astype('float64')
        #print(sum(rotation).shape)
        res[:,s] = sum(rotation)
    return res

# Extract the middle row of the matrix
def rowkey(sino):
    row, col = sino.shape
    row_key = sino[(row + 1) // 2 - 1, :]
    # row_key = sino[math.floor((row + 1) / 2), :]
    return row_key

# Returns a list of file names under the specified path, except for special directories
def osdir(path):
    files = os.listdir(path)
    files = [file for file in files if not file.startswith('.')]

    return sorted(files, key=lambda fn: int(fn.split('.')[0]))

# Perform radon transformation
def generateRadon(data_dir, down_shape, real_time = True, max_img_size=600):
    
    # Get the local map paths
    local_map_dir = os.path.join(data_dir, 'local_maps')
    data_names = osdir(local_map_dir)

    # Odometry poses
    traj_dir = os.path.join(data_dir, 'odometry_result')
    # Load the *.txt file containing in the traj_dir
    traj_file = [file for file in os.listdir(traj_dir) if file.endswith('.txt')]
    traj = np.loadtxt(os.path.join(traj_dir, traj_file[0]), delimiter=' ')
    if traj.shape[0] != (len(data_names) + 1):
        raise ValueError(f"Trajectory length {traj.shape[0]} does not match number of data files {len(data_names)}.")
    
    # Extract the poses to compute the cumulative distance
    distances = np.zeros(len(data_names))
    poses = np.zeros((len(data_names), 4, 4))
    dist_sum = 0.0
    for i in range(1, traj.shape[0]):
        T1_inv = np.eye(4)
        T1_inv[:3, :] = traj[i - 1, 1:].reshape(3, 4)
        T2 = np.eye(4)
        T2[:3, :] = traj[i, 1:].reshape(3, 4)
        T2 = np.linalg.inv(T2)

        dT = T1_inv @ T2
        
        dist = np.linalg.norm(dT[:3, 3])
        dist_sum += dist
        distances[i - 1] = dist_sum
        poses[i-1] = T2
    

    # Prepare the data for the output
    num_data = len(data_names)
    data_names_kept = []
    odom_poses = []
    distances_kept = []
    theta = np.arange(0, 180)
    sinoffts = []
    rowkeys = []
    times = []

    # Loop through the local maps
    time_thr = utils.nameToTime(data_names[0]) - 1.0
    for data_idx in range(num_data):

        time_data = utils.nameToTime(data_names[data_idx])
        if real_time and (time_data < time_thr):
            continue

        t0 = time.time()


        # Print the current processing status
        file_name = data_names[data_idx]
        print("\rDistance:", np.round(distances[data_idx], 2), " / ", np.round(distances[-1], 2), "   index: ", data_idx, " / ", num_data, "   file: ", file_name, end='          ')

        # Extract the timestamp from the file name
        times.append(time_data)

        # Read the image
        data_path = os.path.join(local_map_dir, file_name)
        tmp = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        if tmp.shape[0] > max_img_size:
            tmp = cv2.resize(tmp, (max_img_size, max_img_size))

        # Perform Radon transform
        R = radon(tmp, theta)
        xp = np.arange(-R.shape[1] // 2, R.shape[1] // 2)

        R = R / np.max(R)
        R = R.astype(np.float64)
        R = cv2.resize(R, (int(down_shape * R.shape[1]), int(down_shape * R.shape[0])))

        # sinofft = np.abs(np.fft.fft(R, axis=0)[:R.shape[0] // 2, :])
        sinofft = np.abs(np.fft.fft(R, axis=0))
        sinofft_rows = sinofft.shape[0]
        sinofft = sinofft[:sinofft_rows // 2, :]

        # Append the results
        sinoffts.append(sinofft)
        rowkeys.append(rowkey(R).flatten())
        data_names_kept.append(file_name)
        distances_kept.append(distances[data_idx])
        odom_poses.append(poses[data_idx,:,:])

        t1 = time.time()
        if real_time:
            time_thr = time_data + (t1 - t0)



    # Convert to numpy arrays
    rowkeys = np.array(rowkeys)
    times = np.array(times)
    distances_kept = np.array(distances_kept)
    odom_poses = np.array(odom_poses)

    return sinoffts, rowkeys, data_names_kept, times, odom_poses, distances_kept



if __name__ == "__main__":
    # Get the data directory from the DRO config file
    if( utils.isMultiSequence() ):
        seqs = utils.getOutputDataDirs()
    else:
        seq_dir = utils.getOutputDataDir()
        seqs = [seq_dir]

    for seq_dir in seqs:
        main(seq_dir)