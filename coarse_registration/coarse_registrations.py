import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils

import yaml
import numpy as np
import pandas as pd
import cv2
import time


def main(output_path):
    # Get the data directory
    local_map_path = os.path.join(output_path, "local_maps")

    # Load coarse registrations parameters
    opts = yaml.safe_load(open(os.path.join("coarse_registration", "config.yaml"), 'r'))

    # Get the pixel resolution and maximum image size
    pix_res = utils.getPixelResolution()
    max_img_size = opts['max_img_size']


    # Read the RaPlace matches
    try:
        raw_loops = pd.read_csv(os.path.join(output_path, "raplace_loops.csv"))
        print("Loaded", len(raw_loops), "RaPlace matches.")
    except Exception as e:
        print("Error loading RaPlace matches:", e)
        return


    # Store the valid matches
    valid_matches = []


    # Create the cv tools for feature extraction and matching
    # nfeatures=0 means no limit (default), set to a specific number to limit features
    sift_extractor = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=20, sigma=2.5)
    sift_matcher = cv2.BFMatcher()

    time_start = time.time()

    # Loop through the matches
    for index, row in raw_loops.iterrows():
        # Read the images
        img1 = cv2.imread(os.path.join(local_map_path, row['scan_i_name']), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(local_map_path, row['scan_j_name']), cv2.IMREAD_GRAYSCALE)
        original_shape = img1.shape
        img_size_ratio = 1.0
        if img1.shape[0] > max_img_size:
            img_size_ratio = img1.shape[0] / max_img_size
            img1 = cv2.resize(img1, (max_img_size, max_img_size))
            img2 = cv2.resize(img2, (max_img_size, max_img_size))
            

        ## Perform histogram equalization
        #img1 = cv2.equalizeHist(img1)
        #img2 = cv2.equalizeHist(img2)

        # Extract features
        kp1, des1 = sift_extractor.detectAndCompute(img1, None)
        kp2, des2 = sift_extractor.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            print(f"Skipping match {index} due to missing descriptors.")
            continue
        if img_size_ratio != 1.0:
            # Scale the keypoints to the original image size
            for kp in kp1:
                kp.pt = (kp.pt[0] * img_size_ratio, kp.pt[1] * img_size_ratio)
            for kp in kp2:
                kp.pt = (kp.pt[0] * img_size_ratio, kp.pt[1] * img_size_ratio)

        # Match features
        matches = sift_matcher.knnMatch(des1, des2, k=2)

        # Apply simple rules to remove bad matches
        good_matches = []
        for m, n in matches:
            # Check if the match features have the same scale
            #if np.abs((kp1[m.queryIdx].octave & 255)/(kp2[n.trainIdx].octave & 255) -1) > 0.2:
            if (kp1[m.queryIdx].octave & 255) != (kp2[n.trainIdx].octave & 255):
                continue

            # Apply ratio test
            if m.distance < opts['lowe_ratio'] * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            print(f"Skipping match {index} due to insufficient good matches.")
            continue

        # Print the number of good matches
        print(f"Match {index} has {len(good_matches)} good matches.")



        # Perform RANSAC registration
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=opts['ransac_thr'])
        if M is None:
            print(f"Skipping match {index} due to failed registration.")
            continue

        # Get the pose and scale from the transformation matrix
        pose, scale = utils.affineToPoseAndScale(M, pix_res, original_shape)


        # Reject if the scale is too far from 1
        if np.abs(scale - 1) > 0.05:
            print(f"Skipping match {index} due to scale {scale:.3f} being too far from 1.")
            continue


        # Add the valid match
        xy, theta = utils.poseToXYTheta(pose)
        valid_matches.append({
            'scan_i_name': row['scan_i_name'],
            'scan_j_name': row['scan_j_name'],
            'x': xy[0],
            'y': xy[1],
            'theta': theta
        })

        if opts['visualize']:
            # Show the registration result
            M[:,-1] = M[:,-1] / img_size_ratio  # Scale the transformation matrix
            img1_reg = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
            cv2.imshow("Match", np.hstack((img1_reg, img2)))
            cv2.waitKey(0)
        
    time_end = time.time()
    np.savetxt(output_path + '/coarse_registration_time.txt', np.array([round(time_end - time_start, 2), len(raw_loops), len(valid_matches)]), fmt='%.2f', header='Total time (s), Number of queries, Number of valid matches')

    if opts['visualize']:
        # Close the OpenCV windows
        cv2.destroyAllWindows()


    # Save the valid matches
    valid_matches_df = pd.DataFrame(valid_matches)
    valid_matches_df.to_csv(os.path.join(output_path, "coarse_registrations.csv"), index=False)


if __name__ == "__main__":
    if utils.isMultiSequence():
        seq_list = utils.getOutputDataDirs()
    else:
        seq_list = [utils.getOutputDataDir()]
    for output_path in seq_list:
        main(output_path)