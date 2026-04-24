#!/usr/bin/env python3

# Script to interface the boreas dataset with ROS2
# Read the IMU and radar data and publish them as ROS2 topics (standard IMU and image messages + custom radar info message)
# Options are the playback rate and the path to the sequence folder
import pyboreas as pb
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from dr_pogo.msg import RadarInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import argparse
import time



class BoreasPlayerNode(Node):
    def getElapsedTime(self):
        return ((time.time() * 1e6) - self.actual_time_origin) * self.playback_rate
    
    def loadNextRadar(self, next_radar_idx):
        if next_radar_idx >= len(self.boreas.radar_frames):
            return None
        self.next_radar = self.boreas.get_radar(next_radar_idx)
        if self.old_format:
            self.next_chirps = np.ones((self.next_radar.polar.shape[0],), dtype=np.uint8) * 255
        else:
            base_path = '/'.join(self.next_radar.sensor_root.split('/')[:-1])
            doppler_path = base_path + '/radar/' + self.next_radar.frame + '.png'
            doppler_img = cv2.imread(doppler_path, cv2.IMREAD_GRAYSCALE)
            up_chrips = doppler_img[:,10]
            self.next_chirps = up_chrips

    def __init__(self, sequence_path, playback_rate, no_wait=False):
        super().__init__('boreas_player_node')

        self.playback_rate = playback_rate
        self.no_wait = no_wait

        if no_wait:
            self.odom_sub = self.create_subscription(Odometry, 'dro_odometry', self.odomCallback, 10)

        # Make sure the sequence path exists
        if(not os.path.exists(sequence_path)):
            self.get_logger().error(f"Sequence path {sequence_path} does not exist.")
            return
        
        # Extract sequence ID and data folder from the sequence path
        if(sequence_path[-1] == '/'):
            sequence_path = sequence_path[:-1]
        sequence_id = sequence_path.split('/')[-1]
        data_folder = '/'.join(sequence_path.split('/')[:-1])
        data_folder = data_folder.replace('//', '/')
        data_folder = data_folder.replace('\\ ', ' ')
        self.get_logger().info(f"Data folder: {data_folder}, Sequence ID: {sequence_id}")

        # Get the year from the sequence ID
        year = sequence_id[7:11]
        if int(year) < 2022:
            self.old_format = True
        else:
            self.old_format = False


        # Initialize Boreas dataset
        self.boreas = pb.BoreasDataset(data_folder, [[sequence_id]])

        # Load the IMU data
        imu_path = os.path.join(sequence_path, 'imu/dmu_imu.csv')
        if not self.old_format:
            if not os.path.exists(imu_path):
                self.get_logger().error(f"IMU path {imu_path} does not exist.")
                return
            imu_pd = pd.read_csv(imu_path)
            self.imu_timestamps = imu_pd['time'].values * 1e-3 # Convert to microseconds
            self.imu_gyr = imu_pd[['wx', 'wy', 'wz']].values
            T_applanix_dmu = np.loadtxt(os.path.join(sequence_path, 'calib/T_applanix_dmu.txt'))
            T_radar_lidar = np.loadtxt(os.path.join(sequence_path, 'calib/T_radar_lidar.txt'))
            T_applanix_lidar = np.loadtxt(os.path.join(sequence_path, 'calib/T_applanix_lidar.txt'))
            T_radar_imu = T_radar_lidar @ np.linalg.inv(T_applanix_lidar) @ T_applanix_dmu
        else:
            imu_path = os.path.join(sequence_path, 'applanix/imu_raw.csv')
            imu_pd = pd.read_csv(imu_path)
            self.imu_timestamps = imu_pd['GPSTime'].values * 1e6 # Convert to microseconds
            T_radar_lidar = np.loadtxt(os.path.join(sequence_path, 'calib/T_radar_lidar.txt'))
            T_applanix_lidar = np.loadtxt(os.path.join(sequence_path, 'calib/T_applanix_lidar.txt'))
            T_radar_imu = T_radar_lidar @ np.linalg.inv(T_applanix_lidar)
            self.imu_gyr = imu_pd[['angvel_x', 'angvel_y', 'angvel_z']].values

        # Project the IMU gyroscope measurements to the radar frame
        self.imu_gyr = (T_radar_imu[0:3, 0:3] @ self.imu_gyr.T).T


        self.get_logger().warning(f"Loaded {len(self.imu_timestamps)} IMU measurements.")
        self.get_logger().warning("IMU accelerometer measurements are not published in this version, and the GYR measurements are already projected to the radar frame.")


        # Initialize Boreas dataset reader
        next_radar_idx = 0

        # Get the timestamp of the first and last radar frames
        num_frames = len(self.boreas.radar_frames)
        next_imu_idx = 0
        self.loadNextRadar(next_radar_idx)
        data_time_origin = self.next_radar.timestamps[0][0] - 1e6
        # Set the next_imu_idx to the first IMU measurement after data_time_origin
        while(self.imu_timestamps[next_imu_idx] < data_time_origin):
            next_imu_idx += 1


        # Publishers
        imu_pub = self.create_publisher(Imu, 'boreas/imu', 10)
        radar_image_pub = self.create_publisher(Image, 'boreas/radar_image', 10)
        radar_info_pub = self.create_publisher(RadarInfo, 'boreas/radar_info', 10)

        # CV Bridge
        bridge = CvBridge()

        if self.no_wait:
            self.next_time = self.next_radar.timestamps[-1][0] + 0.5e6  # Set to the end of the radar timestamps to publish all data immediately after receiving the odometry message

        # Playback loop
        self.actual_time_origin = time.time() * 1e6  # in microseconds
        if self.no_wait:
            data_time_origin = 0.0

        loop = True
        while loop:
            if not self.no_wait:
                elapsed_actual_time = self.getElapsedTime()
            else:
                elapsed_actual_time = self.next_time
                #print("Next radar timestamp: ", self.next_radar.timestamps[-1][0] - data_time_origin, " microseconds")



            if(self.next_radar is not None) and (elapsed_actual_time > (self.next_radar.timestamps[-1][0] - data_time_origin)):
                self.get_logger().info("Publishing radar frame " + str(next_radar_idx) + " / " + str(num_frames))

                # Publish radar image
                polar_image = self.next_radar.polar
                radar_image_msg = bridge.cv2_to_imgmsg(polar_image, encoding="32FC1")
                radar_image_msg.header.stamp.sec = int(self.next_radar.timestamp )
                radar_image_msg.header.stamp.nanosec = int((self.next_radar.timestamp % 1) * 1e9)
                radar_image_msg.header.frame_id = "radar"
                radar_image_pub.publish(radar_image_msg)

                # Publish radar info
                radar_info_msg = RadarInfo()
                radar_info_msg.header = radar_image_msg.header
                radar_info_msg.azimuth = np.asarray(self.next_radar.azimuths, dtype=np.float32).ravel().tolist()

                radar_info_msg.timestamps = np.asarray(self.next_radar.timestamps, dtype=np.int64).ravel().tolist()

                radar_info_msg.resolution = self.next_radar.resolution

                radar_info_msg.chirps = np.asarray(self.next_chirps, dtype=np.uint8).ravel().tolist()
                radar_info_msg.sequence_id = sequence_id
                radar_info_pub.publish(radar_info_msg)




                self.next_radar.unload_data()

                next_radar_idx += 1
                if(next_radar_idx == num_frames):
                    self.next_radar = None
                else:
                    self.loadNextRadar(next_radar_idx)

            if(next_imu_idx < len(self.imu_timestamps)) and (elapsed_actual_time > (self.imu_timestamps[next_imu_idx] - data_time_origin)):
                # Publish IMU message
                imu_msg = Imu()
                imu_msg.header.stamp.sec = int(self.imu_timestamps[next_imu_idx] * 1e-6)
                imu_msg.header.stamp.nanosec = int((self.imu_timestamps[next_imu_idx] % 1e6) * 1e3)
                imu_msg.header.frame_id = "radar"
                imu_msg.angular_velocity.x = self.imu_gyr[next_imu_idx, 0]
                imu_msg.angular_velocity.y = self.imu_gyr[next_imu_idx, 1]
                imu_msg.angular_velocity.z = self.imu_gyr[next_imu_idx, 2]
                imu_msg.angular_velocity_covariance[:] = -1  # Unknown covariance
                imu_msg.linear_acceleration_covariance[:] = -1  # Unknown covariance
                imu_msg.linear_acceleration.x = 0.0  # Not available
                imu_msg.linear_acceleration.y = 0.0  # Not available
                imu_msg.linear_acceleration.z = 0.0  # Not available
                imu_pub.publish(imu_msg)

                next_imu_idx += 1

            # Sleep for the estimated time until the next event
            if not self.no_wait:
                time.sleep(0.001)  # Convert microseconds to seconds

            # Check for termination condition            
            if (self.next_radar is None) and (next_imu_idx >= len(self.imu_timestamps)):
                self.get_logger().info("Finished playing all data.")
                break

            rclpy.spin_once(self, timeout_sec=0.001)

    def odomCallback(self, msg):
        self.next_time = msg.header.stamp.sec * 1e6 + msg.header.stamp.nanosec * 1e-3 + 1.5e6  # Add 0.5s to ensure we cover the radar timestamps
        self.get_logger().info(f"Received odometry message, setting next_time to {self.next_time} microseconds")

def main():
    parser = argparse.ArgumentParser(description='Boreas Dataset Player Node')
    parser.add_argument('-p', '--sequence_path', type=str, required=True, help='Path to the Boreas sequence folder')
    parser.add_argument('-r', '--playback_rate', type=float, default=1.0, help='Playback rate (1.0 = real-time)')
    parser.add_argument('-n', '--no_wait', action='store_true', help='Send all messages as fast as possible after receiving the odometry result, ignoring the timestamps. Useful for testing the odometry evaluation pipeline offline.')
    args = parser.parse_args()

    rclpy.init()
    BoreasPlayerNode(args.sequence_path, args.playback_rate, args.no_wait)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

