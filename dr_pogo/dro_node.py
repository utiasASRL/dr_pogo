import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from dr_pogo.msg import RadarInfo
import numpy as np
import yaml
import copy
from dro import Dro, kDefaultDroOpts
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd

class DroNode(Node):
    def __init__(self):
        super().__init__('dro_node')
        self.get_logger().info("DroNode has been started.")

        # Subscribe to the imu topic
        self.imu_subscription = self.create_subscription(
            Imu,
            '/boreas/imu',
            self.imuCallback,
            1000)
        
        # Subscribe synchronously to the image topic and radar info topic
        self.image_subscription = Subscriber(self, Image, '/boreas/radar_image')
        self.radar_info_subscription = Subscriber(self, RadarInfo, '/boreas/radar_info')
        self.ts = TimeSynchronizer([self.image_subscription, self.radar_info_subscription], 10)
        self.ts.registerCallback(self.radarCallback)

        # Set the publisher for the odometry
        self.odometry_publisher = self.create_publisher(Odometry, 'dro_odometry', 10)



        self.radar_data_buffer = []
        self.imu_data_buffer = []

        self.last_imu_time = None

        self.initialized = False

        # Load the config file and populate the DRO options
        config_file_path = "config/config_dro.yaml"
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        dro_opts = copy.deepcopy(kDefaultDroOpts)
        if 'estimation' not in config:
            self.get_logger().error("DRO configuration file is missing the 'estimation' section.")
            return
        for section_name, defaults in dro_opts.items():
            if section_name not in config:
                self.get_logger().warn(f"DRO configuration missing section '{section_name}', using defaults.")
                continue
            for key in defaults.keys():
                if key in config[section_name]:
                    if section_name == 'estimation' and key == 'T_axle_radar':
                        dro_opts[section_name][key] = np.array(config[section_name][key], dtype=np.float64)
                    else:
                        dro_opts[section_name][key] = config[section_name][key]


        for section_name, section_cfg in config.items():
            if section_name not in dro_opts:
                self.get_logger().warn(
                    f"DRO configuration section '{section_name}' is not used by defaults and will be ignored."
                )
                continue
            unknown_keys = set(section_cfg.keys()) - set(dro_opts[section_name].keys())
            if unknown_keys:
                self.get_logger().warn(
                    f"DRO configuration section '{section_name}' has unused keys: {sorted(unknown_keys)}"
                )


        self.dro = Dro(dro_opts)


    def initialize(self, radar_data):
        # Create the output folders for the sequence
        seq_ID = radar_data['sequence_id']
        self.seq_output_folder = os.path.join("output", seq_ID)
        print(f"Creating output folder: {self.seq_output_folder}")
        os.makedirs(self.seq_output_folder, exist_ok=True)

        # Create the odometry output file
        self.odometry_output_path = os.path.join(self.seq_output_folder, "odometry_result")
        if os.path.exists(self.odometry_output_path):
            os.system('rm -r ' + self.odometry_output_path)
        os.makedirs(self.odometry_output_path)
        self.odometry_output_path = os.path.join(self.odometry_output_path, seq_ID + '.txt')
        
        #self.odom_2d_path = os.path.join(self.seq_output_folder, 'odometry_2d')
        #if os.path.exists(self.odom_2d_path):
        #    os.system('rm -r ' + self.odom_2d_path)
        #os.makedirs(self.odom_2d_path)
        #self.odom_2d_path = os.path.join(self.odom_2d_path, seq_ID + '.txt')

        self.initialized = True


    def radarCallback(self, image_msg, radar_info_msg):
        if self.initialized == False:
            self.initialize({'sequence_id': radar_info_msg.sequence_id})
        polar_image = np.frombuffer(image_msg.data, dtype=np.float32).reshape((image_msg.height, image_msg.width))
        azimuths = np.asarray(radar_info_msg.azimuth, dtype=np.float32)
        timestamps = np.asarray(radar_info_msg.timestamps, dtype=np.int64)
        resolution = radar_info_msg.resolution
        chirps = np.asarray(radar_info_msg.chirps, dtype=np.uint8)

        self.radar_data_buffer.append({
            'polar': polar_image,
            'azimuths': azimuths,
            'timestamps': timestamps,
            'resolution': resolution,
            'chirps': chirps,
            'timestamp': np.int64(image_msg.header.stamp.sec * 1e6) + np.int64(image_msg.header.stamp.nanosec / 1e3)
        })

        self.odometryStepIfReady()

    def imuCallback(self, msg):
        time = np.int64(msg.header.stamp.sec * 1e6) + np.int64(msg.header.stamp.nanosec / 1e3)
        if self.last_imu_time is not None and time <= self.last_imu_time:
            self.get_logger().warn(f"Received out-of-order IMU message. Current time: {time}, Last time: {self.last_imu_time}")
            return
        self.last_imu_time = time
        imu_data = {
            'timestamp': time,
            'angular_velocity': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'linear_acceleration': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        }
        self.imu_data_buffer.append(imu_data)

        self.odometryStepIfReady()

    def odometryStepIfReady(self):
        if len(self.radar_data_buffer) > 5:
            self.radar_data_buffer.pop(0)
            self.get_logger().warn("Radar buffer > 5, likely a problem in with IMU data, dropping oldest radar.")

        # Check if the first IMU is before the first radar timestamps and the last IMU is after the last radar timestamp
        if len(self.radar_data_buffer) == 0 or len(self.imu_data_buffer) == 0:
            return
        first_radar_time = self.radar_data_buffer[0]['timestamps'][0]
        last_radar_time = self.radar_data_buffer[0]['timestamps'][-1]
        if self.imu_data_buffer[0]['timestamp'] > first_radar_time or self.imu_data_buffer[-1]['timestamp'] < last_radar_time:
            return
        
        # Get the minimum number of IMU measurements that cover the radar timestamps
        imu_times = np.array([imu['timestamp'] for imu in self.imu_data_buffer])
        start_idx = np.searchsorted(imu_times, first_radar_time, side='left')
        start_idx = max(0, start_idx - 1)  # Ensure at least one IMU before the radar timestamps
        end_idx = np.searchsorted(imu_times, last_radar_time, side='right')
        end_idx = min(len(imu_times), end_idx + 1)  # Ensure at least one IMU after the radar timestamps
        relevant_imus = self.imu_data_buffer[start_idx:end_idx]

        self.get_logger().info(f"Processing radar scan (from {round(first_radar_time*1e-6, 3)} to {round(last_radar_time*1e-6, 3)}) with {len(relevant_imus)} IMU measurements from {round(imu_times[start_idx]*1e-6, 3)} to {round(imu_times[end_idx-1]*1e-6, 3)}")
        self.get_logger().warn("TODO: Add the proper handling of the gyro data (and bias estimation) in the DRO code")
        self.dro.odometryStep(self.radar_data_buffer[0], relevant_imus)


        # Get the odometry results
        current_odometry = self.dro.getPose(self.radar_data_buffer[0]['timestamp'])
        self.get_logger().info(f"Current odometry:\n{current_odometry}")
        self.publishOdometry(current_odometry)
        self.logOdometry(current_odometry, self.radar_data_buffer[0]['timestamp'])

        # If use gyro, feed the velocity estimates with the gyro for 3D state estimation
        self.get_logger().warn("TODO: compute the 3D pose")


        self.get_logger().info("TODO: Publish the local map and pose")


        # Clear the first radar
        temp_last_time = self.radar_data_buffer[0]['timestamps'][-1]
        # Remove the processed radar from the buffer
        self.radar_data_buffer.pop(0)

        # Remove the IMU measurements to keep at least one IMU before the next radar timestamps
        if len(self.radar_data_buffer) > 0:
            next_radar_time = self.radar_data_buffer[0]['timestamps'][0]
        else:
            next_radar_time = temp_last_time
        next_start_idx = np.searchsorted(imu_times, next_radar_time, side='left')
        self.imu_data_buffer = self.imu_data_buffer[next_start_idx - 1:]  # Keep one IMU before the next radar



    def publishOdometry(self, pose):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "radar"

        # Set position
        odom_msg.pose.pose.position.x = pose[0, 3]
        odom_msg.pose.pose.position.y = pose[1, 3]
        odom_msg.pose.pose.position.z = 0.0  # Assuming planar motion

        # Set orientation (yaw only)
        quaternion = R.from_matrix(pose[:3, :3]).as_quat()  # Convert rotation matrix to quaternion
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]
        self.odometry_publisher.publish(odom_msg)


    def logOdometry(self, pose, timestamp):
        inv_pose = np.linalg.inv(pose)
        data = np.concatenate([timestamp.reshape(1, 1), inv_pose[:3, :].reshape(1, -1)], axis=1)
        df_odom = pd.DataFrame(data)
        if not os.path.exists(self.odometry_output_path):
            df_odom.to_csv(self.odometry_output_path, header=None, index=None, sep=' ')
        else:
            df_odom.to_csv(self.odometry_output_path, mode='a', header=None, index=None, sep=' ')

            


if __name__ == '__main__':
    rclpy.init()
    dro_node = DroNode()
    rclpy.spin(dro_node)
    dro_node.destroy_node()
    rclpy.shutdown()