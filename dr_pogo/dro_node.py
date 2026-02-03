import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer
from dr_pogo.msg import RadarInfo
import numpy as np




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

        self.radar_data_buffer = []
        self.imu_data_buffer = []

        self.last_imu_time = None


    def radarCallback(self, image_msg, radar_info_msg):
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

        #self.dro(self.radar_data_buffer[0], relevant_imus)
        self.get_logger().info(f"Processing radar scan (from {round(first_radar_time*1e-6, 3)} to {round(last_radar_time*1e-6, 3)}) with {len(relevant_imus)} IMU measurements from {round(imu_times[start_idx]*1e-6, 3)} to {round(imu_times[end_idx-1]*1e-6, 3)}")

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




if __name__ == '__main__':
    rclpy.init()
    dro_node = DroNode()
    rclpy.spin(dro_node)
    dro_node.destroy_node()
    rclpy.shutdown()