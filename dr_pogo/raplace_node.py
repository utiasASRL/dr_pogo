#!/usr/bin/env python3

import os
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from skimage.transform import radon
import yaml
import message_filters

from dr_pogo.msg import LoopCandidate
from dr_pogo.msg import LocalMapInfo


@dataclass
class MapEntry:
    index: int
    timestamp_us: int
    image_path: str
    sinofft: np.ndarray


class RaplaceNode(Node):
    def __init__(self) -> None:
        super().__init__("raplace_node")
    
        # Read the parameters from config/config_raplace.yaml
        config_file_path = "config/config_raplace.yaml"
        if not os.path.isfile(config_file_path):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_file_path = os.path.join(base_path, "share/dr_pogo", config_file_path)
            self.output_dir = os.path.join(base_path, "share/dr_pogo", "raplace_local_maps")
        else:
            self.output_dir = "raplace_local_maps"
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        self.max_img_size = config["max_img_size"]
        self.min_time_diff = config["min_time_diff"]
        self.max_odom_drift = config["max_odom_drift"]
        self.pix_res = None
        
        self.down_shape = 0.6

        # Ensure the output directory exists and is empty
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        self.entries: List[MapEntry] = []
        self.theta = np.arange(0, 180)

        # Create a synchronous subscription with larger queue and allow_headerless=False for strict sync
        self.subs = []
        self.subs.append(message_filters.Subscriber(self, Image, "dro_local_map_image"))
        self.subs.append(message_filters.Subscriber(self, LocalMapInfo, "dro_local_map_info"))
        self.ts = message_filters.TimeSynchronizer(self.subs, 2)
        self.ts.registerCallback(self.raplaceCallback)
    

        self.cumulated_dists = []
        self.times = []
        self.odome_poses = []

        self.candidate_pub = self.create_publisher(LoopCandidate, "raplace_loop_candidate", 10)

        self.get_logger().info(
            f"RaPlace online node started. Subscribed to 'dro_local_map_image' and 'dro_odometry'. Publishing candidates on 'raplace_loop_candidate'."
        )

    def timestampToFileName(self, timestamp_us: int) -> str:
        return os.path.join(self.output_dir, f"{timestamp_us}.png")


    @staticmethod
    def timestamp2us(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000 + int(msg.header.stamp.nanosec) // 1_000

    @staticmethod
    def fastDft(m_query: np.ndarray, m_item: np.ndarray) -> float:
        f_query = np.fft.fft(m_query, axis=0)
        f_item = np.fft.fft(m_item, axis=0)
        corrmap_2d = np.fft.ifft(f_query * np.conj(f_item), axis=0)
        corrmap = np.sum(corrmap_2d, axis=-1)
        maxval = np.max(corrmap)
        return float(np.real(maxval))


    def computeSinofft(self, img_u8: np.ndarray) -> np.ndarray:
        if img_u8.shape[0] > self.max_img_size:
            scale_factor = self.max_img_size / img_u8.shape[0]
            new_width = int(img_u8.shape[1] * scale_factor)
            img_u8 = cv2.resize(img_u8, (new_width, self.max_img_size))
        elif img_u8.shape[1] > self.max_img_size:
            scale_factor = self.max_img_size / img_u8.shape[1]
            new_height = int(img_u8.shape[0] * scale_factor)
            img_u8 = cv2.resize(img_u8, (self.max_img_size, new_height))

        r = radon(img_u8, self.theta)
        max_r = float(np.max(r))
        if max_r > 1e-8:
            r = r / max_r

        r = r.astype(np.float64)
        r = cv2.resize(
            r,
            (int(self.down_shape * r.shape[1]), int(self.down_shape * r.shape[0]))
        )

        sinofft = np.abs(np.fft.fft(r, axis=0))
        return sinofft[: sinofft.shape[0] // 2, :]

    def saveLocalMap(self, img_u8: np.ndarray, timestamp_us: int) -> str:
        file_path = self.timestampToFileName(timestamp_us)
        cv2.imwrite(file_path, img_u8)
        return file_path

    def findBestCandidate(self, query_entry: MapEntry, odom_pose: np.ndarray):
        # Only compare with entries that are sufficiently far in time and space
        time_mask = np.array(self.times) < (query_entry.timestamp_us - self.min_time_diff * 1e6)
        if not np.any(time_mask):
            return None
        dists = np.linalg.norm(np.array(self.odome_poses)[:,:2] - odom_pose[:2], axis=1)
        space_mask = dists < (self.max_odom_drift * (self.cumulated_dists[-1] - np.array(self.cumulated_dists)) + 50.0)
        valid_mask = np.logical_and(time_mask, space_mask)

        if not np.any(valid_mask):
            return None

        valid_ids = np.where(valid_mask)[0]

        query_norm = (query_entry.sinofft - np.mean(query_entry.sinofft)) / (np.std(query_entry.sinofft) + 1e-8)


        best_entry = None
        best_score = -1.0
        for idx in valid_ids:
            entry = self.entries[idx]
            score = self.fastDft(query_norm, entry.sinofft)
            if score > best_score:
                best_score = score
                best_entry = entry

        self_score = self.fastDft(query_norm, query_norm)
        min_dist = abs(self_score - best_score)
        return best_entry, best_score, min_dist



    def publishCandidate(self, query_entry: MapEntry, candidate_entry: MapEntry, score: float, min_dist: float, source_msg: Image):
        out = LoopCandidate()
        out.header = source_msg.header
        out.query_time = int(query_entry.timestamp_us)
        out.candidate_time = int(candidate_entry.timestamp_us)
        out.query_index = int(query_entry.index)
        out.candidate_index = int(candidate_entry.index)
        out.score = float(score)
        out.min_dist = float(min_dist)
        out.query_image_path = query_entry.image_path
        out.candidate_image_path = candidate_entry.image_path
        out.resolution = float(self.pix_res) if self.pix_res is not None else -1.0
        self.candidate_pub.publish(out)

    def raplaceCallback(self, image_msg: Image, info_msg: LocalMapInfo):
        # Get the resolution of the local map from the first message
        if self.pix_res is None:
            self.pix_res = info_msg.resolution
            self.get_logger().info(f"Set pixel resolution to {self.pix_res} m/px based on the first received LocalMapInfo message.")


        # Compute the cumulated distance based on the odometry info
        xy = np.array([info_msg.x, info_msg.y])
        if len(self.cumulated_dists) == 0:
            self.cumulated_dists.append(0.0)
        else:
            dist = np.linalg.norm(xy - self.odome_poses[-1][:2])
            self.cumulated_dists.append(dist + (self.cumulated_dists[-1]))
        self.odome_poses.append(np.array([info_msg.x, info_msg.y, info_msg.theta]))

        print('Cumulated distance: ', self.cumulated_dists[-1], ' m')


        # Convert the incoming Image message to a numpy array
        image_np = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width))
        timestamp_us = self.timestamp2us(image_msg)
        odom_pose = np.array([info_msg.x, info_msg.y, info_msg.theta])

        # Create a new MapEntry (including computing the sinofft)
        map_entry = MapEntry(
            index=len(self.entries),
            timestamp_us=timestamp_us,
            image_path=self.timestampToFileName(timestamp_us),
            sinofft=self.computeSinofft(image_np),
        )
        self.entries.append(map_entry)
        self.times.append(timestamp_us)

        # Save the local map image to disk
        self.saveLocalMap(image_np, timestamp_us)


        # Find the best candidate for loop closure and publish it if it exists
        best_match = self.findBestCandidate(map_entry, odom_pose)
        if best_match is not None:
            best_entry, best_score, min_dist = best_match
            self.publishCandidate(map_entry, best_entry, best_score, min_dist, image_msg)
        


        if len(self.entries) > 1:
            print(f"Received local map with timestamp {timestamp_us}, delta to previous {timestamp_us - self.entries[-2].timestamp_us} us")



def main(args=None):
    rclpy.init(args=args)
    node = RaplaceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
