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
        config_path = os.path.join("config", "config_raplace.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.max_img_size = config["max_img_size"]
        
        self.output_dir = "temp"
        # Ensure the output directory exists and is empty
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        self.entries: List[MapEntry] = []
        self.frame_counter = 0
        self.theta = np.arange(0, 180)

        # Create a synchronous subscription with larger queue and allow_headerless=False for strict sync
        self.subs = []
        self.subs.append(message_filters.Subscriber(self, Image, "dro_local_map_image"))
        self.subs.append(message_filters.Subscriber(self, LocalMapInfo, "dro_local_map_info"))
        self.ts = message_filters.TimeSynchronizer(self.subs, 2)
        self.ts.registerCallback(self.raplaceCallback)
    
        # Create the subscriber for the local map images for saving them to disk (with a larger queue size)
        self.image_saver_sub = self.create_subscription(
            Image,
            "dro_local_map_image",
            self.imageSaverCallback,
            100,
        )

        self.cumulated_dists = []
        self.last_xy = None

        self.candidate_pub = self.create_publisher(LoopCandidate, "raplace_loop_candidate", 10)

        self.get_logger().info(
            f"RaPlace online node started. Subscribed to 'dro_local_map_image' and 'dro_odometry'. Publishing candidates on 'raplace_loop_candidate'."
        )

    def imageSaverCallback(self, image_msg: Image):
        timestamp_us = self.timestamp2us(image_msg)
        image_path = self.saveLocalMap(image_msg, timestamp_us)
        self.get_logger().info(f"Saved local map image to {image_path}")


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

    def imageMsgToFloat32(self, msg: Image) -> np.ndarray:
        if msg.encoding != "32FC1":
            raise ValueError(f"Expected 32FC1 image encoding, got '{msg.encoding}'")

        data = np.frombuffer(msg.data, dtype=np.float32)
        if data.size != msg.height * msg.width:
            raise ValueError(
                f"Unexpected image payload size: got {data.size}, expected {msg.height * msg.width}"
            )

        image = data.reshape((msg.height, msg.width))
        return np.ascontiguousarray(image)

    def preprocessForRadon(self, local_map_float: np.ndarray) -> np.ndarray:
        img = np.nan_to_num(local_map_float, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)
        img_u8 = (img * 255.0).astype(np.uint8)

        if img_u8.shape[0] > self.max_img_size:
            img_u8 = cv2.resize(img_u8, (self.max_img_size, self.max_img_size), interpolation=cv2.INTER_AREA)

        return img_u8

    def computeSinofft(self, img_u8: np.ndarray) -> np.ndarray:
        r = radon(img_u8, self.theta)
        max_r = float(np.max(r))
        if max_r > 1e-8:
            r = r / max_r

        r = r.astype(np.float64)
        r = cv2.resize(
            r,
            (int(self.down_shape * r.shape[1]), int(self.down_shape * r.shape[0])),
            interpolation=cv2.INTER_AREA,
        )

        sinofft = np.abs(np.fft.fft(r, axis=0))
        return sinofft[: sinofft.shape[0] // 2, :]

    def saveLocalMap(self, img_u8: np.ndarray, timestamp_us: int) -> str:
        file_path = os.path.join(self.output_dir, f"{timestamp_us}.png")
        cv2.imwrite(file_path, img_u8)
        return file_path

    def findBestCandidate(self, query_entry: MapEntry):
        valid_candidates = [
            e for e in self.entries
            if (query_entry.timestamp_us - e.timestamp_us) > int(self.min_time_diff * 1_000_000)
        ]

        if not valid_candidates:
            return None

        query_norm = (query_entry.sinofft - np.mean(query_entry.sinofft)) / (np.std(query_entry.sinofft) + 1e-8)

        best_entry = None
        best_score = -1.0
        for entry in valid_candidates:
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
        self.candidate_pub.publish(out)

    def raplaceCallback(self, image_msg: Image, info_msg: LocalMapInfo):
        xy = np.array([info_msg.x, info_msg.y])
        if self.last_xy is None:
            self.cumulated_dists.append(0.0)
        else:
            dist = np.linalg.norm(xy - self.last_xy)
            self.cumulated_dists.append(dist + (self.cumulated_dists[-1]))
        self.last_xy = xy

        print('Cumulated distance: ', self.cumulated_dists[-1], ' m')


        #self.frame_counter += 1
        #if self.frame_counter % self.publish_every_n != 0:
        #    return

        #t0 = time.time()

        #try:
        #    local_map_float = self.imageMsgToFloat32(msg)
        #    img_u8 = self.preprocessForRadon(local_map_float)
        #    timestamp_us = self.timestamp2us(msg)
        #    image_path = self.saveLocalMap(img_u8, timestamp_us)
        #    sinofft = self.computeSinofft(img_u8)
        #except Exception as exc:
        #    self.get_logger().error(f"Failed to process incoming local map: {exc}")
        #    return

        #query_entry = MapEntry(
        #    index=len(self.entries),
        #    timestamp_us=timestamp_us,
        #    image_path=image_path,
        #    sinofft=sinofft,
        #)
        #self.entries.append(query_entry)

        #result = self.findBestCandidate(query_entry)
        #if result is None:
        #    return

        #best_entry, best_score, min_dist = result
        #self.publishCandidate(query_entry, best_entry, best_score, min_dist, msg)

        #elapsed_ms = (time.time() - t0) * 1000.0
        #self.get_logger().info(
        #    f"Published loop candidate q={query_entry.index} c={best_entry.index} score={best_score:.3f} min_dist={min_dist:.3f} ({elapsed_ms:.1f} ms)"
        #)


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
