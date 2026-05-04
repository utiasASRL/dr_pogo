#!/usr/bin/env python3

import os
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
import yaml

from dr_pogo.msg import LocalMapInfo, LoopCandidate


@dataclass
class RegistrationResult:
    valid: bool
    pose: np.ndarray
    scale: float
    num_matches: int
    reason: str
    warped_query: np.ndarray | None
    candidate_img: np.ndarray | None

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


class RegistrationNode(Node):
    def __init__(self) -> None:
        super().__init__("registration_node")

        config_file_path = "config/config_registration.yaml"
        self.package_share = ""
        if not os.path.isfile(config_file_path):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.package_share = os.path.join(base_path, "share", "dr_pogo")
            config_file_path = os.path.join(self.package_share, config_file_path)
        with open(config_file_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            

        self.lowe_ratio = float(cfg["lowe_ratio"])
        self.ransac_thr = float(cfg["ransac_thr"])
        self.max_img_size = int(cfg["max_img_size"])

        self.max_scale_error = 0.05
        self.sift_extractor = cv2.SIFT_create(
            nfeatures=0,
            contrastThreshold=0.02,
            edgeThreshold=20,
            sigma=2.5,
        )
        self.sift_matcher = cv2.BFMatcher()
        self.bridge = CvBridge()

        self.candidate_sub = self.create_subscription(
            LoopCandidate,
            "raplace_loop_candidate",
            self.candidateCallback,
            20,
        )
        self.pose_pub = self.create_publisher(LocalMapInfo, "registration_relative_pose", 20)
        self.viz_pub = self.create_publisher(Image, "registration_debug_image", 20)

        self.get_logger().info(
            "Registration node started. Subscribed to 'raplace_loop_candidate', publishing 'registration_relative_pose' and 'registration_debug_image'."
        )

        self.counter = 0

    def resolvePath(self, path: str) -> str | None:
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        if os.path.isfile(path):
            return os.path.abspath(path)
        if self.package_share:
            candidate = os.path.join(self.package_share, path)
            if os.path.isfile(candidate):
                return candidate
        return None

    def candidateCallback(self, msg: LoopCandidate) -> None:
        query_path = self.resolvePath(msg.query_image_path)
        candidate_path = self.resolvePath(msg.candidate_image_path)

        if query_path is None or candidate_path is None:
            self.get_logger().warn(
                f"Skipping candidate q={msg.query_index} c={msg.candidate_index}: image path not found "
                f"(query='{msg.query_image_path}', candidate='{msg.candidate_image_path}')."
            )
            return

        query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        candidate_img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None or candidate_img is None:
            self.get_logger().warn(
                f"Skipping candidate q={msg.query_index} c={msg.candidate_index}: failed to read image files."
            )
            return

        result = self.estimateRelativePose(query_img, candidate_img, float(msg.resolution))
        self.publishResult(msg, result)




    def estimateRelativePose(
        self,
        query_img: np.ndarray,
        candidate_img: np.ndarray,
        resolution_m_per_px: float,
    ) -> RegistrationResult:
        original_shape = query_img.shape
        ratio = 1.0
        img2 = query_img.copy()
        img1 = candidate_img.copy()

        if img2.shape[0] > self.max_img_size:
            ratio = img2.shape[0] / float(self.max_img_size)
            img2 = cv2.resize(img2, (self.max_img_size, self.max_img_size))
            img1 = cv2.resize(img1, (self.max_img_size, self.max_img_size))

        kp_2, des_2 = self.sift_extractor.detectAndCompute(img2, None)
        kp_1, des_1 = self.sift_extractor.detectAndCompute(img1, None)
        if des_2 is None or des_1 is None:
            return RegistrationResult(False, None, 1.0, 0, "missing_descriptors", None, None)

        if ratio != 1.0:
            for kp in kp_2:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)
            for kp in kp_1:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)

        matches = self.sift_matcher.knnMatch(des_1, des_2, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if (kp_1[m.queryIdx].octave & 255) != (kp_2[n.trainIdx].octave & 255):
                continue
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return RegistrationResult(False, None, 1.0, 0, "insufficient_matches", None, None)

        dst_pts = np.float32([kp_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thr,
        )
        if M is None:
            return RegistrationResult(False, None, 1.0, len(good_matches), "ransac_failed", None, None)
        
        if inliers is None or np.sum(inliers) < 3:
            return RegistrationResult(False, None, 1.0, len(good_matches), "insufficient_inliers", None, None)


        pose, scale = affineToPoseAndScale(M, resolution_m_per_px, original_shape)


        if abs(scale - 1.0) > self.max_scale_error:
            return RegistrationResult(False, None, scale, len(good_matches), "scale_error", None, None)


        M_viz = M.copy()
        M_viz[:, -1] = M_viz[:, -1] / ratio
        img2_warped = cv2.warpAffine(img2, M_viz, (img1.shape[1], img1.shape[0]))

        if inliers is not None:
            inlier_ratio = float(np.mean(inliers))
            reason = f"ok_inlier_ratio_{inlier_ratio:.2f}"
        else:
            reason = "ok"

        self.counter += 1

        return RegistrationResult(
            True,
            pose,
            scale,
            len(good_matches),
            reason,
            img2_warped,
            img1,
        )

    def publishResult(self, source_msg: LoopCandidate, result: RegistrationResult) -> None:
        if result.valid:
            x_m = result.pose[0, 3]
            y_m = result.pose[1, 3]
            theta_rad = np.arctan2(result.pose[1, 0], result.pose[0, 0])
            pose_msg = LocalMapInfo()
            pose_msg.header = source_msg.header
            pose_msg.x = float(x_m)
            pose_msg.y = float(y_m)
            pose_msg.theta = float(theta_rad)
            pose_msg.resolution = float(source_msg.resolution)
            self.pose_pub.publish(pose_msg)


            if self.viz_pub.get_subscription_count() > 0 and result.warped_query is not None and result.candidate_img is not None:
                left = cv2.cvtColor(result.candidate_img, cv2.COLOR_GRAY2BGR)
                right = cv2.cvtColor(result.warped_query, cv2.COLOR_GRAY2BGR)
                viz = np.hstack((left, right))
                status = "ACCEPTED" if result.valid else "REJECTED"
                text = (
                    f"{status} | q={source_msg.query_index} c={source_msg.candidate_index} "
                    f"| m={result.num_matches} | s={result.scale:.3f} | {result.reason}"
                )
                cv2.putText(
                    viz,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if result.valid else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                image_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
                image_msg.header = source_msg.header
                self.viz_pub.publish(image_msg)

            self.get_logger().info(
                f"Registration accepted q={source_msg.query_index} c={source_msg.candidate_index}: "
                f"x={x_m:.2f} m, y={y_m:.2f} m, theta={theta_rad:.3f} rad, "
                f"matches={result.num_matches}, scale={result.scale:.3f}"
            )
        else:
            self.get_logger().info(
                f"Registration rejected q={source_msg.query_index} c={source_msg.candidate_index}: "
                f"reason={result.reason}, matches={result.num_matches}, scale={result.scale:.3f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegistrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
