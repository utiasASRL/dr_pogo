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
    x_m: float
    y_m: float
    theta_rad: float
    scale: float
    num_matches: int
    reason: str
    warped_query: np.ndarray | None
    candidate_img: np.ndarray | None


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
            
        ##### FOR DEBUG
        self.debug_output_dir = os.path.join(self.package_share, "registration_debug")
        # if exists, clear the debug output directory
        if os.path.exists(self.debug_output_dir):
            for filename in os.listdir(self.debug_output_dir):
                file_path = os.path.join(self.debug_output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(self.debug_output_dir, exist_ok=True)

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
        q = query_img
        c = candidate_img

        if q.shape[0] > self.max_img_size:
            ratio = q.shape[0] / float(self.max_img_size)
            q = cv2.resize(q, (self.max_img_size, self.max_img_size))
            c = cv2.resize(c, (self.max_img_size, self.max_img_size))

        kp_q, des_q = self.sift_extractor.detectAndCompute(q, None)
        kp_c, des_c = self.sift_extractor.detectAndCompute(c, None)
        if des_q is None or des_c is None:
            return RegistrationResult(False, 0.0, 0.0, 0.0, 1.0, 0, "missing_descriptors", None, None)

        if ratio != 1.0:
            for kp in kp_q:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)
            for kp in kp_c:
                kp.pt = (kp.pt[0] * ratio, kp.pt[1] * ratio)

        matches = self.sift_matcher.knnMatch(des_q, des_c, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if (kp_q[m.queryIdx].octave & 255) != (kp_c[n.trainIdx].octave & 255):
                continue
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return RegistrationResult(
                False,
                0.0,
                0.0,
                0.0,
                1.0,
                len(good_matches),
                "insufficient_matches",
                None,
                None,
            )

        dst_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thr,
        )
        if M is None:
            return RegistrationResult(
                False,
                0.0,
                0.0,
                0.0,
                1.0,
                len(good_matches),
                "ransac_failed",
                None,
                None,
            )

        a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        c_m, d, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
        sx = np.hypot(a, c_m)
        sy = np.hypot(b, d)
        scale = 0.5 * (sx + sy)

        if abs(scale - 1.0) > self.max_scale_error:
            return RegistrationResult(
                False,
                0.0,
                0.0,
                0.0,
                scale,
                len(good_matches),
                "scale_rejected",
                None,
                None,
            )

        theta = float(np.arctan2(c_m, a))
        if resolution_m_per_px <= 0.0:
            resolution_m_per_px = 1.0

        cx = (original_shape[1] - 1.0) * 0.5
        cy = (original_shape[0] - 1.0) * 0.5
        center = np.array([cx, cy], dtype=np.float64)
        Rm = np.array([[a, b], [c_m, d]], dtype=np.float64)
        t = np.array([tx, ty], dtype=np.float64)
        t_center = t + Rm @ center - center

        x_m = float(t_center[0] * resolution_m_per_px)
        y_m = float(-t_center[1] * resolution_m_per_px)

        M_viz = M.copy()
        M_viz[:, -1] = M_viz[:, -1] / ratio
        warped_query = cv2.warpAffine(query_img, M_viz, (candidate_img.shape[1], candidate_img.shape[0]))

        if inliers is not None:
            inlier_ratio = float(np.mean(inliers))
            reason = f"ok_inlier_ratio_{inlier_ratio:.2f}"
        else:
            reason = "ok"

        return RegistrationResult(
            True,
            x_m,
            y_m,
            theta,
            scale,
            len(good_matches),
            reason,
            warped_query,
            candidate_img,
        )

    def publishResult(self, source_msg: LoopCandidate, result: RegistrationResult) -> None:
        if result.valid:
            pose_msg = LocalMapInfo()
            pose_msg.header = source_msg.header
            pose_msg.x = float(result.x_m)
            pose_msg.y = float(result.y_m)
            pose_msg.theta = float(result.theta_rad)
            pose_msg.resolution = float(source_msg.resolution)
            self.pose_pub.publish(pose_msg)

        if result.warped_query is not None and result.candidate_img is not None:
            left = cv2.cvtColor(result.warped_query, cv2.COLOR_GRAY2BGR)
            right = cv2.cvtColor(result.candidate_img, cv2.COLOR_GRAY2BGR)
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
            cv2.imwrite(os.path.join(self.debug_output_dir, f"debug_reg_q{source_msg.query_index}_c{source_msg.candidate_index}_{status}.png"), viz)

        if result.valid:
            self.get_logger().info(
                f"Registration accepted q={source_msg.query_index} c={source_msg.candidate_index}: "
                f"x={result.x_m:.2f} m, y={result.y_m:.2f} m, theta={result.theta_rad:.3f} rad, "
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
