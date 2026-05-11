#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import yaml
import torch
import time
from scipy.spatial.transform import Rotation as R

from dr_pogo.msg import LoopCandidate


@dataclass
class RegistrationResult:
    valid: bool
    pose: np.ndarray
    scale: float
    num_matches: int
    reason: str
    viz_image: Optional[np.ndarray]

def poseToxytheta(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([x, y, theta])

def xythetaToPose(xytheta):
    c_rot = np.cos(xytheta[2])
    s_rot = np.sin(xytheta[2])
    pose = np.eye(4)
    pose[0, 0] = c_rot
    pose[0, 1] = -s_rot
    pose[1, 0] = s_rot
    pose[1, 1] = c_rot
    pose[0, 3] = xytheta[0]
    pose[1, 3] = xytheta[1]
    return pose

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

class LocalMapRegistrator:
    def __init__(self, source, target, res, xytheta_init=np.array([0, 0, 0])):

        # Check the input shapes match and that the nb of collumn and rows are odd
        if source.shape[0] != target.shape[0] or source.shape[1] != target.shape[1] or source.shape[0] % 2 == 0 or source.shape[1] % 2 == 0:
            raise ValueError("Source and target images must have the same shape and odd dimensions")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimisation_first_step = 0.1

        with torch.no_grad():
            self.source = torch.tensor(source, device=self.device).float()
            self.target = torch.tensor(target, device=self.device).float()
            self.res = torch.tensor(res, device=self.device).float()
            self.xytheta_init = torch.tensor(xytheta_init, device=self.device).float()

            # Create the cartesian coordinates that correspond to each pixel of the images
            self.cartesian_coords = torch.zeros((self.source.shape[0], self.source.shape[1], 2, 1), device=self.device)
            self.cartesian_coords[:, :, 0, 0] = -((torch.arange(self.source.shape[0], device=self.device)- (self.source.shape[0] // 2)).float() * self.res).reshape((-1, 1))
            self.cartesian_coords[:, :, 1, 0] = ((torch.arange(self.source.shape[1], device=self.device)- (self.source.shape[1] // 2)).float() * self.res).reshape((1, -1))

        #self.costFunctionAndJacobian(self.xytheta_init, with_jac=True)
        #self.gridSearchInitialization([[-2,2],[-2,2], [np.radians(-1.0), np.radians(1.0)]], nb_steps=3)
        #self.register(nb_iter=1, verbose=False)
        #self.getRegistrationScore()


    def cartToImageID_(self, xy):
        with torch.no_grad():
            out = torch.empty_like(xy, device=self.device)
            out[:,:,0,0] = (xy[:,:,0,0] / (-self.res)) + (self.source.shape[0] // 2)
            out[:,:,1,0] = (xy[:,:,1,0] / (self.res)) + (self.source.shape[1] // 2)
            gradient = torch.tensor([[-1.0/self.res, 0], [0, 1.0/self.res]], device=self.device).reshape((1,1,2,2))
            return out, gradient


    def transformSource_(self, xytheta):
        with torch.no_grad():
            # If xytheta is a numpy array, convert it to a torch tensor
            if isinstance(xytheta, np.ndarray):
                xytheta = torch.tensor(xytheta, device=self.device).float()

            c_rot = torch.cos(xytheta[2])
            s_rot = torch.sin(xytheta[2])
            rot_mat_T = torch.tensor([[c_rot, -s_rot], [s_rot, c_rot]], device=self.device).T.reshape((1,1, 2, 2))
            pos = xytheta[:2].reshape((1,1, 2, 1)).to(self.device)

            # Transform the cartesian coordinates
            cartesian_coords_transformed = rot_mat_T @ self.cartesian_coords - rot_mat_T @ pos

            # Convert the cartesian coordinates to image coordinates
            ids, gradient = self.cartToImageID_(cartesian_coords_transformed)

            # Get the interpolated source image
            source_interp = self.bilinearInterpolation_(self.source, ids.squeeze(), with_jac=False)

            # Residuals
            residuals = source_interp * self.source

            return source_interp, residuals


    def bilinearInterpolation_(self, im, az_r, with_jac = False):
        with torch.no_grad():
            az0 = torch.floor(az_r[:, :, 0]).int()
            az1 = az0 + 1
            
            r0 = torch.floor(az_r[:, :, 1]).int()
            r1 = r0 + 1

            az0 = torch.clamp(az0, 0, im.shape[0]-1)
            az1 = torch.clamp(az1, 0, im.shape[0]-1)
            r0 = torch.clamp(r0, 0, im.shape[1]-1)
            r1 = torch.clamp(r1, 0, im.shape[1]-1)
            az_r[:,:,0] = torch.clamp(az_r[:,:,0], 0, im.shape[0]-1)
            az_r[:,:,1] = torch.clamp(az_r[:,:,1], 0, im.shape[1]-1)
            
            Ia = im[ az0, r0 ]
            Ib = im[ az1, r0 ]
            Ic = im[ az0, r1 ]
            Id = im[ az1, r1 ]
            
            local_1_minus_r = (r1.float()-az_r[:, :, 1])
            local_r = (az_r[:, :, 1]-r0.float())
            local_1_minus_az = (az1.float()-az_r[:, :, 0])
            local_az = (az_r[:, :, 0]-az0.float())
            wa = local_1_minus_az * local_1_minus_r
            wb = local_az * local_1_minus_r
            wc = local_1_minus_az * local_r
            wd = local_az * local_r

            img_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

            if not with_jac:
                return img_interp
            else:
                d_I_d_az_r = torch.empty((az_r.shape[0], az_r.shape[1], 1, 2), device=self.device)
                d_I_d_az_r[:, :, 0, 0] = (Ib - Ia)*local_1_minus_r + (Id - Ic)*local_r
                d_I_d_az_r[:, :, 0, 1] = (Ic - Ia)*local_1_minus_az + (Id - Ib)*local_az
                return img_interp, d_I_d_az_r
        


    #@torch.compile
    def register(self, nb_iter=20, cost_tol=1e-6, step_tol=1e-6, verbose=False):
        with torch.no_grad():
            # The gradient ascent keep track of the last increasing state and gradient
            # Thus, if the cost function decreases, we go back to the last increasing
            # state and reduce the step size
            state = self.xytheta_init.clone().to(self.device).float()
            first_cost = torch.tensor(np.inf, device=self.device)
            prev_cost = first_cost
            first_quantum = self.optimisation_first_step
            step_quantum = first_quantum
            last_increasing_state = state.clone()
            last_increasing_grad = torch.zeros_like(state)
            for i in torch.arange(nb_iter, device=self.device):
                
                res, jac = self.costFunctionAndJacobian(state)

                #grad = 3*torch.sum(res.flatten().unsqueeze(-1)**2 * jac.reshape((-1,jac.shape[-1])), 0)
                #cost = torch.sum((res**3).flatten())
                grad = torch.sum(jac, 0)
                cost = torch.sum((res).flatten())

                if i == 0:
                    last_increasing_grad = grad.clone()
                else:
                    if cost < prev_cost:
                        state = last_increasing_state.clone()
                        grad = last_increasing_grad.clone()
                        step_quantum = step_quantum / 2
                    else:
                        last_increasing_state = state.clone()
                        last_increasing_grad = grad.clone()

                grad_norm = torch.linalg.norm(grad)

                if step_quantum < 1e-5:
                    break


                if grad_norm < 1e-9:
                    break
                step = (grad / grad_norm) * step_quantum
                
                state += step

                step_norm = torch.linalg.norm(step)
                cost_change = cost - prev_cost

                if i == 0:
                    first_cost = cost
                
                # Print iter cost step_norm cost_change with 3 decimals and scientific notation
                if verbose:
                    print("Iter: ", i, " - Cost: ", "{:.3e}".format(cost), " - Step norm: ", "{:.3e}".format(step_norm), " - Cost change: ", "{:.3e}".format(cost_change))

                if step_norm < step_tol:
                    break

                if torch.abs(cost_change/cost) < cost_tol:
                    break
                prev_cost = cost

            state_np = state.detach().cpu().numpy()

            self.xytheta_init = state.clone()

            return state_np


    #@torch.compile
    def costFunctionAndJacobian(self, xytheta, with_jac=True):
        with torch.no_grad():
            # Get the rotation matrix
            c_rot = torch.cos(xytheta[2])
            s_rot = torch.sin(xytheta[2])
            rot_mat = torch.tensor([[c_rot, -s_rot], [s_rot, c_rot]], device=self.device).reshape((1,1, 2, 2))
            pos = xytheta[:2].reshape((1,1, 2, 1)).to(self.device)

            # Transform the cartesian coordinates
            cartesian_coords_transformed = rot_mat @ self.cartesian_coords

            if with_jac:
                d_cartesian_coords_transformed_d_state = torch.zeros((self.cartesian_coords.shape[0], self.cartesian_coords.shape[1], 2, 3), device=self.device)
                d_cartesian_coords_transformed_d_state[:,:,0, 0] = 1
                d_cartesian_coords_transformed_d_state[:,:,1, 1] = 1
                d_cartesian_coords_transformed_d_state[:,:,0, 2] = -cartesian_coords_transformed[:,:,1,0]
                d_cartesian_coords_transformed_d_state[:,:,1, 2] = cartesian_coords_transformed[:,:,0,0]
            cartesian_coords_transformed += pos

            # Convert the cartesian coordinates to image coordinates
            ids, gradient = self.cartToImageID_(cartesian_coords_transformed)
            if with_jac:
                d_ids_dstate = gradient @ d_cartesian_coords_transformed_d_state


            # Get the interpolated source image
            if with_jac:
                source_interp, d_source_interp = self.bilinearInterpolation_(self.target, ids.squeeze(), with_jac=True)
                d_source_interp = d_source_interp @ d_ids_dstate
            else:
                source_interp = self.bilinearInterpolation_(self.target, ids.squeeze(), with_jac=False)

            # Residuals
            residuals = source_interp * self.source

            ## For debug, visualize the source_interp and the target
            #import matplotlib.pyplot as plt
            #fig, axs = plt.subplots(1, 2)
            #axs[0].imshow(self.source.cpu().numpy(), cmap='gray')
            #axs[0].imshow(source_interp.cpu().numpy(), cmap='hot', alpha=0.5)
            #axs[0].set_title('Target and overlay')
            #axs[1].imshow(self.source.cpu().numpy(), cmap='gray')
            #axs[1].set_title('Source')
            #plt.show()

            if with_jac:
                gradient = self.source.unsqueeze(-1).unsqueeze(-1) @ d_source_interp
                return residuals.flatten(), gradient.reshape((-1,3))
            else:
                return residuals.flatten()




    def displayOverlay(self, show=True):
        # Display the overlay of the source and target images
        source_interp, _ = self.transformSource_(self.xytheta_init)

        target_np = self.target.cpu().numpy()
        source_interp_np = source_interp.cpu().numpy()
        
        def normalizeToUint8(arr):
            a, b = arr.min(), arr.max()
            if b > a:
                arr = (arr - a) / (b - a)
            return (arr * 255).astype(np.uint8)

        target_uint8 = normalizeToUint8(target_np)
        source_uint8 = normalizeToUint8(source_interp_np)

        # Grayscale target → BGR
        target_bgr = cv2.cvtColor(target_uint8, cv2.COLOR_GRAY2BGR)

        # 'hot' colormap on source_interp
        source_hot = cv2.applyColorMap(source_uint8, cv2.COLORMAP_HOT)

        # alpha=0.5 blend: result = target * 0.5 + source_hot * 0.5
        overlay = cv2.addWeighted(target_bgr, 0.5, source_hot, 0.5, 0)

        if show:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10,10))
            # Swap the color channels from BGR to RGB for correct display in matplotlib
            overlay_dis = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            plt.imshow(overlay_dis)
            plt.title('Overlay of target (grayscale) and source (hot colormap)')
            plt.axis('off')
            plt.show()

        return overlay

    #@torch.compile
    def getRegistrationScore(self):
        # Compute the registration
        with torch.no_grad():
            residuals = self.costFunctionAndJacobian(self.xytheta_init, with_jac=False)
            return torch.sum(residuals) / torch.sum(self.target**2)
    
    #@torch.compile
    def gridSearchInitialization(self, search_ranges, nb_steps):
        with torch.no_grad():
            xs = torch.linspace(search_ranges[0][0], search_ranges[0][1], nb_steps, device=self.device) + self.xytheta_init[0]
            ys = torch.linspace(search_ranges[1][0], search_ranges[1][1], nb_steps, device=self.device) + self.xytheta_init[1]
            thetas = torch.linspace(search_ranges[2][0], search_ranges[2][1], nb_steps, device=self.device) + self.xytheta_init[2]
            best_cost = -np.inf
            best_state = self.xytheta_init.clone()
            for x in xs:
                for y in ys:
                    for theta in thetas:
                        cost = self.costFunctionAndJacobian(torch.tensor([x, y, theta], device=self.device), with_jac=False)
                        cost = torch.sum(cost)
                        if cost > best_cost:
                            best_cost = cost
                            best_state = torch.tensor([x, y, theta], device=self.device)
            self.xytheta_init = best_state

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
        self.pose_pub = self.create_publisher(TransformStamped, "registration_relative_pose", 20)
        self.viz_pub = self.create_publisher(Image, "registration_debug_image", 20)

        self.get_logger().info(
            "Registration node started. Subscribed to 'raplace_loop_candidate', publishing 'registration_relative_pose' and 'registration_debug_image'."
        )

        self.counter = 0

    def resolvePath(self, path: str) -> Optional[str]:
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

        if result.valid:
            t1 = time.time()
            result = self.refineRegistration(candidate_img, query_img, float(msg.resolution), result)
            t2 = time.time()
            self.get_logger().info(
                f"Refinement for candidate q={msg.query_index} c={msg.candidate_index} took {t2-t1:.2f} seconds. Final reason: {result.reason}."
            )

        if result.valid:
            self.get_logger().info(
                f"Candidate q={msg.query_index} c={msg.candidate_index} registered successfully: "
                f"pose={poseToxytheta(result.pose)}, scale={result.scale:.3f}, matches={result.num_matches}, reason={result.reason}."
            )
        else:
            self.get_logger().info(
                f"Candidate q={msg.query_index} c={msg.candidate_index} registration failed: "
                f"scale={result.scale:.3f}, matches={result.num_matches}, reason={result.reason}."
            )
            return

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
            return RegistrationResult(False, None, 1.0, 0, "missing_descriptors", None)

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
            return RegistrationResult(False, None, 1.0, 0, "insufficient_matches", None)

        dst_pts = np.float32([kp_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thr,
        )
        if M is None:
            return RegistrationResult(False, None, 1.0, len(good_matches), "ransac_failed", None)
        
        if inliers is None or np.sum(inliers) < 3:
            return RegistrationResult(False, None, 1.0, len(good_matches), "insufficient_inliers", None)


        pose, scale = affineToPoseAndScale(M, resolution_m_per_px, original_shape)


        if abs(scale - 1.0) > self.max_scale_error:
            return RegistrationResult(False, None, scale, len(good_matches), "scale_error", None)


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
            None
        )
    
    def refineRegistration(self, img_i, img_j, res, reg_result):
            if img_i is None or img_j is None:
                print(f"Skipping registration due to missing images.")
                return
            img_i = cv2.GaussianBlur(img_i, (5, 5), 0)
            img_j = cv2.GaussianBlur(img_j, (5, 5), 0)

            # Resize the images
            img_i_small = cv2.resize(img_i, (img_i.shape[1]//4 + 1, img_i.shape[0]//4 + 1))
            img_j_small = cv2.resize(img_j, (img_j.shape[1]//4 + 1, img_j.shape[0]//4 + 1))
            res_small = res * 4
            # Add Gaussian blur to the images
            img_i_small = cv2.GaussianBlur(img_i_small, (5, 5), 0)
            img_j_small = cv2.GaussianBlur(img_j_small, (5, 5), 0)

            # Perform fine registration using "gp_doppler"
            local_map_registrator = LocalMapRegistrator(img_j_small, img_i_small, res_small, poseToxytheta(reg_result.pose))
            local_map_registrator.gridSearchInitialization([[-2,2],[-2,2], [np.radians(-1.0), np.radians(1.0)]], nb_steps=3)

            state = local_map_registrator.register(nb_iter=20, verbose=False, step_tol=1e-4)

            # Resize the images
            img_i_small = cv2.resize(img_i, (img_i.shape[1]//2 + 1, img_i.shape[0]//2 + 1))
            img_j_small = cv2.resize(img_j, (img_j.shape[1]//2 + 1, img_j.shape[0]//2 + 1))
            res_small = res * 2
            # Add Gaussian blur to the images
            img_i_small = cv2.GaussianBlur(img_i_small, (5, 5), 0)
            img_j_small = cv2.GaussianBlur(img_j_small, (5, 5), 0)
            local_map_registrator = LocalMapRegistrator(img_j_small, img_i_small, res_small, state)
            state = local_map_registrator.register(nb_iter=40, verbose=False, step_tol=1e-4)
            x, y, theta = state

            reg_score = local_map_registrator.getRegistrationScore()
            if reg_score > 0.5:
                new_result = RegistrationResult(
                    True,
                    xythetaToPose(state),
                    reg_result.scale,
                    reg_result.num_matches,
                    f"refined_{reg_result.reason}",
                    local_map_registrator.displayOverlay(show=False)
                )
            else:
                new_result = RegistrationResult(
                    False,
                    reg_result.pose,
                    reg_result.scale,
                    reg_result.num_matches,
                    f"refinement_failed_score_{reg_score:.2f}",
                    None
                )
            return new_result


    def publishResult(self, source_msg: LoopCandidate, result: RegistrationResult) -> None:
        if result.valid:
            x_m = result.pose[0, 3]
            y_m = result.pose[1, 3]
            theta_rad = np.arctan2(result.pose[1, 0], result.pose[0, 0])
            pose_msg = TransformStamped()
            pose_msg.header = source_msg.header
            pose_msg.header.frame_id = str(source_msg.candidate_time)
            pose_msg.child_frame_id = str(source_msg.query_time)
            pose_msg.transform.translation.x = x_m
            pose_msg.transform.translation.y = y_m
            pose_msg.transform.translation.z = 0.0
            quat = R.from_matrix(result.pose[:3, :3]).as_quat()
            pose_msg.transform.rotation.x = quat[0]
            pose_msg.transform.rotation.y = quat[1]
            pose_msg.transform.rotation.z = quat[2]
            pose_msg.transform.rotation.w = quat[3]
            self.pose_pub.publish(pose_msg)


            if self.viz_pub.get_subscription_count() > 0 and result.viz_image is not None:
                viz = result.viz_image.copy()
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
