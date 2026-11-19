"""
XFeat Feature Matching
Visual position estimation using XFeat feature matching
"""

import cv2
import torch
import numpy as np


def process_frame_for_xfeat(frame, device="cpu"):
    """Prepare frame for XFeat processing"""
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_tensor = torch.from_numpy(img_gray).float()[None, None] / 255.0
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    return img_tensor


def estimate_camera_guidance(mkpts_0, mkpts_1, min_matches=8):
    """
    Estimate camera movement guidance based on feature matches
    Returns: (dx, dy, dz_scale, confidence, homography_found)
    """
    if len(mkpts_0) < min_matches:
        return 0, 0, 1.0, 0.0, False

    try:
        H, mask = cv2.findHomography(mkpts_1, mkpts_0, cv2.RANSAC, 5.0)

        if H is None:
            return 0, 0, 1.0, 0.0, False

        tx = H[0, 2]
        ty = H[1, 2]

        sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        scale = (sx + sy) / 2.0

        confidence = np.sum(mask) / len(mask) * 100 if mask is not None else 0

        return tx, ty, scale, confidence, True

    except Exception as e:
        return 0, 0, 1.0, 0.0, False


def fuse_visual_imu_guidance(visual_dx, visual_dy, visual_dz, visual_conf, 
                              imu_data, alpha=0.7):
    """
    Fuse visual guidance with EKF-filtered IMU data using complementary filter
    
    alpha: weight for visual data (0.7 = 70% visual, 30% IMU/EKF)
    Returns: (fused_dx, fused_dy, fused_dz, fused_confidence, fusion_info)
    """
    attitude = imu_data.get_attitude()
    ekf = imu_data.get_ekf_data()
    
    # Check if data is fresh and EKF is healthy
    imu_fresh = attitude['connected'] and attitude['age'] < 1.0
    ekf_fresh = ekf['age'] < 1.0 and ekf['healthy']
    
    fusion_info = {
        'imu_used': imu_fresh,
        'ekf_used': ekf_fresh,
        'correction_applied': False,
        'ekf_vel_mag': 0.0
    }
    
    if not imu_fresh:
        # No IMU data, use visual only
        return visual_dx, visual_dy, visual_dz, visual_conf, fusion_info
    
    # IMU corrections based on attitude
    # If AUV is tilted, visual estimates need compensation
    roll = attitude['roll']
    pitch = attitude['pitch']
    
    # Rotation compensation (simplified - full implementation needs rotation matrices)
    # This is a basic correction for small angles
    dx_corrected = visual_dx * np.cos(roll) - visual_dy * np.sin(roll)
    dy_corrected = visual_dx * np.sin(roll) + visual_dy * np.cos(roll)
    
    fusion_info['correction_applied'] = True
    
    # Use EKF velocity if available (more stable than integrated acceleration)
    if ekf_fresh:
        ekf_vel_magnitude = np.sqrt(ekf['vel_x']**2 + ekf['vel_y']**2)
        fusion_info['ekf_vel_mag'] = ekf_vel_magnitude
        
        # Blend visual with EKF velocity direction
        # Higher visual confidence = trust visual more
        visual_weight = alpha * (visual_conf / 100.0)
        ekf_weight = 1.0 - visual_weight
        
        # Use EKF velocity to scale and validate visual estimates
        # Convert EKF velocity (NED frame) to camera frame approximation
        # This is simplified - proper transformation needs camera calibration
        ekf_dx = -ekf['vel_y'] * 100  # Rough pixel scaling
        ekf_dy = ekf['vel_z'] * 100
        
        # Complementary filter
        fused_dx = visual_weight * dx_corrected + ekf_weight * ekf_dx
        fused_dy = visual_weight * dy_corrected + ekf_weight * ekf_dy
        fused_dz = visual_dz  # Keep visual depth estimate
        
        # Boost confidence if EKF and visual agree on motion direction
        dx_agree = (np.sign(dx_corrected) == np.sign(ekf_dx)) if abs(ekf_dx) > 5 else True
        dy_agree = (np.sign(dy_corrected) == np.sign(ekf_dy)) if abs(ekf_dy) > 5 else True
        
        if dx_agree and dy_agree and ekf_vel_magnitude > 0.1:
            fused_confidence = min(100, visual_conf * 1.3)
        else:
            fused_confidence = visual_conf
    else:
        # No EKF, use attitude-corrected visual only
        fused_dx = dx_corrected
        fused_dy = dy_corrected
        fused_dz = visual_dz
        fused_confidence = visual_conf
    
    return fused_dx, fused_dy, fused_dz, fused_confidence, fusion_info
