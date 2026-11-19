"""
Visual Navigation System
XFeat + YOLOv11 + Optical Flow + Pixhawk IMU Sensor Fusion
Complete hybrid navigation system with attitude and motion data from Pixhawk 2.4.8

Guidance sources (prioritized and fused):
1. Pixhawk IMU/Gyro (attitude, angular velocity, acceleration)
2. XFeat homography (visual position estimation)
3. Object-based tracking (semantic fallback)
4. Optical Flow (motion continuity)

Sensor Fusion: Complementary filter combining visual and inertial data
"""

__version__ = "1.0.0"
__author__ = "AUV Team"

from .core.system import VisualNavSystem

__all__ = ['VisualNavSystem']
