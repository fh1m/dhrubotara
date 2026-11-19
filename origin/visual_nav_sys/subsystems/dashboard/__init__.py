"""Dashboard visualization subsystem"""
from .performance_viz import PerformanceMonitor
from .imu_viz import draw_imu_raw_panel, draw_ekf_panel, draw_fusion_panel
from .horizon_viz import draw_artificial_horizon
from .guidance_viz import create_slick_3d_guidance, draw_optical_flow_vectors
# Use the new modern dashboard
from .modern_dashboard import draw_unified_window, switch_tab

__all__ = [
    'PerformanceMonitor',
    'draw_imu_raw_panel',
    'draw_ekf_panel',
    'draw_fusion_panel',
    'draw_artificial_horizon',
    'create_slick_3d_guidance',
    'draw_optical_flow_vectors',
    'draw_unified_window',
    'switch_tab'
]
