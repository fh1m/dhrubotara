"""
Performance Monitoring
Real-time performance graphs using matplotlib
"""

import cv2
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class PerformanceMonitor:
    """Real-time performance graphs using matplotlib with trajectory tracking"""
    
    def __init__(self, history_length=100):
        self.fps_history = deque(maxlen=history_length)
        self.confidence_history = deque(maxlen=history_length)
        self.matches_history = deque(maxlen=history_length)
        self.ekf_vel_history = deque(maxlen=history_length)
        
        # Trajectory tracking (3D path)
        self.trajectory_x = deque(maxlen=history_length)
        self.trajectory_y = deque(maxlen=history_length)
        self.trajectory_z = deque(maxlen=history_length)
        
        # Initialize with zeros
        for _ in range(10):
            self.fps_history.append(0)
            self.confidence_history.append(0)
            self.matches_history.append(0)
            self.ekf_vel_history.append(0)
            self.trajectory_x.append(0)
            self.trajectory_y.append(0)
            self.trajectory_z.append(0)
    
    def update(self, fps, confidence, matches, ekf_vel_mag, dx=0, dy=0, dz=0):
        """Update performance metrics and trajectory"""
        self.fps_history.append(fps)
        self.confidence_history.append(confidence)
        self.matches_history.append(matches)
        self.ekf_vel_history.append(ekf_vel_mag)
        
        # Update trajectory (cumulative position)
        if len(self.trajectory_x) > 0:
            self.trajectory_x.append(self.trajectory_x[-1] + dx * 0.01)  # Scale for visualization
            self.trajectory_y.append(self.trajectory_y[-1] + dy * 0.01)
            self.trajectory_z.append(self.trajectory_z[-1] + dz * 0.01)
        else:
            self.trajectory_x.append(dx * 0.01)
            self.trajectory_y.append(dy * 0.01)
            self.trajectory_z.append(dz * 0.01)
    
    def draw_performance_panel(self, size=(400, 300)):
        """Draw performance graphs panel with ocean theme"""
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(size[0]/100, size[1]/100), dpi=100)
        fig.patch.set_facecolor('#2d231420')  # Ocean-inspired dark background
        
        # Time axis
        x = list(range(len(self.fps_history)))
        
        # Ocean color palette
        ocean_bg = '#372D1E'  # Deep ocean
        ocean_grid = '#4A3C28'  # Grid lines
        ocean_text = '#C8C8C8'  # Text
        
        # 1. FPS Graph (Green/Cyan gradient)
        ax1.plot(x, list(self.fps_history), color='#50C878', linewidth=2.5, alpha=0.9)
        ax1.fill_between(x, list(self.fps_history), alpha=0.2, color='#50C878')
        ax1.set_ylabel('FPS', color=ocean_text, fontsize=9, fontweight='bold')
        ax1.set_title('Frame Rate', color=ocean_text, fontsize=9, pad=8)
        ax1.set_ylim(0, 60)
        ax1.tick_params(colors=ocean_text, labelsize=7)
        ax1.grid(True, alpha=0.25, color=ocean_grid, linestyle='--')
        ax1.set_facecolor(ocean_bg)
        ax1.axhline(y=30, color='#FFA500', linestyle='--', linewidth=1.5, alpha=0.6)
        for spine in ax1.spines.values():
            spine.set_color(ocean_grid)
            spine.set_linewidth(1.5)
        
        # 2. Confidence Graph (Blue gradient)
        ax2.plot(x, list(self.confidence_history), color='#5B9BD5', linewidth=2.5, alpha=0.9)
        ax2.fill_between(x, list(self.confidence_history), alpha=0.2, color='#5B9BD5')
        ax2.set_ylabel('Confidence %', color=ocean_text, fontsize=9, fontweight='bold')
        ax2.set_title('Tracking Confidence', color=ocean_text, fontsize=9, pad=8)
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors=ocean_text, labelsize=7)
        ax2.grid(True, alpha=0.25, color=ocean_grid, linestyle='--')
        ax2.set_facecolor(ocean_bg)
        ax2.axhline(y=50, color='#FFA500', linestyle='--', linewidth=1.5, alpha=0.6)
        for spine in ax2.spines.values():
            spine.set_color(ocean_grid)
            spine.set_linewidth(1.5)
        
        # 3. Matches Graph (Cyan)
        ax3.plot(x, list(self.matches_history), color='#00CED1', linewidth=2.5, alpha=0.9)
        ax3.fill_between(x, list(self.matches_history), alpha=0.2, color='#00CED1')
        ax3.set_ylabel('Matches', color=ocean_text, fontsize=9, fontweight='bold')
        ax3.set_xlabel('Time (frames)', color=ocean_text, fontsize=8)
        ax3.set_title('Feature Matches', color=ocean_text, fontsize=9, pad=8)
        ax3.tick_params(colors=ocean_text, labelsize=7)
        ax3.grid(True, alpha=0.25, color=ocean_grid, linestyle='--')
        ax3.set_facecolor(ocean_bg)
        for spine in ax3.spines.values():
            spine.set_color(ocean_grid)
            spine.set_linewidth(1.5)
        
        # 4. EKF Velocity Graph (Magenta/Purple)
        ax4.plot(x, list(self.ekf_vel_history), color='#BA55D3', linewidth=2.5, alpha=0.9)
        ax4.fill_between(x, list(self.ekf_vel_history), alpha=0.2, color='#BA55D3')
        ax4.set_ylabel('EKF Vel (m/s)', color=ocean_text, fontsize=9, fontweight='bold')
        ax4.set_xlabel('Time (frames)', color=ocean_text, fontsize=8)
        ax4.set_title('EKF Velocity', color=ocean_text, fontsize=9, pad=8)
        ax4.tick_params(colors=ocean_text, labelsize=7)
        ax4.grid(True, alpha=0.25, color=ocean_grid, linestyle='--')
        ax4.set_facecolor(ocean_bg)
        for spine in ax4.spines.values():
            spine.set_color(ocean_grid)
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        # Resize to exact size
        img = cv2.resize(img, size)
        
        return img
    
    def draw_3d_trajectory(self, size=(400, 400)):
        """Draw 3D trajectory visualization"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D figure
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        fig.patch.set_facecolor('#2d231420')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#372D1E')
        
        # Get trajectory data
        x = list(self.trajectory_x)
        y = list(self.trajectory_y)
        z = list(self.trajectory_z)
        
        # Plot trajectory with color gradient (older = darker)
        n = len(x)
        if n > 1:
            for i in range(n-1):
                factor = i / n
                alpha = 0.3 + 0.7 * factor
                color = plt.cm.cool(factor)  # Color gradient
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                       color=color, linewidth=2, alpha=alpha)
            
            # Mark start and end
            ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, marker='o', 
                      edgecolors='white', linewidths=2, label='Start')
            ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=150, marker='*', 
                      edgecolors='white', linewidths=2, label='Current')
        
        # Labels and styling
        ax.set_xlabel('X Position', color='#C8C8C8', fontsize=9)
        ax.set_ylabel('Y Position', color='#C8C8C8', fontsize=9)
        ax.set_zlabel('Z Position', color='#C8C8C8', fontsize=9)
        ax.set_title('3D Trajectory Path', color='#C8C8C8', fontsize=11, pad=15)
        
        # Grid and ticks
        ax.grid(True, alpha=0.3, color='#4A3C28')
        ax.tick_params(colors='#C8C8C8', labelsize=7)
        
        # Set equal aspect ratio
        if n > 1:
            max_range = max(
                max(x) - min(x),
                max(y) - min(y),
                max(z) - min(z)
            ) / 2.0
            
            mid_x = (max(x) + min(x)) / 2.0
            mid_y = (max(y) + min(y)) / 2.0
            mid_z = (max(z) + min(z)) / 2.0
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set pane colors
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#4A3C28')
        ax.yaxis.pane.set_edgecolor('#4A3C28')
        ax.zaxis.pane.set_edgecolor('#4A3C28')
        
        # Legend
        if n > 1:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        # Resize to exact size
        img = cv2.resize(img, size)
        
        return img
