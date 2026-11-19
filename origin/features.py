"""
XFeat Real-time Webcam Feature Detection with Visual Feedback
Displays detected keypoints with color-coded scores and FPS counter
"""

import cv2
import torch
import numpy as np
import time
import sys
sys.path.insert(0, 'accelerated_features')
from modules.xfeat import XFeat


def draw_keypoints_with_colors(image, keypoints, scores, radius=3):
    """Draw keypoints with color-coded confidence scores"""
    vis_image = image.copy()
    
    if len(keypoints) == 0:
        return vis_image
    
    # Normalize scores to 0-1 range for color mapping
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    for kp, score in zip(keypoints, scores_norm):
        x, y = int(kp[0]), int(kp[1])
        
        # Color from green (low score) to red (high score)
        color = (
            int(255 * (1 - score)),  # Blue
            int(255 * score),         # Green  
            int(255 * score)          # Red
        )
        
        # Draw filled circle for keypoint
        cv2.circle(vis_image, (x, y), radius, color, -1)
        # Draw outline for better visibility
        cv2.circle(vis_image, (x, y), radius + 1, (255, 255, 255), 1)
    
    return vis_image


def draw_info_overlay(image, num_features, fps, top_k=4096):
    """Draw information overlay on the image"""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Semi-transparent dark background for text
    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Draw text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'FPS: {fps:.1f}', (20, 40), font, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f'Features: {num_features}/{top_k}', (20, 70), font, 0.7, (255, 255, 0), 2)
    cv2.putText(image, 'Press Q to quit', (20, 100), font, 0.6, (255, 255, 255), 1)
    
    # Color legend
    legend_y = height - 60
    cv2.putText(image, 'Score:', (20, legend_y), font, 0.5, (255, 255, 255), 1)
    
    # Low score (green/cyan)
    cv2.circle(image, (90, legend_y - 5), 5, (255, 128, 0), -1)
    cv2.putText(image, 'Low', (105, legend_y), font, 0.4, (255, 255, 255), 1)
    
    # High score (red/yellow)  
    cv2.circle(image, (170, legend_y - 5), 5, (0, 255, 255), -1)
    cv2.putText(image, 'High', (185, legend_y), font, 0.4, (255, 255, 255), 1)
    
    return image


def main():
    print("Initializing XFeat...")
    
    # Initialize XFeat
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    xfeat = XFeat()
    if device == 'cuda':
        xfeat = xfeat.cuda()
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution (optional - adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting feature detection... Press 'q' to quit")
    
    # FPS calculation
    fps_history = []
    top_k = 4096
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Prepare input for XFeat (convert BGR to RGB and normalize)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            if device == 'cuda':
                frame_tensor = frame_tensor.cuda()
            
            # Detect and compute features
            with torch.no_grad():
                output = xfeat.detectAndCompute(frame_tensor, top_k=top_k)[0]
            
            # Extract keypoints and scores
            keypoints = output['keypoints'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            
            # Draw keypoints on frame
            vis_frame = draw_keypoints_with_colors(frame, keypoints, scores, radius=3)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Draw info overlay
            vis_frame = draw_info_overlay(vis_frame, len(keypoints), avg_fps, top_k)
            
            # Display the frame
            cv2.imshow('XFeat - Real-time Feature Detection', vis_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
        print("Done!")


if __name__ == '__main__':
    main()
