import cv2
import numpy as np

def detect_vanishing_point(frame):
    """Detect vanishing point using line intersections from edges"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=20)
    
    if lines is None:
        cv2.putText(frame, 'No lines detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, frame
    
    intersections = []
    height, width = frame.shape[:2]
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if abs(denom) > 1e-10:
                px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
                
                if -width <= px <= width*2 and -height <= py <= height*2:
                    intersections.append((int(px), int(py)))
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f'Lines: {len(lines)}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if not intersections:
        cv2.putText(frame, 'No intersections found', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return None, frame
    
    intersections = np.array(intersections)
    vanishing_point = np.median(intersections, axis=0).astype(int)
    
    cv2.circle(frame, tuple(vanishing_point), 15, (0, 0, 255), 3)
    cv2.circle(frame, tuple(vanishing_point), 5, (255, 255, 255), -1)
    
    cv2.line(frame, (vanishing_point[0] - 30, vanishing_point[1]), 
             (vanishing_point[0] + 30, vanishing_point[1]), (0, 0, 255), 2)
    cv2.line(frame, (vanishing_point[0], vanishing_point[1] - 30), 
             (vanishing_point[0], vanishing_point[1] + 30), (0, 0, 255), 2)
    
    cv2.putText(frame, f'VP: ({vanishing_point[0]}, {vanishing_point[1]})', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Intersections: {len(intersections)}', (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return vanishing_point, frame

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame")
            break
        
        vp, processed_frame = detect_vanishing_point(frame)
        
        cv2.imshow('Vanishing Point Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
