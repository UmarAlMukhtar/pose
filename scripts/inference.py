"""
Simple Inference Script for Waste Throwing Detection

This script provides an easy way to test your trained model on new videos or live camera feed.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import json


class WasteThrowingDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Initialize the waste throwing detector
        
        Args:
            model_path (str): Path to trained YOLO11 pose model
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Keypoint names for visualization
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Throwing-related keypoints for analysis
        self.throwing_keypoints = [5, 6, 7, 8, 9, 10, 11, 12]  # shoulders, elbows, wrists, hips
        
        # Skeleton connections for visualization
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    
    def analyze_throwing_pose(self, keypoints, confidences):
        """
        Analyze if detected pose indicates throwing motion
        
        Args:
            keypoints (np.array): Detected keypoints [17, 2]
            confidences (np.array): Keypoint confidences [17]
            
        Returns:
            dict: Analysis results including throwing score
        """
        if len(keypoints) < 17:
            return {'throwing_score': 0, 'indicators': []}
        
        indicators = []
        
        # Check if key keypoints are visible
        key_visible = all(confidences[i] > self.conf_threshold for i in [5, 6, 9, 10])
        if not key_visible:
            return {'throwing_score': 0, 'indicators': ['Key keypoints not visible']}
        
        # 1. Arm elevation - check if wrists are above shoulders
        left_wrist_elevated = keypoints[9, 1] < keypoints[5, 1]  # Y coord (lower is higher)
        right_wrist_elevated = keypoints[10, 1] < keypoints[6, 1]
        
        if left_wrist_elevated or right_wrist_elevated:
            indicators.append('Arm elevation detected')
        
        # 2. Arm extension - check elbow-wrist distance
        left_arm_extended = np.linalg.norm(keypoints[7] - keypoints[9]) > 0.1
        right_arm_extended = np.linalg.norm(keypoints[8] - keypoints[10]) > 0.1
        
        if left_arm_extended or right_arm_extended:
            indicators.append('Arm extension detected')
        
        # 3. Body orientation - check shoulder alignment
        shoulder_diff = abs(keypoints[5, 1] - keypoints[6, 1])
        if shoulder_diff < 0.05:  # Shoulders roughly aligned
            indicators.append('Stable body position')
        
        # 4. Hip stability - check hip alignment  
        if confidences[11] > self.conf_threshold and confidences[12] > self.conf_threshold:
            hip_diff = abs(keypoints[11, 1] - keypoints[12, 1])
            if hip_diff < 0.1:
                indicators.append('Hip stability')
        
        # Calculate throwing score
        throwing_score = len(indicators) / 4.0  # Normalize to 0-1
        
        return {
            'throwing_score': throwing_score,
            'indicators': indicators,
            'arm_elevation': left_wrist_elevated or right_wrist_elevated,
            'arm_extension': left_arm_extended or right_arm_extended
        }
    
    def draw_pose(self, img, keypoints, confidences):
        """Draw pose keypoints and skeleton on image"""
        h, w = img.shape[:2]
        
        # Convert normalized coordinates to pixels
        kpts_px = keypoints * np.array([w, h])
        
        # Draw skeleton connections
        for connection in self.skeleton:
            kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-based
            
            if (kp1_idx < len(confidences) and kp2_idx < len(confidences) and
                confidences[kp1_idx] > self.conf_threshold and confidences[kp2_idx] > self.conf_threshold):
                
                pt1 = tuple(map(int, kpts_px[kp1_idx]))
                pt2 = tuple(map(int, kpts_px[kp2_idx]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (kpt, conf) in enumerate(zip(kpts_px, confidences)):
            if conf > self.conf_threshold:
                x, y = map(int, kpt)
                
                # Color based on keypoint importance for throwing
                if i in self.throwing_keypoints:
                    color = (0, 0, 255)  # Red for throwing-related keypoints
                    radius = 6
                else:
                    color = (255, 0, 0)  # Blue for other keypoints
                    radius = 4
                
                cv2.circle(img, (x, y), radius, color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """
        Process a single frame for pose detection and throwing analysis
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            tuple: (processed_frame, detections_info)
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()  # First detection
                confidences = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(17)
                
                # Analyze throwing pose
                analysis = self.analyze_throwing_pose(keypoints, confidences)
                
                # Draw pose on frame
                self.draw_pose(frame, keypoints, confidences)
                
                # Add detection info
                detections.append({
                    'keypoints': keypoints.tolist(),
                    'confidences': confidences.tolist(),
                    'analysis': analysis
                })
                
                # Display throwing score
                score_text = f"Throwing Score: {analysis['throwing_score']:.2f}"
                cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Display indicators
                y_offset = 70
                for indicator in analysis['indicators']:
                    cv2.putText(frame, f"• {indicator}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Highlight high throwing score
                if analysis['throwing_score'] > 0.6:
                    cv2.putText(frame, "THROWING DETECTED!", (10, frame.shape[0] - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame, detections
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video file for throwing detection
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save processed video
            display (bool): Whether to display video during processing
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        throwing_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Save detection info
                if detections:
                    for detection in detections:
                        if detection['analysis']['throwing_score'] > 0.5:
                            throwing_detections.append({
                                'frame': frame_count,
                                'timestamp': frame_count / fps,
                                'score': detection['analysis']['throwing_score'],
                                'indicators': detection['analysis']['indicators']
                            })
                
                # Write frame if output specified
                if writer:
                    writer.write(processed_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Waste Throwing Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Save throwing detection summary
        if throwing_detections:
            summary_path = Path(video_path).parent / f"{Path(video_path).stem}_throwing_detections.json"
            with open(summary_path, 'w') as f:
                json.dump(throwing_detections, f, indent=2)
            
            print(f"\n🎯 Throwing Detection Summary:")
            print(f"   Total throwing instances: {len(throwing_detections)}")
            print(f"   Detection summary saved: {summary_path}")
            
            # Print top detections
            top_detections = sorted(throwing_detections, key=lambda x: x['score'], reverse=True)[:5]
            print(f"   Top 5 detections:")
            for i, det in enumerate(top_detections, 1):
                print(f"     {i}. Frame {det['frame']} (t={det['timestamp']:.1f}s): Score {det['score']:.3f}")
        else:
            print("\n⚠️  No throwing motions detected")
    
    def process_webcam(self):
        """Process live webcam feed for real-time throwing detection"""
        cap = cv2.VideoCapture(0)  # Default camera
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("Starting live throwing detection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Live Throwing Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Live detection interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Waste Throwing Detection using YOLO11 Pose Estimation')
    parser.add_argument('--model', required=True, help='Path to trained YOLO11 pose model')
    parser.add_argument('--source', help='Video file path (or "webcam" for live feed)')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = WasteThrowingDetector(args.model, args.conf)
    
    if args.source == 'webcam' or args.source is None:
        # Live webcam detection
        detector.process_webcam()
    else:
        # Video file processing
        detector.process_video(
            video_path=args.source,
            output_path=args.output,
            display=not args.no_display
        )


if __name__ == "__main__":
    main()
