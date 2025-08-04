import os
import json
from pathlib import Path
from scripts.inference import WasteThrowingDetector
import argparse

import cv2

def extract_keypoints_from_video(model_path, video_path, output_json):
    detector = WasteThrowingDetector(model_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    keypoints_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, detections = detector.process_frame(frame)
        for detection in detections:
            keypoints_data.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'keypoints': detection['keypoints'],
                'confidences': detection['confidences'],
                'analysis': detection['analysis']
            })
        frame_idx += 1
    cap.release()
    with open(output_json, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    print(f"Saved keypoints to {output_json}")

def main():
    parser = argparse.ArgumentParser(description='Extract pose keypoints from video and save as JSON')
    parser.add_argument('--model', required=True, help='Path to trained YOLO11 pose model')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Path to output JSON file')
    args = parser.parse_args()

    output_json = args.output
    if not output_json:
        out_dir = Path('data/keypoints')
        out_dir.mkdir(parents=True, exist_ok=True)
        output_json = out_dir / (Path(args.video).stem + '_keypoints.json')
    extract_keypoints_from_video(args.model, args.video, str(output_json))

if __name__ == '__main__':
    main()
