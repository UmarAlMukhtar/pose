"""
Frame Extraction Script for Pose Estimation Training

This script extracts frames from videos for training a pose estimation model.
It includes options for different extraction strategies to get quality training data.
"""

import cv2
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json


class FrameExtractor:
    def __init__(self, video_dir, output_dir, strategy='uniform', fps_target=2):
        """
        Initialize the Frame Extractor
        
        Args:
            video_dir (str): Directory containing input videos
            output_dir (str): Directory to save extracted frames
            strategy (str): Extraction strategy ('uniform', 'motion', 'quality')
            fps_target (int): Target frames per second for extraction
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.strategy = strategy
        self.fps_target = fps_target
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata = []
    
    def extract_frames_uniform(self, video_path, video_name):
        """Extract frames at uniform intervals"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.fps_target))
        
        frame_count = 0
        extracted_count = 0
        
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        pbar = tqdm(desc=f"Extracting from {video_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame_{extracted_count:06d}.jpg"
                frame_path = video_output_dir / frame_filename
                
                # Save frame
                cv2.imwrite(str(frame_path), frame)
                
                # Store metadata
                self.metadata.append({
                    'video_name': video_name,
                    'frame_number': frame_count,
                    'extracted_frame_id': extracted_count,
                    'timestamp': frame_count / fps,
                    'frame_path': str(frame_path)
                })
                
                extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        return extracted_count
    
    def extract_frames_motion(self, video_path, video_name):
        """Extract frames based on motion detection"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return 0
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_count = 0
        extracted_count = 0
        motion_threshold = 10000  # Adjust based on your needs
        
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        pbar = tqdm(desc=f"Motion-based extraction from {video_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = np.sum(diff)
            
            # Extract frame if significant motion detected
            if motion_score > motion_threshold:
                frame_filename = f"{video_name}_motion_frame_{extracted_count:06d}.jpg"
                frame_path = video_output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                
                self.metadata.append({
                    'video_name': video_name,
                    'frame_number': frame_count,
                    'extracted_frame_id': extracted_count,
                    'timestamp': frame_count / fps,
                    'motion_score': float(motion_score),
                    'frame_path': str(frame_path)
                })
                
                extracted_count += 1
                prev_gray = gray  # Update reference frame
            
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        return extracted_count
    
    def extract_frames_quality(self, video_path, video_name):
        """Extract frames based on image quality metrics"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.fps_target))
        
        frame_count = 0
        extracted_count = 0
        quality_threshold = 100  # Laplacian variance threshold
        
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        pbar = tqdm(desc=f"Quality-based extraction from {video_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Calculate image quality (sharpness using Laplacian variance)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                quality_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if quality_score > quality_threshold:
                    frame_filename = f"{video_name}_quality_frame_{extracted_count:06d}.jpg"
                    frame_path = video_output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    
                    self.metadata.append({
                        'video_name': video_name,
                        'frame_number': frame_count,
                        'extracted_frame_id': extracted_count,
                        'timestamp': frame_count / fps,
                        'quality_score': float(quality_score),
                        'frame_path': str(frame_path)
                    })
                    
                    extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        return extracted_count
    
    def process_videos(self):
        """Process all videos in the input directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.video_dir.glob(f'*{ext}'))
            video_files.extend(self.video_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        total_extracted = 0
        
        for video_path in video_files:
            video_name = video_path.stem
            print(f"\nProcessing: {video_path.name}")
            
            if self.strategy == 'uniform':
                extracted = self.extract_frames_uniform(video_path, video_name)
            elif self.strategy == 'motion':
                extracted = self.extract_frames_motion(video_path, video_name)
            elif self.strategy == 'quality':
                extracted = self.extract_frames_quality(video_path, video_name)
            else:
                print(f"Unknown strategy: {self.strategy}")
                continue
            
            print(f"Extracted {extracted} frames from {video_name}")
            total_extracted += extracted
        
        # Save metadata
        metadata_path = self.output_dir / 'extraction_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nTotal frames extracted: {total_extracted}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of the extraction process"""
        if not self.metadata:
            return
        
        # Group by video
        video_stats = {}
        for item in self.metadata:
            video_name = item['video_name']
            if video_name not in video_stats:
                video_stats[video_name] = []
            video_stats[video_name].append(item)
        
        # Create report
        report = []
        report.append("# Frame Extraction Summary Report\n")
        report.append(f"**Extraction Strategy:** {self.strategy}\n")
        report.append(f"**Target FPS:** {self.fps_target}\n")
        report.append(f"**Total Videos Processed:** {len(video_stats)}\n")
        report.append(f"**Total Frames Extracted:** {len(self.metadata)}\n\n")
        
        report.append("## Per-Video Statistics\n")
        for video_name, frames in video_stats.items():
            report.append(f"### {video_name}")
            report.append(f"- Frames extracted: {len(frames)}")
            if frames:
                duration = max(f['timestamp'] for f in frames)
                report.append(f"- Video duration: {duration:.2f} seconds")
                report.append(f"- Average extraction rate: {len(frames)/duration:.2f} fps")
            report.append("")
        
        # Save report
        report_path = self.output_dir / 'extraction_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for pose estimation training')
    parser.add_argument('--video_dir', default='data/videos', 
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', default='data/frames',
                       help='Directory to save extracted frames')
    parser.add_argument('--strategy', default='uniform', 
                       choices=['uniform', 'motion', 'quality'],
                       help='Frame extraction strategy')
    parser.add_argument('--fps_target', type=int, default=2,
                       help='Target frames per second for extraction')
    
    args = parser.parse_args()
    
    # Create frame extractor
    extractor = FrameExtractor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        fps_target=args.fps_target
    )
    
    # Process videos
    extractor.process_videos()


if __name__ == "__main__":
    main()
