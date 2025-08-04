import cv2
import os

def extract_clip(video_path, start_frame, end_frame, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


def main():
    video_dir = 'data/videos/'
    output_dir = 'data/clips/'
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print('No video files found in', video_dir)
        return

    print('Available videos:')
    for idx, vf in enumerate(video_files):
        print(f'{idx}: {vf}')
    vid_idx = int(input('Select video index: '))
    video_path = os.path.join(video_dir, video_files[vid_idx])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}')
    cap.release()

    print('Mark the frames for the action:')
    work_start = int(input('Enter start frame for "working" segment: '))
    work_end = int(input('Enter end frame for "working" segment: '))
    throw_start = int(input('Enter start frame for "throwing waste" segment: '))
    throw_end = int(input('Enter end frame for "throwing waste" segment: '))

    work_clip = os.path.join(output_dir, f'{video_files[vid_idx][:-4]}_work_{work_start}_{work_end}.mp4')
    throw_clip = os.path.join(output_dir, f'{video_files[vid_idx][:-4]}_throw_{throw_start}_{throw_end}.mp4')

    extract_clip(video_path, work_start, work_end, work_clip)
    extract_clip(video_path, throw_start, throw_end, throw_clip)
    print('Clips saved to', output_dir)

if __name__ == '__main__':
    main()
