import cv2
import dlib
import numpy as np
import os

# Paths
AVEC_ROOT = os.path.expanduser('~/Documents/AVEC2014_video')
OUTPUT_ROOT = os.path.expanduser('~/STA-DRN/dataset/avec14/image')
PREDICTOR_PATH = os.path.expanduser('~/STA-DRN/weights/shape_predictor_68_face_landmarks.dat')

# Initialise dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.jpg'), frame)
        frame_count += 1
    cap.release()
    return frame_count

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return [(p.x, p.y) for p in landmarks.parts()]

def align_face(image, landmarks, output_size=(224, 224)):
    left_eye = np.mean(landmarks[36:42], axis=0).astype(int)
    right_eye = np.mean(landmarks[42:48], axis=0).astype(int)
    mouth = np.mean(landmarks[48:68], axis=0).astype(int)
    eye_center = ((left_eye + right_eye) // 2).astype(int)
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    dist_eyes = np.sqrt((dX ** 2) + (dY ** 2))
    scale = (output_size[0] * 0.35) / dist_eyes
    dy_mouth_eye = mouth[1] - eye_center[1]
    scale_mouth = (output_size[1] * (1/3)) / dy_mouth_eye
    scale = min(scale, scale_mouth)
    M = cv2.getRotationMatrix2D((float(eye_center[0]), float(eye_center[1])), angle, scale)
    M[0, 2] += (output_size[0] * 0.5 - eye_center[0])
    M[1, 2] += (output_size[1] * (1/3) - eye_center[1])
    return cv2.warpAffine(image, M, output_size, flags=cv2.INTER_CUBIC)

def process_video(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    frames = extract_frames(video_path, output_folder)
    aligned = 0
    failed = 0
    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        image = cv2.imread(frame_path)
        landmarks = get_landmarks(image)
        if landmarks is not None:
            aligned_face = align_face(image, landmarks)
            cv2.imwrite(os.path.join(output_folder, f'aligned_{frame_file}'), aligned_face)
            aligned += 1
        else:
            failed += 1
        os.remove(frame_path)
    return aligned, failed

# Process all splits and tasks
splits = ['train', 'dev', 'test']
tasks = ['Freeform', 'Northwind']

total_aligned = 0
total_failed = 0

for split in splits:
    for task in tasks:
        video_folder = os.path.join(AVEC_ROOT, split, task)
        if not os.path.exists(video_folder):
            print(f"Skipping {split}/{task} - folder not found")
            continue
        videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        print(f"\nProcessing {split}/{task} - {len(videos)} videos")
        for video in videos:
            video_path = os.path.join(video_folder, video)
            output_folder = os.path.join(OUTPUT_ROOT, video.replace('.mp4', '_aligned'))
            print(f"  Processing {video}...")
            aligned, failed = process_video(video_path, output_folder)
            total_aligned += aligned
            total_failed += failed
            print(f"  Done - Aligned: {aligned}, Failed: {failed}")

print(f"\nAll done. Total aligned: {total_aligned}, Total failed: {total_failed}")