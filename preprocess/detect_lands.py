import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from imutils import face_utils


def detect_landmarks(faces_dir):
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    lands_dir = faces_dir.replace('frames', 'landmarks')
    os.makedirs(lands_dir, exist_ok=True)
    
    for path in os.listdir(faces_dir):
        path = os.path.join(faces_dir, path)
        img_cv2 = cv2.imread(path)
        img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        faces = face_detector(img, 1)
        if len(faces) == 0:
            f = open("failure.txt", "a", encoding="utf-8")
            f.writelines(path +f" no face detect in {path}\n")
            f.close()
            continue
        else:
            landmark = face_predictor(img, faces[0])
            landmark = face_utils.shape_to_np(landmark)
            
            land_path = path.replace('frames', 'landmarks').replace('.png', '.npy')
            np.save(str(land_path), landmark)


if __name__ == '__main__':
    faces_dir = './datasets/FaceForensics++/original_sequences/youtube/c23/frames'
    landmarks_dir = faces_dir.replace('frames', 'landmarks')
    os.makedirs(landmarks_dir, exist_ok=True)
    print(f'make dirs {landmarks_dir}')
    
    for dir in tqdm(sorted(os.listdir(faces_dir))):
        dir = os.path.join(faces_dir, dir)
        detect_landmarks(dir)