import cv2
import os
import glob
from tqdm import tqdm
from mtcnn import MTCNN

root = '/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/test-inference'
image_paths = glob.glob(os.path.join(root, "**",  "*.jpg"), recursive=True) \
             + glob.glob(os.path.join(root, "**", "*.png"), recursive=True) \
             + glob.glob(os.path.join(root, "**", "*.tif"), recursive=True) \
             + glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)

detector = MTCNN()
for i, img_path in enumerate(tqdm(image_paths)):
    img_name = img_path.split("/")[-1][:-4]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        info = detector.detect_faces(img)
        info = info[0]['keypoints']
        with open(f'/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/test-inference/detections/{img_name}.txt', 'w') as f:
            le1, le2 = info['left_eye']
            re1, re2 = info['right_eye']
            nose1, nose2 = info['nose']
            ml1, ml2 = info['mouth_left']
            mr1, mr2 = info['mouth_right']
            f.write(str(le1) + ' ' + str(le2) + '\n')
            f.write(str(re1) + ' ' + str(re2) + '\n')
            f.write(str(nose1) + ' ' + str(nose2) + '\n')
            f.write(str(ml1) + ' ' + str(ml2) + '\n')
            f.write(str(mr1) + ' ' + str(mr2) + '\n')
        f.close()
    except IndexError:
        print("No face detected!")