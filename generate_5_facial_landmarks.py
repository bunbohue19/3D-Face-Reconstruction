import argparse
import cv2
import os
import glob
from tqdm import tqdm
from mtcnn import MTCNN

def main(args):    
    # root = '/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/datasets/12-05-2024/train
    root = f"{args.location}"
    # start_index = int(args.start_index)
    # end_index = int(args.end_index)
    
    image_paths = glob.glob(os.path.join(root, "**",  "*.jpg"), recursive=True)
    image_paths = sorted(image_paths, key=lambda x : os.path.basename(x).split('.jpg')[0])
    # image_paths = image_paths[start_index : end_index]
    
    detector = MTCNN()

    for i, img_path in enumerate(tqdm(image_paths)):
        img_name = img_path.split("/")[-1][:-4]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            info = detector.detect_faces(img)
            keypoints = info[0]['keypoints']
            with open(root + f'/detections/{img_name}.txt', 'w') as f:
                le1, le2 = keypoints['left_eye']
                re1, re2 = keypoints['right_eye']
                nose1, nose2 = keypoints['nose']
                ml1, ml2 = keypoints['mouth_left']
                mr1, mr2 = keypoints['mouth_right']
                f.write(str(le1) + ' ' + str(le2) + '\n')
                f.write(str(re1) + ' ' + str(re2) + '\n')
                f.write(str(nose1) + ' ' + str(nose2) + '\n')
                f.write(str(ml1) + ' ' + str(ml2) + '\n')
                f.write(str(mr1) + ' ' + str(mr2) + '\n')
            f.close()
            
            # Write success image infomation
            # with open(root.replace('train', 'train_success.txt'), 'a') as f:
            with open(root.replace('val', 'val_success.txt'), 'a') as f:
                f.write(img_name + '\n')
            f.close()
            
        except IndexError:
            print("No face detected!")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None, help='Location where store face images')
    # parser.add_argument('--start_index', type=str, default=None, help='Start from index of image')
    # parser.add_argument('--end_index', type=str, default=None, help='End index of image')
    args = parser.parse_args()
    main(args)