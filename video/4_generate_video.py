# importing libraries 
import os 
import cv2
import glob
from tqdm import tqdm
from PIL import Image 

# Video Generating function 
def generate_video():
    fps = cv2.VideoCapture('skincare.mp4').get(cv2.CAP_PROP_FPS)        # -> 30 fps
    video_name = 'video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    root = '/mnt/4TData/vuquang/3d-face-rec/video/processed-frame'
    image_paths = glob.glob(os.path.join(root, "**",  "*.jpeg"), recursive=True)
    image_paths = sorted(image_paths, key=lambda x : int(x.split("/")[-1][:-5]))
    frame = cv2.imread(image_paths[0]) 
    height, width, _ = frame.shape 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    
    # Write the frames into the video file
    for img_path in image_paths:
        video.write(cv2.imread(img_path))
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    root = "/mnt/4TData/vuquang/3d-face-rec/Deep3DFaceRecon_pytorch/checkpoints/03-06-2024/results/data/epoch_latest_000000"
    image_paths = glob.glob(os.path.join(root, "**",  "*.png"), recursive=True)
    image_paths = sorted(image_paths, key=lambda x : int(x.split("/")[-1][:-4]))
    num_of_images = len(image_paths)
    
    mean_height, mean_width = 0, 0
    for i, img_path in enumerate(tqdm(image_paths)):
        im = Image.open(img_path)
        width, height = im.size 
        mean_width += width 
        mean_height += height 
    mean_width, mean_height = int(mean_width / num_of_images), int(mean_height / num_of_images)
    
    # Resizing of the images to give 
    # them same width and height 
    for i, img_path in enumerate(tqdm(image_paths)):
        img_name = img_path.split("/")[-1][:-4]
        im = Image.open(img_path) 
        width, height = im.size 
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
        imResize.save(f'/mnt/4TData/vuquang/3d-face-rec/video/processed-frame/{img_name}.jpeg', 'JPEG', quality = 95) # setting quality 

    # Calling the generate_video function 
    generate_video() 
