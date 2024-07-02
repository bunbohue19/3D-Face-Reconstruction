# Importing all necessary libraries 
import cv2 
import os 

if __name__ == "__main__":
    cam = cv2.VideoCapture("skincare.mp4") 
    try:
        if not os.path.exists('data'): 
            os.makedirs('data') 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    currentframe = 0
    
    while(True): 
        ret, frame = cam.read() 
    
        if ret: 
            name = './data/' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
            cv2.imwrite(name, frame) 
            currentframe += 1
        else: 
            break
        
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 