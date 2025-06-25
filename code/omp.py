import multiprocessing as mp
import time
import os
import cv2
from imageai.Detection import ObjectDetection
import sys
 
from utils import vid_to_img
execution_path = os.getcwd()

DETECTOR = ObjectDetection()
DETECTOR.setModelTypeAsYOLOv3()
DETECTOR.setModelPath("../yolov3.pt")
DETECTOR.loadModel()
 
CUSTOM = DETECTOR.CustomObjects(person=True)
 
 
def detect_person(image, count: int):
    detections = DETECTOR.detectObjectsFromImage(
        custom_objects=CUSTOM,
        input_image=image,
        minimum_percentage_probability=30,
        output_image_path=os.path.join(execution_path, f"../Images/detected_images_openmp/frame{count}.jpg")
    )
    print(count)
    # cv2.imwrite(f"media/detected_images_from_video/frame::{count}.jpg", detections[0])
    return detections
 
def create_video(file_paths, width, height, fps, output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height), True)
    for file_path in file_paths:
        out.write(cv2.imread(file_path))
    out.release()    
 
if __name__ == "__main__":
    cap = cv2.VideoCapture("../samplevideo.mp4")
 
    success, files = vid_to_img(cap, folder=os.path.join(execution_path, f"../Images"))
    print(files)
    if success:
        start_time = time.time()
        pcount = int(sys.argv[1])

        # Multiprocessing pool
        pool = mp.Pool(pcount)
 
        result = pool.starmap(
            detect_person,
            [(img, count+1) for count, img in enumerate(files)]
        )
 
        pool.close()
 
        print("Creating video")
 
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        fps = int(cap.get(cv2.CAP_PROP_FPS))
 
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter()
        output_file_name = os.path.join(execution_path,"../parallel_detected_2.avi")
        out.open(output_file_name, fourcc, fps, (width, height), True)
 
        _ = [out.write(cv2.imread(os.path.join(execution_path, f"../Images/detected_images_openmp/frame{i}.jpg"))) for i in range(1,len(files)+1)]
       
        print(f"Time taken: {time.time() - start_time}")
 