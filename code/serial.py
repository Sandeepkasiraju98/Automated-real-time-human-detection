from mpi4py import MPI
import cv2
import os
from imageai.Detection import ObjectDetection
import time


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(os.getcwd(), "../yolov3.pt"))
detector.loadModel()
detector.useCPU()


def detect_person(image_path, filename, detector):
    image = cv2.imread(image_path)
    detections = detector.detectObjectsFromImage(
        custom_objects=detector.CustomObjects(person=True),
        input_image=image,
        minimum_percentage_probability=30,
        output_image_path=f"../Images/detected_images_serial/{filename}.jpg"
    )
    return filename, detections


def vid_to_img(cap, folder: str):
    if not cap.isOpened():
        print("Error opening video stream or file")
        return False, []

    os.makedirs(folder, exist_ok=True)

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    count = 1
    files = []

    while ret:
        try:
            file_name = os.path.join(folder, f"frame{count}.jpg")
            cv2.imwrite(file_name, frame)
            # print(f"Saved frame: {count}/{number_of_frames}")
            ret, frame = cap.read()
            count += 1
            files.append(file_name)
        except Exception as e:
            print(f"Error processing frame {count}: {e}")
            break

    return True, files


def preprocess_video(execution_path):
    cap = cv2.VideoCapture(os.path.join(execution_path, "../samplevideo.mp4"))
    if not cap.isOpened():
        print("Error opening video file")
        return False, None, None
    success, files = vid_to_img(cap, folder=os.path.join(execution_path, "../Images/images_serial"))
    return success, files, cap


def create_video(file_paths, width, height, fps, output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height), True)
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                frame = cv2.imread(file_path)
                if frame is not None:
                    out.write(frame)
                else:
                    print(f"Warning: Frame {file_path} is None.")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        else:
            print(f"Warning: File {file_path} does not exist.")
    out.release()


def postprocess_video(cap, execution_path, files):
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    output_file_name = os.path.join(execution_path, "../Images/parallel_detected.avi")

    create_video([os.path.join(execution_path, f"../Images/detected_images_serial/frame{i}.jpg") for i in range(1, len(files) + 1)],
                 width, height, fps, output_file_name)


def get_image_files(folder: str):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            files.append(os.path.join(folder, filename))
    return files


if __name__ == "__main__":
    execution_path = os.getcwd()

    success, _, cap = preprocess_video(execution_path)

    output_folder = '../Images/images_serial'

    files = get_image_files(output_folder)

    start_time = time.time()
    results = []

    for image_path in files:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        result = detect_person(image_path, filename, detector)

    postprocess_video(cap, execution_path, files)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")

