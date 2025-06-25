from mpi4py import MPI
import cv2
import os
from imageai.Detection import ObjectDetection
import time
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
        output_image_path=f"../detected_images_mpi_weak_scaling/{filename}.jpg"
    )
    return filename, detections

# def create_video(file_paths, width, height, fps, output_file_name):
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height), True)
#     for file_path in file_paths:
#         if os.path.exists(file_path):
#             try:
#                 frame = cv2.imread(file_path)
#                 if frame is not None:
#                     out.write(frame)
#                 else:
#                     print(f"Warning: Frame {file_path} is None.")
#             except Exception as e:
#                 print(f"Error reading file {file_path}: {e}")
#         else:
#             print(f"Warning: File {file_path} does not exist.")
#     out.release()

# def vid_to_img(cap, folder: str):
#     if not cap.isOpened():
#         print("Error opening video stream or file")
#         return False, []

#     os.makedirs(folder, exist_ok=True)

#     number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = cap.read()
#     count = 1
#     files = []

#     while ret:
#         try:
#             file_name = os.path.join(folder, f"frame{count}.jpg")
#             cv2.imwrite(file_name, frame)
#             ret, frame = cap.read()
#             count += 1
#             files.append(file_name)
#         except Exception as e:
#             print(f"Error processing frame {count}: {e}")
#             break

#     return True, files


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
            ret, frame = cap.read()
            count += 1
            files.append(file_name)
        except Exception as e:
            print(f"Error processing frame {count}: {e}")
            break

    return True, files

def create_video(file_paths, width, height, fps, output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height), True)
    
    # Batch writing images
    images = [cv2.imread(file_path) for file_path in file_paths if os.path.exists(file_path)]
    
    for frame in images:
        try:
            if frame is not None:
                out.write(frame)
            else:
                print(f"Warning: Frame is None.")
        except Exception as e:
            print(f"Error writing frame: {e}")

    out.release()

def preprocess_video(execution_path, total_processes, rank):
    cap = cv2.VideoCapture(os.path.join(execution_path, "../samplevideo.mp4"))

    if not cap.isOpened():
        print("Error opening video file")
        return False, None, None

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust the range of frames based on the total number of processes
    start_frame = int((rank / total_processes) * number_of_frames)
    end_frame = int(((rank + 1) / total_processes) * number_of_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    success, files = vid_to_img(cap, folder=os.path.join(execution_path, f"../images_mpi_weak_scaling_rank_{rank}"))

    return success, files, cap

def postprocess_video(cap, execution_path, files):
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    output_file_name = os.path.join(execution_path, "../parallel_detected_weak_scaling.avi")

    # Synchronize processes before creating the video
    comm.Barrier()

    # Sort the files based on frame number
    files = sorted(files, key=lambda x: int(x.split('frame')[1].split('.jpg')[0]))

    create_video(files, width, height, fps, output_file_name)

def process_file_chunks(file_chunks, rank, detector):
    results = []
    for image_path in file_chunks:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        result = detect_person(image_path, filename, detector)
        results.append(result)
    return results

def get_image_files(folder: str):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            files.append(os.path.join(folder, filename))
    return files

if __name__ == "__main__":
    execution_path = os.getcwd()
    total_processes = size
    start_time = time.time()

    success, files, cap = preprocess_video(execution_path, total_processes, rank)

    output_folder = '../images_mpi_weak_scaling'
    files = get_image_files(output_folder)

    file_chunks = np.array_split(files, size)
    file_chunk = comm.scatter(file_chunks, root=0)

    results = process_file_chunks(file_chunk, rank, detector)

    # Results from all processes
    all_results = comm.gather(results, root=0)

    if rank == 0:

        all_results = [result for sublist in all_results for result in sublist]

        combined_results = {}
        for filename, detections in all_results:
            if filename not in combined_results:
                combined_results[filename] = []
            combined_results[filename].extend(detections)

        # Keep detections with probability >= 50%
        filtered_results = {}
        confidence_threshold = 50
        for filename, detections in combined_results.items():
            filtered_detections = [detection for detection in detections if detection['percentage_probability'] >= confidence_threshold]
            filtered_results[filename] = filtered_detections

        postprocess_video(cap, execution_path, files)
        end_time = time.time()
        elapsed_time = end_time - start_time
        process_number = MPI.COMM_WORLD.Get_rank()
        print(f"{elapsed_time:.6f}")
