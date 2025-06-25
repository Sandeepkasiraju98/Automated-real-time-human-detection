from mpi4py import MPI
import cv2
import os
from imageai.Detection import ObjectDetection
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

detector = ObjectDetection()
execution_path = os.getcwd()

# Process 0 initializes the detector and broadcasts it to all other processes
if rank == 0:
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "../yolov3.pt"))
    detector.loadModel()
    detector.useCPU()

# Broadcast the detector model to all processes
detector = comm.bcast(detector, root=0)

def detect_person(image_path, filename, detector):
    image = cv2.imread(image_path)
    detections = detector.detectObjectsFromImage(
        custom_objects=detector.CustomObjects(person=True),
        input_image=image,
        minimum_percentage_probability=50,
        output_image_path=f"../Images/detected_images_mpibcast/{filename}.jpg"
    )
    return filename, detections

def process_file_chunks(file_chunks, rank, detector):
    results = []
    for image_path in file_chunks:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        result = detect_person(image_path, filename, detector)
        #print(rank, image_path)
        results.append(result)
    return results


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


def get_image_files(folder: str):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            files.append(os.path.join(folder, filename))
    return files

def preprocess_video(execution_path):
    cap = cv2.VideoCapture(os.path.join(execution_path, "../samplevideo.mp4"))
    if not cap.isOpened():
        print("Error opening video file")
        return False, None, None
    success, files = vid_to_img(cap, folder=os.path.join(execution_path, "../Images/images_mpibcast"))
    return success, files, cap

def postprocess_video(cap, execution_path, files):
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    output_file_name = os.path.join(execution_path, "../parallel_detected.avi")

    create_video([os.path.join(execution_path, f"../Images/detected_images_mpibcast/frame{i}.jpg") for i in range(1, len(files) + 1)],
                width, height, fps, output_file_name)

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


if __name__ == "__main__":

    success, _, cap = preprocess_video(execution_path)

    output_folder = '../Images/images_mpibcast'
    files = get_image_files(output_folder)

    # start_time = time.time()

    # Divide the files among processes
    chunk_size = len(files) // size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank < size - 1 else len(files)

    start_time = time.time()

    local_files = files[start_index:end_index]
    local_results = process_file_chunks(local_files, rank, detector)
    
    # Gather results from all processes to process in process 0
    all_results = comm.gather(local_results, root=0)
    comm.Barrier()
    end_time = time.time()

    # Process 0 combines all results and performs post-processing
    if rank == 0:
        combined_results = [result for sublist in all_results for result in sublist]
        postprocess_video(cap, execution_path, combined_results)
        # end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"{elapsed_time:.6f}")
        process_number = MPI.COMM_WORLD.Get_rank()
        print(f"Process {process_number} completed in {elapsed_time:.6f} seconds")