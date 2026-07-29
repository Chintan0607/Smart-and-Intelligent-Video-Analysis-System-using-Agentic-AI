import os
import cv2
import shutil
from PIL import Image

def save_uploaded_video(uploaded_file,upload_folder):
    """
    Save the uploaded video inside the uploads folder
    """

    file_path = os.path.join(upload_folder,uploaded_file.name)

    with open(file_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def get_video_metadata(video_path):
    """
    This function reads the metadata of the video file
    """

    cap = cv2.VideoCapture(video_path)

    metadata = {
        "fps" : cap.get(cv2.CAP_PROP_FPS),
        "frame_count" : cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "width" : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height" : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    cap.release()

    metadata["duration"] = (
        metadata["frame_count"] / metadata["fps"]
        if metadata["fps"] > 0
        else 0
    )

    return metadata

def extract_frames(video_path,output_folder,interval):
    """
    Extract every nth frame from the video
    """

    cap = cv2.VideoCapture(video_path)

    extracted_frames = []

    frame_number = 0

    while True:

        success = cap.grab()
        if not success:
            break

        if frame_number % interval == 0:
            ret,frame = cap.retrieve()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            filename = f"frame_{frame_number:05d}.jpg"
            filepath = os.path.join(output_folder,filename)
            cv2.imwrite(filepath,frame)

            extracted_frames.append(
                {
                    "frame_number" : frame_number,
                    "filepath" : filepath,
                    "rgb_image" : frame_rgb,
                    "pil_image": pil_image
                }
            )

        frame_number += 1
    cap.release()
    return extracted_frames

def clear_previous_frames(output_folder):
    """
    Delete previously extracted frames.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder,exist_ok=True)