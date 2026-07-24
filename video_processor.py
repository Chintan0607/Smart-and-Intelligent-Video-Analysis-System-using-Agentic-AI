import os

def save_uploaded_video(uploaded_file,upload_folder):
    """
    Save the uploaded video inside the uploads folder
    """

    file_path = os.path.join(upload_folder,uploaded_file.name)

    with open(file_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path