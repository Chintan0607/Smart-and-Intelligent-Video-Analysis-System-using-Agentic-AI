import os

def create_folders():
    """
    Create all required folders if they don't exist already
    """

    folders = [
        "uploads",
        "extracted_frames",
        "enhanced_frames",
        "outputs"
    ]

    for folder in folders:
        os.makedirs(folder,exist_ok=True)

