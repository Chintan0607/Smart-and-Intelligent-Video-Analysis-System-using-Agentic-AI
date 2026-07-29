# from motion_extractor import MotionExtractor


# def main():

#     video_path = "D:\\OpenCV\\Anomaly_Detection\\data\\input_videos\\Arson007_x264.mp4"

#     output_folder = "data/extracted_frames"

#     extractor = MotionExtractor(
#         video_path=video_path,
#         output_folder=output_folder,
#         min_contour_area=300
#     )

#     extractor.extract()


# if __name__ == "__main__":
#     main()


import os

from motion_extractor import MotionExtractor


VIDEO_FOLDER = "D:\\OpenCV\\Anomaly_Detection\\data\\input_videos"
OUTPUT_FOLDER = "D:\\OpenCV\\Anomaly_Detection\\data\\motion_frames"


def main():

    videos = [
        file for file in os.listdir(VIDEO_FOLDER)
        if file.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    print(f"Found {len(videos)} videos\n")

    for index, video in enumerate(videos, start=1):

        video_path = os.path.join(VIDEO_FOLDER, video)

        # Create a separate output folder for each CCTV video
        video_name = os.path.splitext(video)[0]

        output_path = os.path.join(
            OUTPUT_FOLDER,
            video_name
        )

        print("=" * 70)
        print(f"Processing Video {index}/{len(videos)}")
        print(f"Video : {video}")
        print("=" * 70)

        extractor = MotionExtractor(
            video_path=video_path,
            output_folder=output_path,
            min_contour_area=300
        )

        extractor.extract()

        print(f"Completed {video}\n")


if __name__ == "__main__":
    main()