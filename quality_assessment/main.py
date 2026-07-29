import os
import cv2
import csv

from quality_agent import QualityAssessmentAgent

# Input folder containing all CCTV folders
INPUT_FOLDER = "../data/motion_frames"

# Output folder
REPORT_FOLDER = "../reports"

os.makedirs(REPORT_FOLDER, exist_ok=True)

OUTPUT_CSV = os.path.join(
    REPORT_FOLDER,
    "quality_report.csv"
)

agent = QualityAssessmentAgent()

with open(OUTPUT_CSV, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Video",
        "Frame",
        "Overall Score",
        "Overall Status",
        "Blur Score",
        "Blur Status",
        "Brightness Score",
        "Brightness Status",
        "Contrast Score",
        "Contrast Status",
        "Sharpness Score",
        "Sharpness Status",
        "Noise Score",
        "Noise Status",
        "Resolution",
        "Recommendations"
    ])

    video_folders = sorted(os.listdir(INPUT_FOLDER))

    print(f"\nFound {len(video_folders)} CCTV Videos\n")

    for folder in video_folders:

        folder_path = os.path.join(INPUT_FOLDER, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"Processing {folder}")

        for image_name in sorted(os.listdir(folder_path)):

            if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            report = agent.evaluate(image)

            writer.writerow([

                folder,

                image_name,

                report["overall_score"],

                report["overall_status"],

                report["blur"]["score"],

                report["blur"]["status"],

                report["brightness"]["score"],

                report["brightness"]["status"],

                report["contrast"]["score"],

                report["contrast"]["status"],

                report["sharpness"]["score"],

                report["sharpness"]["status"],

                report["noise"]["score"],

                report["noise"]["status"],

                f"{report['resolution']['width']}x{report['resolution']['height']}",

                ", ".join(report["recommendations"])

            ])

print("\nQuality Assessment Completed")

print(f"Report Saved : {OUTPUT_CSV}")