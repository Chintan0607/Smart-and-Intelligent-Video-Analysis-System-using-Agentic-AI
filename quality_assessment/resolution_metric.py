from base_metric import BaseQualityMetric


class ResolutionMetric(BaseQualityMetric):

    def evaluate(self, image):

        height, width = image.shape[:2]

        total_pixels = width * height

        if total_pixels >= 1920 * 1080:
            status = "FULL HD"

        elif total_pixels >= 1280 * 720:
            status = "HD"

        elif total_pixels >= 640 * 480:
            status = "LOW"

        else:
            status = "VERY LOW"

        return {
            "metric": "Resolution",
            "width": width,
            "height": height,
            "pixels": total_pixels,
            "status": status
        }