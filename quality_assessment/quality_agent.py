from blur_metric import BlurMetric
from brightness_metric import BrightnessMetric
from contrast_metric import ContrastMetric
from sharpness_metric import SharpnessMetric
from noise_metric import NoiseMetric
from resolution_metric import ResolutionMetric


class QualityAssessmentAgent:

    def __init__(self):

        self.blur = BlurMetric()
        self.brightness = BrightnessMetric()
        self.contrast = ContrastMetric()
        self.sharpness = SharpnessMetric()
        self.noise = NoiseMetric()
        self.resolution = ResolutionMetric()

    def evaluate(self, image):

        blur = self.blur.evaluate(image)
        brightness = self.brightness.evaluate(image)
        contrast = self.contrast.evaluate(image)
        sharpness = self.sharpness.evaluate(image)
        noise = self.noise.evaluate(image)
        resolution = self.resolution.evaluate(image)

        score = 100
        recommendations = []

        # Blur
        if blur["status"] == "BLURRY":
            score -= 25
            recommendations.append("Apply Deblurring")

        # Brightness
        if brightness["status"] == "DARK":
            score -= 15
            recommendations.append("Increase Brightness")

        elif brightness["status"] == "BRIGHT":
            score -= 10
            recommendations.append("Reduce Exposure")

        # Contrast
        if contrast["status"] == "LOW":
            score -= 10
            recommendations.append("Enhance Contrast")

        # Sharpness
        if sharpness["status"] == "SOFT":
            score -= 15
            recommendations.append("Sharpen Image")

        # Noise
        if noise["status"] == "HIGH":
            score -= 15
            recommendations.append("Apply Denoising")

        score = max(score, 0)

        # Overall Status
        if score >= 90:
            overall = "EXCELLENT"

        elif score >= 75:
            overall = "GOOD"

        elif score >= 50:
            overall = "MODERATE"

        else:
            overall = "POOR"

        if len(recommendations) == 0:
            recommendations.append("No Enhancement Required")

        return {

            "overall_score": score,

            "overall_status": overall,

            "blur": blur,

            "brightness": brightness,

            "contrast": contrast,

            "sharpness": sharpness,

            "noise": noise,

            "resolution": resolution,

            "recommendations": recommendations

        }