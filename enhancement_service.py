from realesrgan import RealESRGANer
import cv2
import os
import torch
from config import (
    REALESRGAN_MODEL_PATH,
    DEVICE,
    REALESRGAN_OUTSCALE,
    REALESRGAN_SCALE,
    ENHANCED_FRAME_FOLDER,
)
from basicsr.archs.rrdbnet_arch import RRDBNet


class EnhancementService:
    def __init__(self):
        print("Loading Real-ESRGAN...")
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=REALESRGAN_SCALE
        )

        self.upsampler = RealESRGANer(
            scale=REALESRGAN_SCALE,
            model_path=REALESRGAN_MODEL_PATH,
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available()
        )

        os.makedirs(ENHANCED_FRAME_FOLDER, exist_ok=True)
        print("Real-ESRGAN Loaded Successfully!!")

    def enhance_frame(self, path, filename):
        img = cv2.imread(path)
        enhanced_frame, _ = self.upsampler.enhance(
            img, outscale=REALESRGAN_OUTSCALE
        )
        output_path = os.path.join(ENHANCED_FRAME_FOLDER, f"{filename}_enhanced.jpg")
        cv2.imwrite(output_path, enhanced_frame)
        print(f"Enhanced Image saved to {output_path}")