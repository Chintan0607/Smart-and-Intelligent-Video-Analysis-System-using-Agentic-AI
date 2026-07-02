import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_degraded_test_image(filename="degraded_cctv_test.jpg"):
    width, height = 800, 600
    base_image = Image.new("RGB", (width, height), "lightgray")
    draw = ImageDraw.Draw(base_image)
    
    draw.rectangle([150, 200, 350, 500], fill="darkblue")
    draw.ellipse([450, 100, 650, 300], fill="darkred")
    
    draw.text((20, 20), "CAM 04 - 2026-07-01 23:15:00", fill="black", font_size=24)
    draw.text((200, 300), "VEHICLE-XYZ", fill="white", font_size=32)

    img_array = np.array(base_image)
    
    noise = np.random.normal(loc=0, scale=45, size=img_array.shape)
    noisy_img_array = img_array + noise
    
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_img_array)

    final_degraded_image = noisy_image.filter(ImageFilter.GaussianBlur(radius=4))

    final_degraded_image.save(filename)
    print(f"✅ Success! Test image saved as: {filename}")
    print("You can now upload this file to your Streamlit app.")

if __name__ == "__main__":
    create_degraded_test_image()