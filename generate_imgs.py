from PIL import Image
import os

os.makedirs("test_images", exist_ok=True)

for i in range(1500):
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    img.save(f"test_images/blank_{i:04d}.jpg")
