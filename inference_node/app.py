import os
import json
import time
import redis
import cv2
from ultralytics import YOLO

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT'))
DATA_FOLDER = os.getenv('DATA_FOLDER')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt").to(device)
print("YOLOv8n-seg model loaded successfully (instance segmentation).")

# List of classes you want to keep
ALLOWED_CLASSES = {
    "bottle", "wine glass", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake"
}

def polygon_area(polygon):
    """
    Calculate area of polygon using Shoelace formula.
    polygon: List of points [(x1,y1), (x2,y2), ..., (xn,yn)]
    """
    x = [point[0] for point in polygon]
    y = [point[1] for point in polygon]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(polygon)-1)))


def save_segmentation_results(job_id, image_id, result):
    results_folder = os.path.join(DATA_FOLDER, job_id, "results")
    os.makedirs(results_folder, exist_ok=True)
    result_json_path = os.path.join(results_folder, f"{image_id}.json")

    detections = []
    boxes = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    masks = result.masks.xy if result.masks is not None else []

    for box, cls, conf, mask in zip(boxes, classes, confidences, masks):
        class_name = result.names[int(cls)]
        
        # Only save detections in allowed classes
        if class_name not in ALLOWED_CLASSES:
            continue
        
        polygon = mask.tolist()
        area = polygon_area(polygon)

        detections.append({
            "class": class_name,
            "confidence": conf,
            "box": box,
            "mask_area_pixels": area
        })

    # Save results
    with open(result_json_path, 'w') as f:
        json.dump({"detections": detections}, f, indent=4)
    print(f"Segmentation results saved for {image_id} in job {job_id}")


def inference_loop():
    print("Inference loop started.")
    while True:
        batch = []
        metadata = []

        # Collect images from Redis queue until BATCH_SIZE or queue is empty
        for _ in range(BATCH_SIZE):
            task_json = redis_client.lpop("inference_queue")
            if task_json is None:
                break  # Queue is empty

            task = json.loads(task_json)
            job_id = task["job_id"]
            image_id = task["image_id"]

            image_path = os.path.join(DATA_FOLDER, job_id, f"{image_id}.jpg")
            batch.append(image_path)
            metadata.append((job_id, image_id))

        if batch:
            results = model.predict(batch, batch=len(batch), conf=0.5)

            # Save each segmentation result individually
            for (job_id, image_id), result in zip(metadata, results):
                save_segmentation_results(job_id, image_id, result)

        else:
            # Queue empty; sleep briefly
            time.sleep(1)

if __name__ == "__main__":
    inference_loop()
