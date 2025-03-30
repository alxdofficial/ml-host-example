import os
import shutil
import redis
import json
import secrets
import string
from flask import Flask, send_file, request, jsonify
from PIL import Image
from dotenv import load_dotenv
import cv2
import numpy

app = Flask(__name__)

### debug only comment out if deploying ###
# Load .env variables
load_dotenv()

# Configurations
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT'))
DATA_FOLDER = os.getenv('DATA_FOLDER')
print("DATA_FOLDER =", DATA_FOLDER)

# Redis client
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

## end of debug only 

def generate_id(n=6):
    return ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(n))

def resize_image(image, max_size=960):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / float(max(width, height))
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)

@app.route('/submit-job', methods=['POST'])
def submit_job():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    job_id = generate_id(6)
    job_folder = os.path.join(DATA_FOLDER, job_id)
    os.makedirs(job_folder, exist_ok=True)
    os.makedirs(os.path.join(job_folder, "results"))
    response_payload = {
        'job_id': job_id,
        'images': []
    }

    manifest_path = os.path.join(job_folder, 'job_manifest.txt')
    image_ids = []

    for file in files:
        try:
            image = Image.open(file.stream)
            image.verify()
            file.stream.seek(0)
            image = Image.open(file.stream).convert('RGB')
        except Exception:
            continue

        image_id = generate_id(16)
        resized_image = resize_image(image, max_size=960)

        image_filename = f'{image_id}.jpg'
        image_path = os.path.join(job_folder, image_filename)
        resized_image.save(image_path, format='JPEG', quality=85)

        # Add to Redis queue
        task_data = json.dumps({'job_id': job_id, 'image_id': image_id})
        r.rpush('inference_queue', task_data)

        response_payload['images'].append({
            'image_id': image_id,
            'original_name': file.filename
        })

        image_ids.append(image_id)

    # Write manifest file with all image IDs
    with open(manifest_path, 'w') as manifest_file:
        for img_id in image_ids:
            manifest_file.write(f'{img_id}\n')

    return jsonify({
        'status': 'processing started',
        **response_payload
    }), 202

@app.route('/get-result/<job_id>', methods=['GET'])
def get_result(job_id):
    job_folder = os.path.join(DATA_FOLDER, job_id)
    manifest_path = os.path.join(job_folder, 'job_manifest.txt')
    results_folder = os.path.join(job_folder, 'results')

    # Check if manifest exists
    if not os.path.exists(manifest_path):
        return jsonify({'error': 'Job ID not found'}), 404

    # Read image IDs from manifest
    with open(manifest_path, 'r') as manifest_file:
        image_ids = [line.strip() for line in manifest_file if line.strip()]

    # Check which images are completed
    completed_jsons = [f for f in os.listdir(results_folder) if f.endswith('.json')] if os.path.exists(results_folder) else []
    completed_ids = [os.path.splitext(f)[0] for f in completed_jsons]

    total_images = len(image_ids)
    completed_images = len(completed_ids)

    if completed_images < total_images:
        # Job not complete yet
        return jsonify({
            'status': 'processing',
            'completed': completed_images,
            'total': total_images
        }), 202

    # Job complete - prepare response
    results = []
    for img_id in image_ids:
        json_path = os.path.join(results_folder, f'{img_id}.json')
        if not os.path.exists(json_path):
            continue  # should not happen, but safeguard

        with open(json_path, 'r') as json_file:
            img_result = json.load(json_file)

        results.append({img_id: img_result})

    response_payload = {
        'status': 'completed',
        'result': results
    }

    return jsonify(response_payload), 200

# Additional route for downloading ZIP separately
@app.route('/download-results/<job_id>', methods=['GET'])
def download_results(job_id):
    job_folder = os.path.join(DATA_FOLDER, job_id)
    results_folder = os.path.join(job_folder, 'results')
    annotated_folder = os.path.join(job_folder, 'annotated')
    manifest_path = os.path.join(job_folder, 'job_manifest.txt')

    if not os.path.exists(manifest_path):
        return jsonify({'error': 'Invalid job ID'}), 404

    os.makedirs(annotated_folder, exist_ok=True)

    # Read image IDs
    with open(manifest_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]

    for img_id in image_ids:
        json_path = os.path.join(results_folder, f'{img_id}.json')
        image_path = os.path.join(job_folder, f'{img_id}.jpg')
        annotated_path = os.path.join(annotated_folder, f'{img_id}.jpg')

        if not os.path.exists(json_path) or not os.path.exists(image_path):
            continue

        with open(json_path, 'r') as f:
            result_data = json.load(f)

        image = cv2.imread(image_path)

        for detection in result_data.get('detections', []):
            box = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']

            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} ({confidence:.2f})"

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(annotated_path, image)

    # Zip the annotated folder
    zip_path = os.path.join(job_folder, f"{job_id}_annotated")
    zip_file = f"{zip_path}.zip"
    if not os.path.exists(zip_file):
        shutil.make_archive(base_name=zip_path, format='zip', root_dir=annotated_folder)

    return send_file(zip_file, mimetype='application/zip',
                     as_attachment=True, download_name=f'{job_id}_annotated.zip')



if __name__ == '__main__':
    os.makedirs(DATA_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)


