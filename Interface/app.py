from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import base64
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
pipeline.load_lora_weights("Checkpoints_L1/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")

def generate_images(prompt, num_images, task_id):
    # Emit initial progress
    socketio.emit('progress', {'task_id': task_id, 'progress': 0}, namespace='/generate')
    
    images = []
    progress_per_image = 100 // int(num_images)
    for i in range(int(num_images)):
        result = pipeline(prompt, num_images_per_prompt=1).images[0]  # num_inference_steps can be added
        img_io = BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
        images.append(f"data:image/png;base64,{img_data}")
        
        # Update progress for each image generated
        progress = (i + 1) * progress_per_image
        socketio.emit('progress', {'task_id': task_id, 'progress': progress}, namespace='/generate')
    
    socketio.emit('progress', {'task_id': task_id, 'progress': 100}, namespace='/generate')  # Ensure 100% is sent at the end
    return images

class Task:
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.task_number = 0

    def add_task(self, prompt, num_images):
        self.task_number += 1
        self.tasks[self.task_number] = (prompt, num_images)
        return self.task_number

    def run(self):
        while True:
            done_tasks = []
            for task_number, (prompt, num_images) in self.tasks.items():
                image_urls = generate_images(prompt, num_images, task_number)
                self.results[task_number] = {"urls": image_urls}
                done_tasks.append(task_number)
            for task_number in done_tasks:
                del self.tasks[task_number]
            time.sleep(1)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    if 'prompt' in data and 'num_images' in data:
        task_id = task.add_task(data['prompt'], data['num_images'])
        return jsonify({"taskID": task_id}), 202
    else:
        return jsonify({"error": "Missing prompt or num_images in request"}), 400

@app.route('/status/<task_id>')
def status(task_id):
    task_id = int(task_id)
    if task_id in task.tasks:
        return jsonify({"status": "processing"})
    elif task_id in task.results:
        return jsonify({"status": "done", "urls": task.results[task_id]['urls']})
    else:
        return jsonify({"status": "not found"}), 404

@socketio.on('connect', namespace='/generate')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect', namespace='/generate')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    task = Task()
    threading.Thread(target=task.run, daemon=True).start()
    socketio.run(app, debug=True, port=5000)
















