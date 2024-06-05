from flask import Flask, render_template, request, jsonify
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import base64
import time
import threading

app = Flask(__name__)

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
pipeline.load_lora_weights("output/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")

def generate_images(prompt):
    results = pipeline(prompt, num_images_per_prompt=4).images
    img_data_list = []
    for result in results:
        img_io = BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
        img_data_list.append(f"data:image/png;base64,{img_data}")
    return img_data_list

class Task:
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.task_number = 0

    def add_task(self, prompt):
        self.task_number += 1
        self.tasks[self.task_number] = prompt
        return self.task_number

    def run(self):
        while True:
            done_tasks = []
            for task_number, prompt in self.tasks.items():
                image_urls = generate_images(prompt)
                self.results[task_number] = {"urls": image_urls}
                done_tasks.append(task_number)
            for task_number in done_tasks:
                del self.tasks[task_number]
            time.sleep(5)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    if 'prompt' in data:
        task_id = task.add_task(data['prompt'])
        return jsonify({"taskID": task_id}), 202
    else:
        return jsonify({"error": "Missing prompt in request"}), 400

@app.route('/status/<task_id>')
def status(task_id):
    task_id = int(task_id)
    if task_id in task.tasks:
        return jsonify({"status": "processing"})
    elif task_id in task.results:
        return jsonify({"status": "done", "urls": task.results[task_id]['urls']})
    else:
        return jsonify({"status": "not found"}), 404

if __name__ == '__main__':
    task = Task()
    threading.Thread(target=task.run, daemon=True).start()
    app.run(debug=True, port=5000)



