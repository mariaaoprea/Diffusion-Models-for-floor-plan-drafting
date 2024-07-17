import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import base64
import time
import threading
from diffusers import (DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, 
                       DPMSolverMultistepScheduler, HeunDiscreteScheduler, 
                       EulerAncestralDiscreteScheduler)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipeline.load_lora_weights("checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")

def generate_images(prompt, num_images, scheduler, inference_steps, task_id):
    """
    Generate a specified number of images based on a given prompt using a diffusion model.

    Args:
        prompt (str): The text prompt used to generate the images.
        num_images (int): The number of images to generate.
        scheduler (str): The name of the scheduler to use for the diffusion model.
        inference_steps (int): The number of inference steps to perform for each image.
        task_id (str): The ID of the task associated with the image generation.

    Returns:
        list: A list of generated images in base64-encoded PNG format.
    """
    logging.info(f"Starting image generation for task: {task_id}")
    start_time = time.time()
    
    # Set scheduler 
    if scheduler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    elif scheduler == "PNDM":
        pipeline.scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    elif scheduler == "EulerDiscrete":
        pipeline.scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    elif scheduler == "DPMSolverMultistep":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    elif scheduler == "HeunDiscrete":
        pipeline.scheduler = HeunDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    elif scheduler == "EulerAncestralDiscrete":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # Emit initial progress
    socketio.emit('progress', {'task_id': task_id, 'progress': 0}, namespace='/generate')
    
    images = []
    progress_per_image = 100 // int(num_images)
    for i in range(int(num_images)):
        result = pipeline(prompt, num_inference_steps=int(inference_steps)).images[0]
        img_io = BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
        images.append(f"data:image/png;base64,{img_data}")
        
        # Update progress for each image generated
        progress = (i + 1) * progress_per_image
        socketio.emit('progress', {'task_id': task_id, 'progress': progress}, namespace='/generate')
    
    socketio.emit('progress', {'task_id': task_id, 'progress': 100}, namespace='/generate')  # Ensure 100% is sent at the end
    
    end_time = time.time()
    logging.info(f"Completed image generation for task: {task_id} in {end_time - start_time} seconds")
    return images

class Task:
    """
    Represents a task that can be added and run.

    Attributes:
        tasks (dict): A dictionary to store the tasks.
        results (dict): A dictionary to store the results of the tasks.
        task_number (int): A counter to keep track of the task number.

    Methods:
        add_task: Adds a new task to the tasks dictionary.
        run: Executes the tasks and generates the results.
    """

    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.task_number = 0

    def add_task(self, prompt, num_images, scheduler, inference_steps):
        """
        Adds a new task to the tasks dictionary.

        Args:
            prompt (str): The prompt for the task.
            num_images (int): The number of images to generate.
            scheduler (str): The scheduler to use for image generation.
            inference_steps (int): The number of inference steps to perform.

        Returns:
            int: The task number assigned to the new task.
        """
        self.task_number += 1
        self.tasks[self.task_number] = (prompt, num_images, scheduler, inference_steps)
        logging.info(f"Task added: {self.task_number}")
        return self.task_number

    def run(self):
        """
        Executes the tasks and generates the results.
        """
        while True:
            done_tasks = []
            for task_number, (prompt, num_images, scheduler, inference_steps) in self.tasks.items():
                logging.info(f"Generating images for task: {task_number}")
                image_urls = generate_images(prompt, num_images, scheduler, inference_steps, task_number)
                self.results[task_number] = {"urls": image_urls}
                done_tasks.append(task_number)
                logging.info(f"Task completed: {task_number}")
            for task_number in done_tasks:
                del self.tasks[task_number]
            time.sleep(1)

@app.route('/', methods=['GET'])
def home():
    """
    Renders the 'index.html' template.

    Returns:
        The rendered 'index.html' template.
    """
    return render_template('index_2.html')

@app.route('/submit', methods=['POST'])
def submit():
    """
    Submits a task with the provided data.

    Parameters:
        data (dict): A dictionary containing the following keys:
        prompt (str): The prompt for the task.
        num_images (int): The number of images for the task.
        scheduler (str): The scheduler for the task.
        inference_steps (int): The number of inference steps for the task.

    Returns:
        response (json): A JSON response containing the task ID if the task was successfully added,
        or an error message if any of the required data is missing.
    """
    data = request.json
    if 'prompt' in data and 'num_images' in data and 'scheduler' in data and 'inference_steps' in data:
        task_id = task.add_task(data['prompt'], data['num_images'], data['scheduler'], data['inference_steps'])
        logging.info(f"Task submitted: {task_id}")
        return jsonify({"taskID": task_id}), 202
    else:
        logging.error("Missing prompt, num_images, scheduler, or inference_steps in request")
        return jsonify({"error": "Missing prompt, num_images, scheduler, or inference_steps in request"}), 400

@app.route('/status/<task_id>')
def status(task_id):
    """
    Returns the status of a task based on the given task_id.

    Args:
        task_id (int): The ID of the task.

    Returns:
        A JSON response containing the status of the task. If the task is found in `task.tasks`, 
        the status will be "processing". If the task is found in `task.results`, the status will 
        be "done" along with the URLs of the results. If the task is not found, the status will 
        be "not found" along with a 404 status code.
    """
    task_id = int(task_id)
    if task_id in task.tasks:
        return jsonify({"status": "processing"})
    elif task_id in task.results:
        return jsonify({"status": "done", "urls": task.results[task_id]['urls']})
    else:
        logging.error(f"Task not found: {task_id}")
        return jsonify({"status": "not found"}), 404

@socketio.on('connect', namespace='/generate')
def handle_connect():
    """
    Handles the connection of a client.

    This function is called when a client connects to the server. It prints a message indicating that a client has connected.
    """
    logging.info('Client connected')

@socketio.on('disconnect', namespace='/generate')
def handle_disconnect():
    """
    Handles the disconnection of a client.

    This function is called when a client disconnects from the server. It prints a message indicating that the client has disconnected.
    """
    logging.info('Client disconnected')
    
if __name__ == '__main__':
    task = Task()
    threading.Thread(target=task.run, daemon=True).start()
    socketio.run(app, debug=True, port=5000)



















