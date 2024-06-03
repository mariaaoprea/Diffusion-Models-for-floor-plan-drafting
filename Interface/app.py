from flask import Flask, render_template, request
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import base64

app = Flask(__name__)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
pipeline.load_lora_weights("output/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    prompt = ""
    img_urls = []
    if request.method == 'POST':
        prompt = request.form['prompt']
        images = [generate_image(prompt) for _ in range(4)]
        for image in images:
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
            img_urls.append(f"data:image/png;base64,{img_data}")
    return render_template('index.html', img_urls=img_urls, prompt=prompt)


def generate_image(prompt):
    result = pipeline(prompt).images[0] 
    return result 

if __name__ == '__main__':
    app.run(debug=True, port=5000)






