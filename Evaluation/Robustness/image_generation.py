from diffusers import AutoPipelineForText2Image
import torch

# Load the pre-trained AutoPipelineForText2Image model
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights for the model
pipeline.load_lora_weights("Checkpoints_L1/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")

# Define a list of prompts for generating floor plan images
prompts = [
"Floor plan of a small apartment, few rooms, one bathroom, small kitchen, many windows",
"Floor plan of a big apartment, few rooms, multiple bathrooms, big kitchen, many windows",
"Floor plan of a big apartment, many rooms, multiple bathrooms, big kitchen, many windows" ,
"Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few windows",
"Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows" ,

"Floor plan of a big apartment, few rooms, multiple bathrooms, small kitchen, few windows" ,
"Floor plan of a small apartment, many rooms, multiple bathrooms, big kitchen, many windows" ,
"Floor plan of a small apartment, many rooms, multiple bathrooms, big kitchen, few windows" ,
"Floor plan of a small apartment, many rooms, one bathroom, small kitchen, few windows" ,
"Floor plan of a small apartment, many rooms, one bathroom, big kitchen, few windows" ,

"A compact apartment layout, limited rooms, one washroom, a kitchenette, limited fenestration" ,
"Sketch of a small flat,  few rooms, one restroom, tiny kitchen, numerous windows" ,
"Outline of a small apartment, a couple of rooms, one washroom, spacious kitchen, a few windows" , 
"Floorplan of a spacious apartment, numerous rooms, several bathrooms, large kitchen, many windows",
"Outline of a roomy apartment, few rooms, one restroom, roomy kitchen, multiple windows",

"A floor plan of a small apartment that includes a small kitchen, one bathroom, a few rooms, and a few windows",
"A floor plan of a big apartment featuring a few rooms, many windows, multiple bathrooms, and a big kitchen",
"A floor plan of a big apartment with many rooms, a big kitchen, multiple bathrooms, and numerous windows",
"A floor plan of a small apartment that has a few rooms, one bathroom, a few windows, and a big kitchen",
"A floor plan of a small apartment featuring a big kitchen, limited windows, a single bathroom, and a few rooms",

"Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows, a balcony",
"Floor plan of a big apartment, few rooms, multiple bathrooms, big kitchen, many windows, an office room",
"Floor plan of a big apartment, many rooms, multiple bathrooms, big kitchen, many windows, an utility room with laundry facilities",
"Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few windows, a big storage room",
"Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few windows, a guest room with a private toilet",

"Floor plan of a small apartment, few rooms, one bathroom, a kitchen with an island, few windows",
"Floor plan of a small one-bedroom apartment, few rooms, one bathroom, big kitchen, few windows",
"Floor plan of a small apartment, few rooms with irregular shapes including angled walls, one bathroom, big kitchen, few windows" ,
"Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few large windows",
"Floor plan of a small apartment, few rooms, one bathroom with two sinks, big kitchen, few windows",

"Floor plan of a small apartment, few rooms, one bathroom, a mall kitchen, few windows, and a central hallway connecting all rooms",
"Floor plan of a small apartment, few rooms, one bathroom, a small kitchen, few windows, and a U-shaped hallway connecting all rooms",
"Floor plan of a small apartment, few rooms, one bathroom, a kitchen located directly next to the bedroom, few windows ",
"Floor plan of a small apartment, few rooms, one bathroom located adjacent to the living room, a kitchen with modern appliances, few windows",
"Floor plan of a small apartment, few rooms, one bathroom, a kitchen with a breakfast bar, few windows, and a living room that must be passed through to access other rooms ",

"Layout of an artist’s loft with an open plan living area, a mezzanine level bedroom, a bathroom to the left of the kitchenette, a small studio space next to a big balcony" ,
"Layout of a family apartment featuring an open-plan living and dining area, a separate kitchen, three bedrooms including a master bedroom with an ensuite bathroom, a children's play area, a home office, a guest bathroom, and a utility room with laundry facilities" ,
"Design of a modern minimalist apartment with a large open-plan living area, a sleek kitchen with an island, a master bedroom with a walk-in closet, a guest bedroom, a minimalist bathroom with a bathtub and a balcony with seating space ",
"Blueprint of an elegant two-bedroom apartment with a spacious living area, a kitchen with an island, a dining room, a master bedroom with a walk-in closet, a second bedroom with an adjacent bathroom and a small library ",
"Blueprint of a compact studio apartment featuring a combined living and sleeping area, a kitchenette, a small dining area, a bathroom with a walk-in shower, and a workspace with a desk "

]

# Define a list of labels for the generated images
labels = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]

# Generate and save the floor plan images for each prompt and label
for i, prompt in enumerate(prompts):
    for j in range(1,5):
        # Generate the image using the pipeline
        image = pipeline(prompt).images[0]
        
        # Save the generated image with the corresponding label and index
        image.save(f"Evaluation/Robustness/images/{(i//5)+1}_{i%5+1}_{j}.png")


