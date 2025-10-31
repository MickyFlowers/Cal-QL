import jax
import jax.numpy as jnp
import jaxlib

print("avaiable devices:", jax.devices())
seed = 0
jax.random.PRNGKey(seed)
print("set jax random rng key successfully")

import requests
from PIL import Image
# import transformers
from transformers import AutoImageProcessor, FlaxResNetModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
print("Successfully loading image processor")
model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
print("Successfully loading Resnet-50 Jax model")
inputs = image_processor(images=image, return_tensors="np")
print(inputs["pixel_values"].shape)
outputs = model(**inputs)
print("all attrs:", dir(outputs))
last_hidden_states = outputs.last_hidden_state
pooler_output = outputs.pooler_output
print(last_hidden_states.shape)
print(pooler_output.shape)
