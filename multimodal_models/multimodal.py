from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Model multimodalny
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Wczytaj obrazy
image_paths = ["./img/img1.jpg", "./img/img2.jpg", "./img/img3.jpg", "./img/img4.jpg", "./img/img5.jpg"]
images = [Image.open(p).convert("RGB") for p in image_paths]

# 3. Zapytania tekstowe
queries = [
    "dog on the grass",
    "animal in the house",
    "something to eat",
    "building in the city",
    "red car"
]

# 4. Embedowanie
with torch.no_grad():
    inputs = processor(text=queries, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)

text_emb = outputs.text_embeds.numpy()
image_emb = outputs.image_embeds.numpy()

# 5. Cosine similarity — text vs każdy image
sim = cosine_similarity(text_emb, image_emb)

# 6. Wyniki
for i, q in enumerate(queries):
    best = np.argmax(sim[i])
    print(f"Query: {q}")
    print(f"Best image: {image_paths[best]} (score={sim[i][best]:.3f})")
    print("-" * 60)
