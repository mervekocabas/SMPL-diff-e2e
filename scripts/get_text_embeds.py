import torch
from diffusers import StableDiffusionPipeline

model_key = "stabilityai/stable-diffusion-2"
precision_t = torch.float32

pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)

device = 'cuda'
prompt = ""
inputs = pipe.tokenizer(
    prompt, 
    padding='max_length', 
    max_length=pipe.tokenizer.model_max_length, 
    return_tensors='pt'
    )

embeddings = pipe.text_encoder(inputs.input_ids.to(pipe.device))[0]

print("Saving text embedding to 'data/empty_text_embedding.pt'")
torch.save(embeddings, 'data/empty_text_embedding.pt')