import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1


def load_model(device="cpu"):
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def preprocess_image(img):
    if isinstance(img, (str, bytes)):
        img = np.array(Image.open(img).convert("RGB"))
    img_pil = Image.fromarray(img.astype(np.uint8)).resize((160, 160))
    arr = np.array(img_pil, dtype=np.float32) / 255.0   
    arr = (arr - 0.5) / 0.5                              
    return torch.tensor(arr).permute(2, 0, 1)            


def embed_single(tensor, model, device="cpu"):
    # (3, 160, 160) torch.Tensor from preprocess_image

    model.eval()
    with torch.no_grad():
        emb = model(tensor.unsqueeze(0).to(device))
    return emb[0].cpu().numpy()


def embed_images(images, model, batch_size=64, device="cpu"):
    # (N, H, W, C) numpy array
    
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([preprocess_image(img) for img in batch]).to(device)
            embs = model(tensors)
            all_embeddings.append(embs.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)
