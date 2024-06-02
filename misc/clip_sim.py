import torch
import clip
from PIL import Image
from typing import List


class SimilarityCalculator:
    def __init__(self):
        self.device = self._get_device()
        self.model, self.preprocess = self._initialize_model("ViT-B/32", self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_name="ViT-B/32", device="cpu"):
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    def _embed_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        preprocessed_image = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_embeddings = self.model.encode_image(preprocessed_image)
        return image_embeddings

    def _embed_text(self, text):
        text_token = clip.tokenize([text]).to(device)
        text_embed = self.model.encode_text(text_token)
        print(text_embed.shape)
        return text_embed

    def temporal_similarity(self, images):
        image_embeds_1 = self._embed_images(images)
        image_embeds_2 = torch.roll(image_embeds_1.clone(), -1, 0)
        # print(image_embeds_1.shape, image_embeds_2.shape)
        raw_similarity = self.cosine_similarity(image_embeds_1, image_embeds_2)
        # print(raw_similarity.shape)
        return raw_similarity.mean()

    def t2i_similarity(self, text, images):
        text_embeds = self._embed_text(text)
        image_embeds = self._embed_images(images)
        raw_similarity = self.cosine_similarity(text_embeds, image_embeds)
        print(raw_similarity.shape)
        return raw_similarity.mean()
