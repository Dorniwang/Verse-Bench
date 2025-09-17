from transformers import pipeline
from transformers.image_utils import load_image
from PIL import Image
import numpy as np
import torch


class DinoV3Inferencer:
    def __init__(self, model_path):
        model_path = f"{model_path}/dinov3-vitl16-pretrain-lvd1689m"
        self.feature_extractor = pipeline(
            model=model_path,
            task="image-feature-extraction",
        )

    def infer(self, image_pil1, image_pil2):
        features1 = self.feature_extractor(image_pil1)[0][-2]
        features2 = self.feature_extractor(image_pil2)[0][-2]
        cosing_sim = self.infer_feature(features1, features2)
        return cosing_sim

    def infer_feature(self, feature1, feature2):
        feature1 = torch.from_numpy(np.array(feature1))
        feature2 = torch.from_numpy(np.array(feature2))
        cosing_sim = (feature1 @ feature2) / (feature1.norm() * feature2.norm())
        return cosing_sim.cpu().item()

    def get_feature(self, image_pil):
        features = self.feature_extractor(image_pil)[0][-2]
        return features

