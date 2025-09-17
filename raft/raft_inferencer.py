import os
from easydict import EasyDict
import torch
from raft.core.raft import RAFT
import numpy as np
from raft.core.utils.utils import InputPadder
import cv2
from raft.core.utils.flow_viz import flow_to_image

def resize_image(image):
    h,w,_ = image.shape
    target_w = w
    target_h = h
    while target_w > 640 or target_h > 640:
        target_w = target_w // 2
        target_h = target_h // 2
    target_size = (target_w, target_h)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR), (w, h)

class RAFTInferencer:
    def __init__(self, model_path):
        args = EasyDict()
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        args.dropout = 0.0

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(os.path.join(model_path, "raft-things.pth")))

        model = model.module
        model.to("cuda:0")
        model.eval()
        self.model = model

    def infer(self, img1_pil, img2_pil):
        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)
        img1, (src_w1, src_h1) = resize_image(img1)
        img2, (src_w2, src_h2) = resize_image(img2)

        with torch.inference_mode():
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to("cuda:0")
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to("cuda:0")

            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        return torch.abs(flow_up).mean().cpu().item()


