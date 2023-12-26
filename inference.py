import argparse
import torch
import os

import numpy as np

from omegaconf import OmegaConf
from models import VAE
from PIL import Image
from utils import seed_everything


class DeepFake():
    def __init__(self, args):
        if args.common.use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        seed_everything(args.train.seed)
        self.args = args
        self.model = self.get_model()


    def get_model(self):
        model = VAE(args.model.in_channels, args.model.latent_dim)
        checkpoint_dict = torch.load(self.args.test.checkpoint_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint_dict['model'])
        else:
            model.load_state_dict(checkpoint_dict['model'])
        return model.to(self.device)

    
    def preprocess(self, image_path):
        image = Image.open(image_path)
        image = image.resize((self.args.model.image_size, self.args.model.image_size))
        image = np.asarray(image) / 255.0
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor).to(self.device)
        return image


    def __call__(self, image_path):
        image = self.preprocess(image_path)
        result_a = self.model.generate(image, a_decoder=True)
        result_b = self.model.generate(image, a_decoder=False)

        image = (image.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        imga = (result_a.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        imgb = (result_b.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

        result = np.concatenate((image, imga, imgb), axis=2)
        width = result.shape[2]
        result = result.reshape(-1, width, 3)
        result = Image.fromarray(result, 'RGB')
        file_name, file_ext = os.path.splitext(image_path)
        file_name = file_name + '_result'
        result.save(file_name + file_ext)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Code inference Deepfake')
    parser.add_argument("--image_path", required=True, type=str, help="Path to image which need to exchange")
    
    inference_args = parser.parse_args()

    args = OmegaConf.load("./config.yaml")

    deepfake = DeepFake(args)

    deepfake(inference_args.image_path)

    # Usage example: python inference.py --image_path=/home/duckhoan/Documents/Code/VAE/data/hai/1.jpg