from math import inf
from statistics import mean
import numpy as np
import os
from PIL import Image
import torch
from models import VAE
from tqdm.auto import tqdm, trange
from utils import save_checkpoint, load_checkpoint, seed_everything


class Trainer(object):
    def __init__(self, args, dataloader) -> None:
        if args.common.use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        seed_everything(args.train.seed)

        self.args = args
        self.dataloader = dataloader

        self.model = VAE(args.model.in_channels, args.model.latent_dim)
        self.model = self.model.to(device=self.device)
        
        
    def init_optimizer(self):
        
        optimizer_a = torch.optim.Adam(
            [
                {'params': self.model.encoder.parameters()},
                {'params': self.model.decoder_a.parameters()},
                {'params': self.model.final_layers.parameters()}
            ],
            lr=self.args.train.learning_rate,
            betas=(0.5, 0.999),
        )
        
        optimizer_b = torch.optim.Adam(
            [
                {'params': self.model.encoder.parameters()},
                {'params': self.model.decoder_b.parameters()},
                {'params': self.model.final_layers.parameters()}
            ],
            lr=self.args.train.learning_rate,
            betas=(0.5, 0.999),
        )
    
        return optimizer_a, optimizer_b
    
    
    def get_loss(self, log_loss, loss):
        log_loss['loss'].append(loss['loss'].item())
        log_loss['reconstruct_loss'].append(loss['reconstruct_loss'].item())
        log_loss['kld'].append(loss['kld'].item())
        
        return (
            round(mean(log_loss['loss']), 3),
            round(mean(log_loss['reconstruct_loss']), 3),
            round(mean(log_loss['kld']), 3),
        )
        
        
    def train(self):
        
        optimizer_a, optimizer_b = self.init_optimizer()
        
        if self.args.train.continue_once:
            if os.path.isfile(self.args.train.continue_once):
                self.model, optimizer_a, optimizer_b, _ = load_checkpoint(
                                                            self.args.train.continue_once,
                                                            self.model,
                                                            optimizer_a,
                                                            optimizer_b,
                                                            self.device,
                                                        )
    
        log_lossa = {
            'loss' : [],
            'reconstruct_loss': [],
            'kld' : [],
        }
        
        log_lossb = {
            'loss' : [],
            'reconstruct_loss': [],
            'kld' : [],
        }
        
        train_iterator = trange(int(self.args.train.epochs), desc="Epoch")
        global_loss_a = inf
        global_loss_b = inf
        for it in train_iterator:
            epoch_iterator = tqdm(self.dataloader, desc="Iteration", position=0, leave=True)
            for batch in epoch_iterator:
                self.model.train()
                imga, imgb = batch
                
                optimizer_a.zero_grad()
                optimizer_b.zero_grad()
                
                outputa = self.model(imga.to(self.device), a_decoder=True)
                lossa = self.model.loss_function(*outputa, M_N=self.args.model.kld_weight)
                
                outputb = self.model(imgb.to(self.device), a_decoder=False)
                lossb = self.model.loss_function(*outputb, M_N=self.args.model.kld_weight)

                lossa['loss'].backward()
                lossb['loss'].backward()
                
                optimizer_a.step()
                optimizer_b.step()
                
                lossesa = self.get_loss(log_lossa, lossa)
                lossesb = self.get_loss(log_lossb, lossb)
                
                if lossa['loss'].item() <= global_loss_a and lossb['loss'].item() <= global_loss_b:
                    global_loss_a = lossa['loss'].item()
                    global_loss_b = lossb['loss'].item()
                    if not os.path.isdir(self.args.common.checkpoint_folder):
                        os.makedirs(self.args.common.checkpoint_folder)
                    
                    checkpoint_path = os.path.join(self.args.common.checkpoint_folder,
                                                   "checkpoint_{}.pt".format(it))
                    save_checkpoint(self.model,
                                    (optimizer_a, optimizer_b),
                                    it,
                                    checkpoint_path,
                                )
                    if not os.path.isdir(self.args.common.sample_folder):
                        os.makedirs(self.args.common.sample_folder)
                        
                    sample_path = os.path.join(self.args.common.sample_folder,
                                                   "sample_{}.png".format(it))
                    
                    self.save_sample(imga, imgb, sample_path)
                
                epoch_iterator.set_postfix({
                    'loss_a': lossesa[0],
                    'reconstruct_loss_a': lossesa[1],
                    'kld_a': lossesa[2],
                    'loss_b': lossesb[0],
                    'reconstruct_loss_b': lossesb[1],
                    'kld_b': lossesb[2],
                })
        
    def save_sample(self, imagea, imageb, path):
        sample_number = self.args.train.sample_number
        self.model.eval()
        imga = imagea[:sample_number]
        imgb = imageb[:sample_number]
        a2b = self.model.generate(imga.to(self.device), a_decoder=False)
        b2a = self.model.generate(imgb.to(self.device), a_decoder=True)
        
        imga = (imga.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        imgb = (imgb.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        a2b = (a2b.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        b2a = (b2a.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
   
        image = np.concatenate((imga, imgb, a2b, b2a), axis=-3)
        height = image.shape[1]
        image = image.transpose(0, 2, 1, 3).reshape(-1, height, 3).transpose(1, 0, 2)
        image = Image.fromarray(image, 'RGB')
        
        image.save(path)
        self.model.train()
        
        
        