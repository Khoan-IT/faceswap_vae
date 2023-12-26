from ast import arg
from itertools import chain
from altair import sample
import torch

from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from .types_ import *


class VAE(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 latent_dim:int,
                 hidden_dims:List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        modules = []
        
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256, 512]
            
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
            
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4*4)
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4*4, latent_dim)
        
        hidden_dims.reverse()
        
        self.decoder_a = self.init_decode(hidden_dims)
        self.decoder_b = self.init_decode(hidden_dims)
        
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        
    def init_decode(self, hidden_dims: List) -> nn.Sequential:
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        
        decoder = nn.Sequential(*modules)
        
        return decoder
        
        
    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    
    def decode_a(self, z: torch.tensor) -> torch.tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        result = self.decoder_a(result)
        result = self.final_layers(result)
        return result
    
    
    def decode_b(self, z: torch.tensor) -> torch.tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        result = self.decoder_b(result)
        result = self.final_layers(result)
        return result
    
    
    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
    def forward(self, input: torch.tensor, **kwargs) -> List[torch.tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if kwargs['a_decoder']:
            return [self.decode_a(z), input, mu, log_var]
        else:
            return [self.decode_b(z), input, mu, log_var]
    
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruct_loss': recons_loss.detach(), 'kld': -kld_loss.detach()}
    
    
    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.tensor:
        z = torch.randn(num_samples, self.latent_dim)
        
        z = z.to(current_device)
        
        if kwargs['a_decoder']:
            samples = self.decode_a(z)
        else:
            samples = self.decode_b(z)
        
        return samples
    
    
    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return self.forward(x, **kwargs)[0]
    
    
    
if __name__=="__main__":
    vae = VAE(3, 100)
    x = torch.zeros(5, 3, 256, 256)
    
    print(vae.generate(x, a_decoder=True).shape)