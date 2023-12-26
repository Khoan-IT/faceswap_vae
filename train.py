import glob
import os
from omegaconf import OmegaConf
from trainer import Trainer
from dataloader import ImageDataset, DataCollator
from torch_snippets import *


def main():
    args  = OmegaConf.load("./config.yaml")
    train(args=args)

def train(args):
    # data loader
    first_user = glob.glob(os.path.join(args.data.first_user, "*.jpg"))
    second_user = glob.glob(os.path.join(args.data.second_user, "*.jpg"))
    dataset = ImageDataset(first_user, second_user)
    collate_fn = DataCollator()
    dataloader = DataLoader(dataset,
                            batch_size=args.train.batch_size,
                            collate_fn=collate_fn,
                        )

    trainer = Trainer(
        args,
        dataloader=dataloader,
    )
    
    trainer.train()


if __name__=='__main__':
    main()