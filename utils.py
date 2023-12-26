import os
import glob
import torch


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # Get name of last checkpoint
    checkpoint_folder = os.path.dirname(checkpoint_path)
    last_checkpoint = glob.glob(os.path.join(checkpoint_folder, '*.pt'))
    # Save best checkpoint
    torch.save({'model': state_dict,
                'epoch': epoch,
                'optimizer_a': optimizer[0].state_dict(),
                'optimizer_b': optimizer[1].state_dict(),
                }, checkpoint_path)
    # Remove last checkpoint
    if len(last_checkpoint) != 0:
        os.remove(last_checkpoint[0])
        
        
def load_checkpoint(checkpoint_path, model, optimizer_a, optimizer_b, device):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer_a.load_state_dict(checkpoint_dict['optimizer_a'])
    optimizer_b.load_state_dict(checkpoint_dict['optimizer_b'])
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict['model'])
    print("Continue from epoch: {}".format(epoch))
    return model.to(device), optimizer_a, optimizer_b, epoch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False