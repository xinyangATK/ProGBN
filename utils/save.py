import torch
import os

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        filename = os.path.join(filename, 'ckpt_last.pth')
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


