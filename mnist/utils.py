"""
Contains various utility functions for Pytorch model saving and training.
"""
import torch
from pathlib import Path

def save_model(
    model,
    target_dir,
    model_name
):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(
        parents=True,
        exist_ok=True
    )
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt')
    model_savepath = target_dir_path / model_name

    print(f'[INFO] Saving model to {model_savepath}')
    torch.save(
        obj=model.state_dict(),
        f=model_savepath
    )
