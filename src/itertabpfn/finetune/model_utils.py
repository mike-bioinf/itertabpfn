from pathlib import Path
import torch
from tabpfn.model.transformer import PerFeatureTransformer


def save_model(model:PerFeatureTransformer, save_path_to_fine_tuned_model: str | Path, checkpoint_config: dict) -> None:
    """Save the fine-tuned model to disk in a TabPFN-readable checkpoint format."""
    torch.save(
        dict(state_dict=model.state_dict(), config=checkpoint_config),
        f=str(save_path_to_fine_tuned_model)
    )
  



