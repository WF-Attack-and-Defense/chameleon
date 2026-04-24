import os
import pickle
import torch
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


def load_trained_model(model_class, checkpoint_path: str, device: Optional[torch.device] = None, **model_kwargs):
    """
    Load a trained PyTorch model from checkpoint.
    
    Args:
        model_class: The model class to instantiate
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        **model_kwargs: Additional arguments to pass to model constructor
    
    Returns:
        Loaded model with trained weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}. Please train the model first.')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model instance
    model = model_class(**model_kwargs)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback: assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    if device is not None:
        model = model.to(device)
    
    model.eval()
    return model


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: int = 0,
                   metrics: Optional[Dict[str, Any]] = None,
                   filepath: str = './checkpoints/model.pth',
                   **kwargs):
    """
    Save model checkpoint with state dict, optimizer state, and metadata.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save (optional)
        filepath: Path to save the checkpoint
        **kwargs: Additional metadata to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Add any additional metadata
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, 
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None):
    """
    Load checkpoint and optionally restore model and optimizer states.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load the checkpoint on
    
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Checkpoint not found: {filepath}')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def del_checkpoint(filepath: str):
    """
    Delete a checkpoint file.
    
    Args:
        filepath: Path to the checkpoint file to delete
    """
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Checkpoint deleted: {filepath}")
    else:
        logger.warning(f"Checkpoint not found: {filepath}")


def save_final_model_and_cleanup(attack_type_name: str,
                                 checkpoints_dir: str,
                                 device: torch.device,
                                 model_builder: Callable[[], torch.nn.Module],
                                 dataset: str,
                                 open_world: bool = False,
                                 seq_length: Optional[int] = None):
    """
    After successful training, delete all checkpoints and save the final model.
    
    Args:
        attack_type_name: Name of the attack type (e.g., 'var_cnn', 'df')
        checkpoints_dir: Base directory for checkpoints
        device: Device to load checkpoints on
        model_builder: Callable that returns a model instance (e.g., lambda: self._build_model())
        dataset: Dataset name (used in the final model filename)
        open_world: Whether the model was trained in open-world setting (True) or closed-world (False)
        seq_length: If set, appended to the filename (e.g. ``rf_DF_CW_1000.h5`` for Chameleon / RF).
    """
    model_checkpoint_dir = os.path.join(checkpoints_dir, attack_type_name)
    
    if not os.path.exists(model_checkpoint_dir):
        return
    
    # Find the best checkpoint (highest epoch or best metrics)
    checkpoint_files = [f for f in os.listdir(model_checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        logger.warning(f"No checkpoints found in {model_checkpoint_dir}")
        return
    
    # Load the best model (use the last checkpoint or find best by metrics)
    best_checkpoint = None
    best_epoch = -1
    best_model = None
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(model_checkpoint_dir, checkpoint_file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            epoch = checkpoint.get('epoch', 0)
            if epoch > best_epoch:
                best_epoch = epoch
                best_checkpoint = checkpoint_path
                # Build model and load weights
                best_model = model_builder().to(device)
                best_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            continue
    
    if best_model is None:
        logger.warning("No valid checkpoint found to save as final model")
        return
    
    # Save final model as {model}_{dataset}_{CW|OW}.h5 or {model}_{dataset}_{CW|OW}_{seq_length}.h5
    suffix = "OW" if open_world else "CW"
    if seq_length is not None:
        final_model_path = os.path.join(
            checkpoints_dir, f"{attack_type_name}_{dataset}_{suffix}_{int(seq_length)}.h5"
        )
    else:
        final_model_path = os.path.join(checkpoints_dir, f"{attack_type_name}_{dataset}_{suffix}.h5")
    torch.save(best_model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Delete all checkpoint files
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(model_checkpoint_dir, checkpoint_file)
        del_checkpoint(checkpoint_path)
    
    # Remove the model-specific checkpoint directory if empty
    try:
        if not os.listdir(model_checkpoint_dir):
            os.rmdir(model_checkpoint_dir)
            logger.info(f"Removed empty checkpoint directory: {model_checkpoint_dir}")
    except OSError:
        pass

