import os
import jax

jax.config.update("jax_enable_x64", True)
import sys
sys.path.append('..')
sys.path.append('../..')
import equinox as eqx
from datasets import load_dataset
from examples.utils.data import shrink_and_concatenate
from onsagernet.dynamics import OnsagerNetV2, OnsagerNetHD2_V2

from onsagernet.models import (
    PotentialResMLPV2,
    DissipationMatrixMLPV2,
    ConservationMatrixMLPV2,
    DiffusionMLPV2,
    HamiltonianMLP,
)

from datasets import Dataset
from onsagernet.trainers import MLETrainer
import numpy as np

import hydra
import logging

# ------------------------- Typing imports ------------------------- #
from omegaconf import DictConfig
from onsagernet.dynamics import SDE


def build_model(config: DictConfig) -> SDE:
    """
    Builds the model for polymer dynamics using the OnsagerNet framework.

    Args:
        config (DictConfig): Configuration object containing model parameters.
        dataset (Dataset): Dataset object used to guide model configuration.

    Returns:
        SDE: The constructed OnsagerNet model for polymer dynamics.
    """
    # Initialize random keys for model initialization
    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key, d_key = jax.random.split(init_keys, 4)

    # Initialize each model component
    potential = PotentialResMLPV2(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        param_dim=config.model.potential.param_dim,
    )
    dissipation = DissipationMatrixMLPV2(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        param_dim=config.model.dissipation.param_dim,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLPV2(
        key=w_key,
        dim=config.dim,
        activation=config.model.conservation.activation,
        units=config.model.conservation.units,
        param_dim=config.model.conservation.param_dim,
        is_bounded=config.model.conservation.is_bounded,
    )
    diffusion = DiffusionMLPV2(
        key=d_key,
        dim=config.dim,
        units=config.model.diffusion.units,
        activation=config.model.diffusion.activation,
        alpha=config.model.diffusion.alpha,
        param_dim=config.model.diffusion.param_dim,
    )

    # Construct the OnsagerNet model using the individual components
    sde = OnsagerNetV2(
        potential=potential,
        dissipation=dissipation,
        conservation=conservation,
        diffusion=diffusion,
    )

    return sde

def build_HHD_model(config: DictConfig) -> SDE:
    """
    Builds the model for polymer dynamics using the OnsagerNet-HHD framework.

    Args:
        config (DictConfig): Configuration object containing model parameters.
        dataset (Dataset): Dataset object used to guide model configuration.

    Returns:
        SDE: The constructed OnsagerNet model for polymer dynamics.
    """
    # Initialize random keys for model initialization
    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key, d_key = jax.random.split(init_keys, 4)

    # Initialize each model component
    potential = PotentialResMLPV2(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        param_dim=config.model.potential.param_dim,
    )
    # dissipation = DissipationMatrixMLPV2(
    #     key=m_key,
    #     dim=config.dim,
    #     units=config.model.dissipation.units,
    #     activation=config.model.dissipation.activation,
    #     alpha=config.model.dissipation.alpha,
    #     param_dim=config.model.dissipation.param_dim,
    #     is_bounded=config.model.dissipation.is_bounded,
    # )
    # conservation = ConservationMatrixMLPV2(
    #     key=w_key,
    #     dim=config.dim,
    #     activation=config.model.conservation.activation,
    #     units=config.model.conservation.units,
    #     param_dim=config.model.conservation.param_dim,
    #     is_bounded=config.model.conservation.is_bounded,
    # )
    Hamiltonian = HamiltonianMLP(
        key=w_key,
        dim=config.dim,
        activation=config.model.hamiltonian.activation,
        units=config.model.hamiltonian.units,
        param_dim=config.model.hamiltonian.param_dim,
        is_bounded=config.model.hamiltonian.is_bounded,
    )
    diffusion = DiffusionMLPV2(
        key=d_key,
        dim=config.dim,
        units=config.model.diffusion.units,
        activation=config.model.diffusion.activation,
        alpha=config.model.diffusion.alpha,
        param_dim=config.model.diffusion.param_dim,
    )

    # Construct the OnsagerNet model using the individual components
    sde = OnsagerNetHD2_V2(
        D=config.dim,
        potential=potential,
        Hamiltonian=Hamiltonian,
        diffusion=diffusion,
    )

    return sde

def log_transform(data: Dataset) -> Dataset:
    """Transforms the dataset by applying a log transformation to the second column of the 'args' field.

    Args:
        data (Dataset): Input dataset with the 'args' field.

    Returns:
        Dataset: Transformed dataset with the second column of 'args' log-transformed.
    """
    return data.map(
        lambda batch: {
            "args": np.concatenate(
                [
                    batch["args"][:, :, :1],
                    np.log10(2000.0 * batch["args"][:, :, 1:2]),
                ],
                axis=-1,
            )
        },
        batched=True,
    )


@hydra.main(
    config_path="./config",
    config_name="polymer_dynamics_temperature",
    version_base=None,
)
def train_model(config: DictConfig) -> None:
    """
    Main training script for polymer dynamics with closure modelling. This function
    handles data loading, model building, and model training.

    Args:
        config (DictConfig): Configuration object containing training parameters and paths.
    """
    # Get the runtime directory from Hydra's configuration
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set up logger for monitoring training progress
    log_file = os.path.join(runtime_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Load the data from the specified repository
    logger.info(f"Loading data from {config.data.repo}...")

    # Pull the data and convert it to a fix-length trajectory
    splits = {split: split for split in config.data.splits}
    dataset_dict = load_dataset(config.data.repo)
    logger.info(f"Loaded data splits:\n{', '.join(dataset_dict.keys())}")
    logger.info("Concatenating dataset")
    dataset = shrink_and_concatenate(
        dataset_dict, new_traj_len=config.train.train_traj_len
    )

    # TODO: add disk cache feature
    # # Define cache directory path
    # cache_dir = os.path.join(runtime_dir, "cached_dataset")

    # # Check if the cache directory exists
    # if os.path.exists(cache_dir):
    #     logger.info("Loading dataset from cache...")
    #     dataset = Dataset.load_from_disk(cache_dir)
    # else:
    #     logger.info("Processing and caching dataset...")
    #     dataset = shrink_and_concatenate(
    #         dataset_dict, new_traj_len=config.train.train_traj_len
    #     )
    #     # Save the processed dataset to cache
    #     dataset.save_to_disk(cache_dir)

    if config.data.get("log_transform", False):
        logger.info("Applying log transform to Fs")
        dataset = log_transform(dataset)

    # Build the model using the configuration and dataset
    logger.info("Building model...")
    model = build_model(config)

    # Initialize the MLE trainer with configuration options
    trainer = MLETrainer(opt_options=config.train.opt, rop_options=config.train.rop)

    # Start training the model using the trainer
    logger.info(f"Training OnsagerNet for {config.train.num_epochs} epochs...")
    trained_model, _, _ = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        logger=logger,
        checkpoint_dir=runtime_dir,  # Directory to save checkpoints
        checkpoint_every=config.train.checkpoint_every,  # Frequency to save checkpoints
    )

    # Log the completion of training and save the trained model
    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()
