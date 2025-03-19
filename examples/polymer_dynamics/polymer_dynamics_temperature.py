# Importing necessary libraries and modules
import os
import jax

# Enable 64-bit precision for JAX computations
jax.config.update("jax_enable_x64", True)

# Import JAX libraries and other dependencies
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx  # Neural network framework for JAX

# Import data handling utilities
from datasets import load_dataset  # Dataset loading function
from examples.utils.data import shrink_trajectory_len  # Function to shrink trajectory length
from examples.utils.data_temperature import pull_data_and_convert  # Function to load and convert data

# Import dynamics and models from OnsagerNet library
from onsagernet.dynamics import OnsagerNet, SDE  # Import OnsagerNet and SDE models

# Import specific components from OnsagerNet models
from onsagernet.models import (
    PotentialResMLP,  # Model for potential energy
    DissipationMatrixMLP,  # Model for dissipation matrix
    ConservationMatrixMLP,  # Model for conservation matrix
    DiffusionDiagonalConstant,  # Model for diffusion constant
)

# Import dataset classes for handling data
from datasets import Dataset, Features, Array2D  # Import Dataset, Features, and Array2D utilities
from onsagernet.trainers import MLETrainer  # Import MLETrainer for training
import numpy as np

# Hydra and OmegaConf for configuration management
import hydra
from omegaconf import DictConfig
import logging
import time

# ------------------------- Typing imports ------------------------- #
# JAX specific typing for better type hinting
from jax import Array
from jax.typing import ArrayLike
from datasets import Dataset  # Dataset class from datasets library
from typing import Any  # General purpose type for any object

# Function to build the model for polymer dynamics
def build_model(config: DictConfig, dataset: Dataset) -> SDE:
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
    potential = PotentialResMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.potential.activation,
        alpha=config.model.dissipation.alpha,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        activation=config.model.potential.activation,
        units=config.model.conservation.units,
    )
    diffusion = DiffusionDiagonalConstant(
        key=d_key,
        dim=config.dim,
        alpha=config.model.diffusion.alpha,
    )

    # Construct the OnsagerNet model using the individual components
    sde = OnsagerNet(
        potential=potential,
        dissipation=dissipation,
        conservation=conservation,
        diffusion=diffusion,
    )

    return sde  # Return the built model


# Hydra main function to start the training process
@hydra.main(config_path="./config", config_name="polymer_dynamics_temperature", version_base=None)
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
    log_file = os.path.join(runtime_dir,"training.log")
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
                ]
            )
    logger = logging.getLogger(__name__)

    # Load the data from the specified repository
    logger.info(f"Loading data from {config.data.repo}...")

    # Retrieve trajectory length from configuration (if specified)
    train_traj_len = config.train.get("train_traj_len", None)

    # Pull the data and convert it to a JAX-compatible format (2D array)
    dataset = pull_data_and_convert(config.data.repo, np.atleast_1d(config.data.splits), train_traj_len)

    # Build the model using the configuration and dataset
    logger.info("Building model...")
    model = build_model(config, dataset)

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


# Main entry point to start the training when the script is executed
if __name__ == "__main__":
    train_model()  # Call the main training function

