import os
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
from datasets import load_from_disk
from onsagernet.dynamics import OnsagerNet
from examples.utils.data import get_path

from onsagernet.models import (
    PotentialResMLP,
    DissipationMatrixMLP,
    ConservationMatrixMLP,
    DiffusionMLP,
)

from onsagernet._augmentations import (
    RandomChoiceAugmentation,
    ReducedHeadTailFlip,
    ReducedReflectionX,
)

from onsagernet.trainers import MLETrainer, RegularisedMLETrainer

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
    potential = PotentialResMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
        param_idx=config.model.potential.param_idx,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        param_idx=config.model.dissipation.param_idx,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        activation=config.model.conservation.activation,
        units=config.model.conservation.units,
        param_idx=config.model.conservation.param_idx,
        is_bounded=config.model.conservation.is_bounded,
    )
    diffusion = DiffusionMLP(
        key=d_key,
        dim=config.dim,
        units=config.model.diffusion.units,
        activation=config.model.diffusion.activation,
        alpha=config.model.diffusion.alpha,
        param_idx=config.model.diffusion.param_idx,
    )

    # Construct the OnsagerNet model using the individual components
    sde = OnsagerNet(
        potential=potential,
        dissipation=dissipation,
        conservation=conservation,
        diffusion=diffusion,
    )

    return sde


@hydra.main(
    config_path="./config",
    config_name="polymer_dynamics_wi",
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

    # Use Hydra's default logger
    logger = logging.getLogger(__name__)
    train_path = get_path(config.data.cache_path, f"{config.data.filename}_train")
    dataset = load_from_disk(train_path).with_format("jax")
    # dataset = load_and_process_data(config)
    logger.info(f"Loaded training dataset from {train_path}")
    logger.info("=" * 60)

    # Build the model using the configuration and dataset
    logger.info("Building model...")
    model = build_model(config)

    # Load model if specified in the configuration
    if config.model.get("load_model", None):
        model_path = config.model.load_model
        logger.info(f"Loading model from {model_path}...")
        model = eqx.tree_deserialise_leaves(model_path, model)
    logger.info("=" * 60)

    # Initialize the MLE trainer with configuration options
    logger.info("Setting up data augmentation...")
    ht_aug = ReducedHeadTailFlip()
    ref_aug = ReducedReflectionX()
    aug = RandomChoiceAugmentation([ht_aug, ref_aug])

    logger.info("Configured augmentations:")
    logger.info("ReducedHeadTailFlip: Flips z2 - [1, -1, 1]")
    logger.info("ReducedReflectionX: Flips z2,z3 - [1, -1, -1]")
    logger.info("RandomChoiceAugmentation: Randomly selects one per batch")
    logger.info(f"Augmentation probability: {config.train.aug_prob}")
    logger.info("=" * 60)

    trainer = RegularisedMLETrainer(
        opt_options=config.train.opt,
        rop_options=config.train.rop,
        loss_options=config.train.loss,
        data_augmentation=aug,
        augmentation_prob=config.train.aug_prob,
    )

    # Start training the model using the trainer
    logger.info(f"Training OnsagerNet for {config.train.num_epochs} epochs...")
    logger.info("=" * 60)

    # Create a random key for augmentation
    training_key = jax.random.PRNGKey(config.train.get("seed", 123))

    trained_model, _, _ = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        logger=logger,
        checkpoint_dir=runtime_dir,  # Directory to save checkpoints
        checkpoint_every=config.train.checkpoint_every,  # Frequency to save checkpoints
        rng_key=training_key,
    )

    # Log the completion of training and save the trained model
    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()
