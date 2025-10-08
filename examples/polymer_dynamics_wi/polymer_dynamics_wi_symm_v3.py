import os
import jax
import jax.numpy as jnp

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
from onsagernet.icnn import ICNN

from onsagernet.trainers import MLETrainer
from onsagernet.trainers import RegularisedMLETrainer

import hydra
import logging

# ------------------------- Typing imports ------------------------- #
from omegaconf import DictConfig
from onsagernet.dynamics import SDE


# Add symmetric wrapper classes (correct parity implementation)


def make_even(x):
    return jnp.array([x[0], x[1] ** 2, x[2] ** 2])


class PotentialResMLPConv(eqx.Module):
    base: PotentialResMLP
    icnn: ICNN

    def weight(args):
        return -jnp.expm1(-args[1])  # 1 - exp(-F)

    def __call__(self, x, args):
        return ICNN(x) + self.weight(args) * self.base(x, args)


class SymmetricPotential(eqx.Module):
    base: PotentialResMLPConv

    def __call__(self, x, args):
        # Transform input to ensure V is even in x2, x3: V(x) = φ(x1, x2², x3²)
        transformed_x = make_even(x)

        return self.base(transformed_x, args)


class SymmetricDissipation(eqx.Module):
    base: DissipationMatrixMLP

    def __call__(self, x, args):
        # Use even inputs for the base network so base outputs are even functions
        even_input = make_even(x)
        M_base = self.base(even_input, args)

        # Ensure symmetry robustly (in case base is numerically not exactly symmetric)
        M_base = 0.5 * (M_base + M_base.T)

        # Build a diagonal congruence transform D that introduces the required parity
        # - D[0] = 1 (leave first coord even)
        # - D[1] carries the sign of x[1] so M_{01} and M_{12} acquire the correct parity
        # - D[2] carries the sign of x[2]
        # Use sqrt(abs(x_i) + eps) so D is nonzero and the congruence preserves PD.
        eps = 1e-8
        d0 = 1.0
        d1 = jnp.sign(x[1]) * jnp.sqrt(jnp.abs(x[1]) + eps)
        d2 = jnp.sign(x[2]) * jnp.sqrt(jnp.abs(x[2]) + eps)
        D = jnp.array([d0, d1, d2])

        # Congruence transform: M = D M_base D  (implemented via outer multiplications)
        M = (D[:, None] * M_base) * D[None, :]

        return M


class SymmetricConservation(eqx.Module):
    base: ConservationMatrixMLP

    def __call__(self, x, args):
        # Get the base matrix from even inputs (x1, x2², x3²)
        even_input = make_even(x)
        W_base = self.base(even_input, args)

        # For antisymmetric W, we need:
        # - W12, W13: odd components
        # - W23: even component
        # - diagonals: zero

        # Extract components from base matrix
        # Note: W_base is antisymmetric, so W_base[1,2] = -W_base[2,1]
        W12_odd = x[1] * W_base[0, 1]  # x[1] * even_function
        W13_odd = x[2] * W_base[0, 2]  # x[2] * even_function
        # For W23, since it's even and W is antisymmetric, we take the symmetric part
        W23_even = W_base[1, 2]  # This should be even under the transformation

        # Construct antisymmetric W matrix with correct parity
        W = jnp.array(
            [
                [0.0, W12_odd, W13_odd],
                [-W12_odd, 0.0, W23_even],
                [-W13_odd, -W23_even, 0.0],
            ]
        )

        return W


class SymmetricDiffusion(eqx.Module):
    base: DiffusionMLP

    def __call__(self, x, args):
        # Transform input to even coordinates for consistency
        transformed_x = make_even(x)
        return self.base(transformed_x, args)


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

    # ICNN
    icnn = ICNN(d_in=3, widths=config.model.potential.units, key=v_key, mu=0.1)

    # Wrap with symmetric versions (input transformation)
    potential_conv = PotentialResMLPConv(base=potential, icnn=icnn)
    potential = SymmetricPotential(base=potential_conv)
    dissipation = SymmetricDissipation(base=dissipation)
    conservation = SymmetricConservation(base=conservation)
    diffusion = SymmetricDiffusion(base=diffusion)

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
    # trainer = MLETrainer(
    trainer = RegularisedMLETrainer(
        opt_options=config.train.opt,
        rop_options=config.train.rop,
        loss_options=config.train.loss,
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
