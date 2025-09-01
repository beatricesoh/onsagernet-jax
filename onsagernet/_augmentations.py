"""
# Data augmentation for SDE training

This module provides data augmentation strategies for training SDE models.
Data augmentation can improve model robustness and generalization by applying
physically-motivated transformations to training trajectories.

## Usage

```python
from onsagernet._augmentations import (
    ReducedHeadTailFlip,
    ReducedReflectionX,
    RandomChoiceAugmentation
)
from onsagernet.trainers import MLETrainer

# Create augmentation strategy
head_tail_flip = ReducedHeadTailFlip()
reflection_x = ReducedReflectionX()

# Use RandomChoiceAugmentation to randomly select one augmentation per batch
random_choice_aug = RandomChoiceAugmentation([head_tail_flip, reflection_x])

# Use in trainer
trainer = MLETrainer(
    opt_options=config.train.opt,
    rop_options=config.train.rop,
    data_augmentation=random_choice_aug,
    augmentation_prob=0.5
)
```

## Available Augmentations

- `NoAugmentation`: Identity transformation (no augmentation)
- `ReducedHeadTailFlip`: Flip y-coordinate by multiplying dimensions by [1, -1, 1]
- `ReducedReflectionX`: Reflect across x-axis by multiplying dimensions by [1, -1, -1]
- `CompositeAugmentation`: Apply multiple augmentations sequentially
- `RandomChoiceAugmentation`: Randomly choose one augmentation from a list per batch
- `ConditionalAugmentation`: Apply augmentation based on custom conditions

"""

import equinox as eqx
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from jax import random
from jax.typing import ArrayLike

# ------------------------------------------------------------------ #
#                        Base Augmentation                           #
# ------------------------------------------------------------------ #

class DataAugmentation(eqx.Module, ABC):
    """Base class for data augmentation strategies."""

    @abstractmethod
    def __call__(
        self,
        key: random.PRNGKey,
        t: ArrayLike,
        x: ArrayLike,
        args: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Apply data augmentation to batch data.

        Args:
            key: JAX random key for stochastic augmentations
            t: Time data [batch_size, n_steps, 1]
            x: State data [batch_size, n_steps, n_dim]
            args: Arguments [batch_size, n_steps, n_args]

        Returns:
            Augmented (t, x, args) tuple
        """
        pass

# ------------------------------------------------------------------ #
#                     Specific Augmentations                         #
# ------------------------------------------------------------------ #

class NoAugmentation(DataAugmentation):
    """Identity augmentation (no transformation)."""

    def __call__(self, key, t, x, args):
        return t, x, args


class ReducedHeadTailFlip(DataAugmentation):
    """Flip the y-coordinate (second dimension) by multiplying by [1, -1, 1]."""

    def __call__(self, key, t, x, args):
        # Create flip pattern [1, -1, 1] for dimensions [x, y, z]
        flip_pattern = jnp.array([1, -1, 1])

        # Apply flip to all batch samples and time steps
        # x has shape [batch_size, n_steps, n_dim], flip_pattern broadcasts correctly
        x_flipped = x * flip_pattern

        return t, x_flipped, args



class ReducedReflectionX(DataAugmentation):

    def __call__(self, key, t, x, args):
        # Create flip pattern [1, -1, -1] for dimensions [x, y, z]
        flip_pattern = jnp.array([1, -1, -1])

        # Apply flip to all batch samples and time steps
        # x has shape [batch_size, n_steps, n_dim], flip_pattern broadcasts correctly
        x_flipped = x * flip_pattern

        return t, x_flipped, args


# ------------------------------------------------------------------ #
#                      Composite Augmentations                       #
# ------------------------------------------------------------------ #

class CompositeAugmentation(DataAugmentation):
    """Apply multiple augmentations sequentially."""

    augmentations: List[DataAugmentation]

    def __init__(self, augmentations: List[DataAugmentation]):
        """Initialize composite augmentation.

        Args:
            augmentations: List of augmentations to apply sequentially
        """
        self.augmentations = augmentations

    def __call__(self, key, t, x, args):
        keys = random.split(key, len(self.augmentations))

        for aug, subkey in zip(self.augmentations, keys):
            t, x, args = aug(subkey, t, x, args)

        return t, x, args

class RandomChoiceAugmentation(DataAugmentation):
    """Randomly choose one augmentation from a list."""

    augmentations: List[DataAugmentation]

    def __init__(self, augmentations: List[DataAugmentation]):
        """Initialize random choice augmentation.

        Args:
            augmentations: List of augmentations to choose from
        """
        self.augmentations = augmentations

    def __call__(self, key, t, x, args):
        choice_key, aug_key = random.split(key)

        # Randomly select an augmentation
        choice_idx = random.randint(choice_key, (), 0, len(self.augmentations))

        # Use jax.lax.switch for efficient conditional execution
        # Fix the closure issue by creating proper lambda functions
        def make_aug_fn(aug):
            return lambda: aug(aug_key, t, x, args)

        aug_functions = [make_aug_fn(aug) for aug in self.augmentations]
        return jax.lax.switch(choice_idx, aug_functions)

class ConditionalAugmentation(DataAugmentation):
    """Apply augmentation based on a condition function."""

    condition_fn: Callable
    augmentation: DataAugmentation
    fallback_augmentation: DataAugmentation

    def __init__(
        self,
        condition_fn: Callable[[ArrayLike, ArrayLike, ArrayLike], bool],
        augmentation: DataAugmentation,
        fallback_augmentation: Optional[DataAugmentation] = None
    ):
        """Initialize conditional augmentation.

        Args:
            condition_fn: Function that takes (t, x, args) and returns bool
            augmentation: Augmentation to apply if condition is True
            fallback_augmentation: Augmentation to apply if condition is False
        """
        self.condition_fn = condition_fn
        self.augmentation = augmentation
        self.fallback_augmentation = fallback_augmentation or NoAugmentation()

    def __call__(self, key, t, x, args):
        condition = self.condition_fn(t, x, args)

        return jax.lax.cond(
            condition,
            lambda: self.augmentation(key, t, x, args),
            lambda: self.fallback_augmentation(key, t, x, args)
        )
