"""Data utilities."""

import jax.numpy as jnp
from datasets import Dataset, Features, Array2D, DatasetDict
from datasets import concatenate_datasets
from tqdm import tqdm


def shrink_trajectory_len(
    dataset: Dataset, new_traj_len: int, batch_size: int = 32
) -> Dataset:
    """Reshapes a dataset to shrink trajectory length and increase number of examples

    This is to optimise GPU usage when the trajectory length is too long.

    Some factors to consider
    - The `new_traj_len` must be at least 2 for the loss function to work.
    - The `new_traj_len` must be smaller than or equal to the original trajectory length.
      It is suggested to use a value that is a divisor of the original trajectory length.

    Args:
        dataset (Dataset): input dataset object
        new_traj_len (int): new trajectory length
        batch_size (int, optional): Batch size for iterating through the dataset.
            Defaults to 32. A smaller batch size may reduce memory usage but
            increase processing time.

    Returns:
        Dataset: processed dataset object
    """
    dataset.set_format("numpy")  # * iterating this is faster than using jax format
    old_traj_len = dataset[0]["t"].shape[0]

    assert new_traj_len >= 2, "new_traj_len must be at least 2"
    assert (
        old_traj_len >= new_traj_len
    ), "new_traj_len must be smaller than the original trajectory length"

    old_dtype = str(dataset[0]["t"].dtype)

    # Iterate to load the dataset and change shage
    ts, xs, argss = [], [], []
    shrink_factor = old_traj_len // new_traj_len
    max_time_idx = (
        new_traj_len * shrink_factor
    )  # this truncates the trajectories if old_traj_len is not divisible by new_traj_len
    for example in tqdm(
        dataset.iter(batch_size=batch_size), total=dataset.num_rows // batch_size
    ):
        for col, dlist in zip(["t", "x", "args"], [ts, xs, argss]):
            arr = example[col]
            new_shape = (new_traj_len, arr.shape[-1])
            arr_reshaped = arr[:, :max_time_idx, :].reshape(-1, *new_shape)
            dlist.append(arr_reshaped)

    ts = jnp.concatenate(ts)
    xs = jnp.concatenate(xs)
    argss = jnp.concatenate(argss)

    # return the dataset
    new_features = Features(
        {
            "t": Array2D(shape=ts.shape[1:], dtype=old_dtype),
            "x": Array2D(shape=xs.shape[1:], dtype=old_dtype),
            "args": Array2D(shape=argss.shape[1:], dtype=old_dtype),
        }
    )

    new_dataset = Dataset.from_dict(
        {
            "t": ts,
            "x": xs,
            "args": argss,
        },
        features=new_features,
    )
    return new_dataset.with_format("jax")


def shrink_and_concatenate(
    dataset_dict: DatasetDict, new_traj_len: int, batch_size: int = 128
) -> Dataset:
    concatenated_dataset = concatenate_datasets(
        [
            shrink_trajectory_len(dataset, new_traj_len, batch_size)
            for _, dataset in dataset_dict.items()
        ]
    )
    return concatenated_dataset.with_format("jax")
