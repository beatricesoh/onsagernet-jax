"""Data utilities."""

import jax.numpy as jnp
from datasets import Dataset, Features, Array2D, DatasetDict
from datasets import concatenate_datasets


def shrink_trajectory_len(dataset: Dataset, new_traj_len: int) -> Dataset:
    """Reshapes a dataset to shrink trajectory length and increase number of examples

    This is to optimise GPU usage when the trajectory length is too long.

    Some factors to consider
    - The `new_traj_len` must be at least 2 for the loss function to work.
    - The `new_traj_len` must be smaller than or equal to the original trajectory length.
      It is suggested to use a value that is a divisor of the original trajectory length.

    Args:
        dataset (Dataset): input dataset object
        new_traj_len (int): new trajectory length

    Returns:
        Dataset: processed dataset object
    """
    old_traj_len = dataset[0]["t"].shape[0]
    # old_traj_len = dataset.features["t"].shape[0]

    assert new_traj_len >= 2, "new_traj_len must be at least 2"
    assert (
        old_traj_len >= new_traj_len
    ), "new_traj_len must be smaller than the original trajectory length"

    old_dtype = dataset.features["t"].dtype

    # Iterate to load the dataset and change shage
    ts, xs, argss = [], [], []
    shrink_factor = old_traj_len // new_traj_len
    max_time_idx = (
        new_traj_len * shrink_factor
    )  # this truncates the trajectories if old_traj_len is not divisible by new_traj_len
    for example in dataset.iter(batch_size=1):
        for col, dlist in zip(["t", "x", "args"], [ts, xs, argss]):
            arr = example[col]
            new_shape = (new_traj_len, arr.shape[-1])
            arr_reshaped = arr[:, :max_time_idx, :].reshape(1, *new_shape)
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


def shrink_and_concatenate(dataset_dict: DatasetDict, new_traj_len: int):
    """
    Processes each split in a DatasetDict by:
      1. Looping over all splits.
      2. For each example, truncating each sequence (t, x, args) to an integer multiple of new_traj_len,
         then reshaping each from shape [traj_len, feature_dim] to multiple examples of shape [new_traj_len, feature_dim].
      3. Concatenating the processed examples from all splits into a single dataset.

    This version handles the case where each feature has a different feature dimension.

    Args:
      dataset_dict (DatasetDict): A Hugging Face DatasetDict with splits.
      new_traj_len (int): The target trajectory length (must be <= traj_len for each split).

    Returns:
      Dataset: A new concatenated dataset where each example has features:
               - "t" of shape [new_traj_len, dim_t]
               - "x" of shape [new_traj_len, dim_x]
               - "args" of shape [new_traj_len, dim_args]
    """

    # # TODO: add safety check for when train_traj_len larger than the actual trajectory length

    reshaped_splits = []

    for split_name, ds in dataset_dict.items():
        all_t = []
        all_x = []
        all_args = []

        for example in ds:
            # Convert features to numpy arrays.
            arr_t = jnp.array(example["t"])
            arr_x = jnp.array(example["x"])
            arr_args = jnp.array(example["args"])

            # Ensure the trajectory length is the same across all features.
            traj_len_t, _ = arr_t.shape
            traj_len_x, _ = arr_x.shape
            traj_len_args, _ = arr_args.shape
            assert (
                traj_len_t == traj_len_x == traj_len_args
            ), f"Trajectory lengths differ across features: {traj_len_t}, {traj_len_x}, {traj_len_args}"

            traj_len = traj_len_t  # common trajectory length

            # Determine the number of chunks that fit in the current trajectory.
            factor = traj_len // new_traj_len
            truncated_len = (
                factor * new_traj_len
            )  # ensure it's a multiple of new_traj_len

            # Truncate the arrays.
            arr_t_truncated = arr_t[:truncated_len, :]
            arr_x_truncated = arr_x[:truncated_len, :]
            arr_args_truncated = arr_args[:truncated_len, :]

            # Reshape each array to split the trajectory into 'factor' chunks.
            arr_t_reshaped = arr_t_truncated.reshape(
                factor, new_traj_len, arr_t.shape[1]
            )
            arr_x_reshaped = arr_x_truncated.reshape(
                factor, new_traj_len, arr_x.shape[1]
            )
            arr_args_reshaped = arr_args_truncated.reshape(
                factor, new_traj_len, arr_args.shape[1]
            )

            # Append each new chunk as a separate example.
            for i in range(factor):
                all_t.append(arr_t_reshaped[i])
                all_x.append(arr_x_reshaped[i])
                all_args.append(arr_args_reshaped[i])

        # Build a new dataset for the current split.
        new_ds = Dataset.from_dict({"t": all_t, "x": all_x, "args": all_args})
        reshaped_splits.append(new_ds)

    # Concatenate the datasets from all splits into a single dataset.
    combined_dataset = concatenate_datasets(reshaped_splits)
    return combined_dataset.with_format("jax")
