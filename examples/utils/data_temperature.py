"""Data utilities."""

import jax.numpy as jnp
from datasets import Dataset, Features, Array2D, concatenate_datasets
from datasets import load_dataset
import numpy as np

import numpy as np
import jax.numpy as jnp
from datasets import Dataset, Features, Array2D

# Function to apply element-wise multiplication on the 't' feature of the dataset
def multiply_t(example):
    # Multiply the 't' feature by a small scalar (0.00005) element-wise - to be adjusted
    example['t'] = np.multiply(example['t'], 0.00005)
    return example

# Function to load datasets and optionally shrink the trajectory length
def pull_data_and_convert(data_name: str, t_splits: list = [], train_traj: int = None) -> Dataset:
    """
    Loads datasets from a given source and converts them into a desired format. Optionally,
    the trajectory length can be adjusted to improve GPU usage.
    
    Args:
        data_name (str): Name of the dataset to load.
        t_splits (list): List of dataset splits to load (e.g., ['train', 'validation']).
        train_traj (int, optional): If provided, the trajectory length will be resized to this value.
    
    Returns:
        Dataset: The final concatenated dataset after processing all splits.
    """
    datasets = []  # List to hold the processed datasets from each split
    print(len(t_splits))  # Print the number of splits to load

    for sp in t_splits:  # Loop over each dataset split (train, validation, etc.)
        # Load the dataset for the current split
        dataset = load_dataset(data_name, split=sp)
        #dataset = dataset.map(multiply_t)
        # Define new features with reshaped arrays for 't', 'x', and 'args'
        new_features = Features(
            { 
                "t": Array2D(shape=(len(dataset['t'][0]), 1), dtype='float64'),
                "x": Array2D(shape=(len(dataset['x'][0]), 3), dtype='float64'),
                "args": Array2D(shape=(len(dataset['args'][0]), 2), dtype='float64'),
            }
        )

        # Create a new dataset with the reshaped features
        new_dataset = Dataset.from_dict(
            {
                "t": dataset['t'],
                "x": dataset['x'],
                "args": dataset['args'],
            },
            features=new_features,
        )

        # Clean up the original dataset to free memory
        del(dataset)

        # Convert the dataset format to 'jax' for compatibility with JAX-based operations
        the_dataset = new_dataset.with_format("jax")

        # If a new trajectory length is specified, shrink the trajectory length
        if train_traj is not None:
            shrunk_dataset = shrink_trajectory_len(
                the_dataset, train_traj
            )  # Reduce trajectory length to improve GPU usage
            del(the_dataset)  # Clean up the original dataset
            datasets.append(shrunk_dataset)  # Add the processed dataset to the list
            del(shrunk_dataset)  # Clean up the shrunk dataset
        else: 
            # If no trajectory length adjustment is needed, add the dataset directly
            datasets.append(the_dataset)

        # After processing all splits, the datasets are appended to a list
        # Additional code could be added here if more processing steps are needed

    # Concatenate all datasets from different splits
    final_dataset = concatenate_datasets([*datasets])

    # Clean up the list of datasets to free memory
    del(datasets)

    return final_dataset  # Return the concatenated final dataset

# Function to shrink the trajectory length of a dataset
def shrink_trajectory_len(dataset: Dataset, new_traj_len: int) -> Dataset:
    """
    Reshapes a dataset to shrink the trajectory length and increase the number of examples.
    This is done to optimize GPU usage when the trajectory length is too long.
    
    Some factors to consider:
    - The `new_traj_len` must be at least 2 for the loss function to work.
    - The `new_traj_len` must be smaller than or equal to the original trajectory length.
      It's suggested to use a value that is a divisor of the original trajectory length.

    Args:
        dataset (Dataset): The input dataset object to reshape.
        new_traj_len (int): The new trajectory length (must be less than or equal to the original).
    
    Returns:
        Dataset: The processed dataset with the reshaped trajectories.
    """
    # Get the original trajectory length from the 't' feature of the dataset
    old_traj_len = dataset.features["t"].shape[0]

    # Ensure the new trajectory length is valid
    assert new_traj_len >= 2, "new_traj_len must be at least 2"  # Error if the new length is too small
    assert old_traj_len >= new_traj_len, "new_traj_len must be smaller than the original trajectory length"

    # Store the original data type of the 't' feature for later use
    old_dtype = dataset.features["t"].dtype

    # Initialize empty lists to store the reshaped data for 't', 'x', and 'args'
    ts, xs, argss = [], [], []

    # Calculate the shrink factor (how many times the new length fits into the old length)
    shrink_factor = old_traj_len // new_traj_len
    max_time_idx = new_traj_len * shrink_factor  # This ensures we don't exceed the original length

    # Iterate over the dataset in batches (batch_size=1 for simplicity)
    for example in dataset.iter(batch_size=1):
        # For each feature ('t', 'x', 'args'), reshape and append the data
        for col, dlist in zip(["t", "x", "args"], [ts, xs, argss]):
            arr = example[col]  # Get the data for the current feature
            new_shape = (new_traj_len, arr.shape[-1])  # New shape for the data (length, features)
            # Slice the data and reshape it to fit the new trajectory length
            arr_reshaped = arr[:,:max_time_idx, :].reshape(-1, *new_shape)
            dlist.append(arr_reshaped)  # Append the reshaped data to the list

    # Concatenate the reshaped data across all examples
    ts = jnp.concatenate(ts)
    xs = jnp.concatenate(xs)
    argss = jnp.concatenate(argss)

    # Create new feature definitions based on the reshaped data
    new_features = Features(
        {
            "t": Array2D(shape=ts.shape[1:], dtype=old_dtype),
            "x": Array2D(shape=xs.shape[1:], dtype=old_dtype),
            "args": Array2D(shape=argss.shape[1:], dtype=old_dtype),
        }
    )

    # Create a new dataset from the reshaped data
    new_dataset = Dataset.from_dict(
        {
            "t": ts,
            "x": xs,
            "args": argss,
        },
        features=new_features,
    )

    # Clean up the original lists and dataset to free memory
    del(ts, xs, argss, dataset)

    # Return the reshaped dataset with the 'jax' format
    return new_dataset.with_format("jax")

