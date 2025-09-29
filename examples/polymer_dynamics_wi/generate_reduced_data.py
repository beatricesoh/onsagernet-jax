import os
import jax
import hydra
import logging
import pickle
import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from omegaconf import DictConfig
from examples.utils.data import shrink_and_concatenate
from examples.utils.data import get_path
from pathlib import Path


# ------------------------------------------------------------------ #
#         Build two symmetry-constrained principal components        #
# ------------------------------------------------------------------ #

# ----------------- Build symmetry linear operators ---------------- #


def make_ops(
    n: int = 300,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    idx_x = jnp.arange(0, 3 * n, 3)
    idx_y = jnp.arange(1, 3 * n, 3)
    idx_z = jnp.arange(2, 3 * n, 3)

    I = jnp.eye(3 * n)
    J = jnp.flipud(jnp.eye(n))
    R = jnp.kron(J, jnp.eye(3))

    Tx = jnp.eye(3 * n).at[idx_x, idx_x].set(-1)
    Ty = jnp.eye(3 * n).at[idx_y, idx_y].set(-1)
    Tz = jnp.eye(3 * n).at[idx_z, idx_z].set(-1)
    return I, R, Tx, Ty, Tz


# ----------------- projector for a given character ---------------- #


def projector(
    chR: int,
    chTx: int,
    chTy: int,
    chTz: int,
    ops: Optional[Tuple[jnp.ndarray, ...]] = None,
) -> jnp.ndarray:
    if ops is None:
        I, R, Tx, Ty, Tz = make_ops()
    else:
        I, R, Tx, Ty, Tz = ops

    ops_list = [
        I,
        R,
        Tx,
        Ty,
        Tz,
        R @ Tx,
        R @ Ty,
        R @ Tz,
        Tx @ Ty,
        Tx @ Tz,
        Ty @ Tz,
        R @ Tx @ Ty,
        R @ Tx @ Tz,
        R @ Ty @ Tz,
        Tx @ Ty @ Tz,
        R @ Tx @ Ty @ Tz,
    ]
    signs = [
        1,
        chR,
        chTx,
        chTy,
        chTz,
        chR * chTx,
        chR * chTy,
        chR * chTz,
        chTx * chTy,
        chTx * chTz,
        chTy * chTz,
        chR * chTx * chTy,
        chR * chTx * chTz,
        chR * chTy * chTz,
        chTx * chTy * chTz,
        chR * chTx * chTy * chTz,
    ]
    return 0.0625 * sum(s * o for s, o in zip(signs, ops_list))  # 1/16 factor


# ----------- dominant eigenvector of a symmetric matrix ----------- #


def top_eigvec(S: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    vals, vecs = jnp.linalg.eigh(S)
    v = vecs[:, -1]
    lam = vals[-1]
    return v, lam


# -------------- Build symmetric principal components -------------- #


def build_two_PCs(
    X: jnp.ndarray,
    symm_cov: bool = True,
    symm_mean: bool = True,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    Tuple[float, float],
]:

    # Generators & character projectors
    ops = make_ops()
    P1 = projector(chR=-1, chTx=-1, chTy=+1, chTz=+1, ops=ops)  # PC-1 sector
    P2 = projector(chR=+1, chTx=-1, chTy=+1, chTz=+1, ops=ops)  # PC-2 sector

    mu = X.mean(axis=0, keepdims=True)  # (1, d)
    if symm_mean:
        # --- Symmetrised centering (mean in the trivial sector) ---
        mu = symmetrise_mean(mu, ops)  # (1, d) G-invariant mean
    else:
        mu = mu.at[:, 0::3].set(0.0)

    Xc = X - mu

    # Raw covariance
    Sigma = Xc.T @ Xc / (len(X) - 1)  # (d, d)

    if symm_cov:
        # --- Symmetrise covariance over the group ---
        Sigma = symmetrize_covariance(Sigma, ops)  # (d, d)

    # PCs inside symmetry sectors
    v1, _ = top_eigvec(P1 @ Sigma @ P1)
    v1 = P1 @ v1
    v1 = v1 / jnp.linalg.norm(v1)

    v2, _ = top_eigvec(P2 @ Sigma @ P2)
    v2 = P2 @ v2
    v2 = v2 / jnp.linalg.norm(v2)

    P = jnp.vstack([v1, v2])  # (2, d)

    # Variances on centered *raw* data (fine for whitening)
    raw_projected = (P @ Xc.T).T
    lam1 = jnp.var(raw_projected[:, 0])
    lam2 = jnp.var(raw_projected[:, 1])

    whitening_scale = 1.0 / jnp.sqrt(jnp.array([lam1, lam2]))

    def encode(X_new: jnp.ndarray) -> jnp.ndarray:
        projected = (P @ (X_new - mu).T).T
        return projected * whitening_scale

    def decode(Z: jnp.ndarray) -> jnp.ndarray:
        Z_unwhitened = Z / whitening_scale
        return (Z_unwhitened @ P) + mu

    return P, mu, encode, decode, (lam1, lam2)


# def build_two_PCs(X: jnp.ndarray, eps: float = 1e-12) -> Tuple[jnp.ndarray, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray], Tuple[float, float]]:
#     """
#     Original PCA method without manual scaling
#     X  : (N, 900) flattened chains, inter-leaved (x1,y1,z1,…)
#     returns  P (2x900), mu (1x900), encode callable, decode callable, eigenvalues
#     """
#     # Create projectors for the two sectors
#     ops = make_ops()
#     P1 = projector(chR=-1, chTx=-1, chTy=+1, chTz=+1, ops=ops)   # PC-1
#     P2 = projector(chR=+1, chTx=-1, chTy=+1, chTz=+1, ops=ops)   # PC-2

#     # centre with Tx-even mean  (zero x-mean)
#     mu = X.mean(axis=0, keepdims=True)
#     mu = mu.at[:, 0::3].set(0.0)
#     Xc = X - mu
#     Sigma = Xc.T @ Xc / (len(X) - 1)        # (900,900)

#     # PC-1  (odd R & Tx)
#     v1, _ = top_eigvec(P1 @ Sigma @ P1)
#     v1 = P1 @ v1
#     v1 = v1 / jnp.linalg.norm(v1)

#     # PC-2  (odd Tx only)
#     v2, _ = top_eigvec(P2 @ Sigma @ P2)
#     v2 = P2 @ v2
#     v2 = v2 / jnp.linalg.norm(v2)  # already orthogonal to v1

#     P = jnp.vstack([v1, v2])                 # (2,900)

#     # Compute actual variances by projecting the data
#     raw_projected = (P @ Xc.T).T  # Project centered data
#     lam1 = jnp.var(raw_projected[:, 0])  # Actual variance of PC-1
#     lam2 = jnp.var(raw_projected[:, 1])  # Actual variance of PC-2

#     # Whitening: divide by square root of variances
#     whitening_scale = 1.0 / jnp.sqrt(jnp.array([lam1, lam2]))

#     def encode(X_new: jnp.ndarray) -> jnp.ndarray:
#         projected = (P @ (X_new - mu).T).T       # (N,2)
#         return projected * whitening_scale       # Apply whitening

#     def decode(Z: jnp.ndarray) -> jnp.ndarray:
#         """
#         Decode from PCA space back to original space
#         Z: (N, 2) array of PCA coordinates (whitened)
#         returns: (N, 900) array in original space
#         """
#         # Undo whitening
#         Z_unwhitened = Z / whitening_scale
#         # Project back to original space and add mean
#         reconstructed = (Z_unwhitened @ P) + mu
#         return reconstructed

#     return P, mu, encode, decode, (lam1, lam2)


# ------------------------------------------------------------------ #
#          Data processing and stratified sampling functions         #
# ------------------------------------------------------------------ #


@jax.jit
def get_extension(x: jnp.ndarray) -> float:
    x = x.reshape(-1, 3)
    extension = jnp.max(x[:, 0]) - jnp.min(x[:, 0])
    return extension


def sample_pca_data(
    dataset_name: str,
    batch_size: int = 32,
    num_bins: int = 32,
    samples_per_batch: int = 64,
    seed: int = 0,
    cache_path: Optional[str] = None,
    filename: Optional[str] = None,
    split: Optional[str] = "train",
) -> jnp.ndarray:
    """
    Load dataset and perform stratified sampling by extension lengths
    Memory-optimized version that processes data in streaming fashion
    """

    # Check if cached data exists and load it
    if filename is not None:
        cache_file = get_path(cache_path, f"{filename}.npy")
        if os.path.exists(cache_file):
            logging.info(f"Loading cached PCA data from: {cache_file}")
            x_data = np.load(cache_file)
            x_data = jnp.array(x_data)
            logging.info(f"Loaded {len(x_data)} cached samples for PCA fitting")
            return x_data

    # If no cached data, proceed with processing
    logging.info(
        f"No cached data found. Loading and processing data from: {dataset_name}"
    )

    # Set random seed for reproducibility
    np.random.seed(seed)
    logging.info(f"Set random seed to {seed} for reproducible sampling")

    # Load the specified dataset
    data = load_dataset(dataset_name).with_format("numpy")
    # Choose splits
    if split:
        # use split if given
        data = data[split]
    else:
        # Else, concatenate all available splits
        all_splits = []
        for split_name in data.keys():
            all_splits.append(data[split_name])
        data = concatenate_datasets(all_splits)

    x_data = []
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    # Process data in streaming fashion to minimize memory usage
    for batch_idx, d in enumerate(tqdm(data.iter(batch_size), total=total_batches)):
        # Process current batch
        x = d["x"].reshape(-1, 900)
        extensions = jax.vmap(get_extension)(x)

        # Stratified sampling by extension lengths
        # Bin the extensions into bins
        bins = jnp.linspace(extensions.min(), extensions.max(), num_bins + 1)
        bin_indices = (
            jnp.digitize(extensions, bins) - 1
        )  # bin_indices in [0, num_bins-1]
        x_chosen = []
        for i in range(num_bins):
            idx_in_bin = jnp.where(bin_indices == i)[0]
            if len(idx_in_bin) > 0:
                # Randomly pick one sample from this bin
                chosen_idx = np.random.choice(idx_in_bin, size=1)
                x_chosen.append(x[chosen_idx])

        # If less than samples_per_batch bins have samples, randomly fill up to samples_per_batch
        if len(x_chosen) < samples_per_batch:
            remaining = samples_per_batch - len(x_chosen)
            all_indices = jnp.arange(x.shape[0])
            already_chosen = jnp.concatenate(x_chosen).reshape(-1, x.shape[1])
            mask = jnp.ones(x.shape[0], dtype=bool)
            for arr in x_chosen:
                mask = mask.at[jnp.where((x == arr).all(axis=1))[0][0]].set(False)
            remaining_indices = all_indices[mask]
            if len(remaining_indices) > 0:
                extra_idx = np.random.choice(
                    remaining_indices,
                    size=min(remaining, len(remaining_indices)),
                    replace=False,
                )
                x_chosen.extend([x[i : i + 1] for i in extra_idx])

        # Convert to jnp array and add to collection
        if x_chosen:
            x_chosen = jnp.concatenate(x_chosen, axis=0)
            x_data.append(x_chosen)

        # Explicitly delete large objects to free memory
        del x, extensions, bins, bin_indices, d

        # Periodic memory cleanup and progress reporting
        if (batch_idx + 1) % 100 == 0:
            current_samples = sum(len(chunk) for chunk in x_data)
            logging.info(
                f"  Processed {batch_idx + 1}/{total_batches} batches, collected {current_samples} samples"
            )

    # Final concatenation
    x_data = jnp.concatenate(x_data, axis=0)
    logging.info(f"Processed and sampled {len(x_data)} samples for PCA fitting")

    # Save as NumPy array
    if filename is not None:
        cache_file = get_path(cache_path, f"{filename}.npy")
        np.save(cache_file, np.array(x_data))
        logging.info(f"Saved PCA data to: {cache_file}")

    return x_data


# ------------------------------------------------------------------ #
#         Save and load PCA components (encoder/decoder)            #
# ------------------------------------------------------------------ #


def save_pca_components(
    cache_path: str,
    filename: str,
    P: jnp.ndarray,
    mu: jnp.ndarray,
    encode: Callable,
    decode: Callable,
    lams: Tuple[float, float],
) -> None:
    """
    Save PCA components (P, mu, eigenvalues) and create encoder/decoder functions.

    Args:
        cache_path: Directory to save components
        filename: Base filename (without extension)
        P: PCA projection matrix (2, 900)
        mu: Mean vector (1, 900)
        encode: Encoder function
        decode: Decoder function
        lams: Eigenvalues tuple
    """
    # Save the raw components that can be used to reconstruct the functions
    components_file = get_path(cache_path, f"{filename}_components.pkl")

    components = {
        "P": np.array(P),
        "mu": np.array(mu),
        "eigenvalues": lams,
        "whitening_scale": 1.0 / np.sqrt(np.array(lams)),
    }

    with open(components_file, "wb") as f:
        pickle.dump(components, f)

    logging.info(f"Saved PCA components to: {components_file}")


def load_pca_components(cache_path: str, filename: str) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    Tuple[float, float],
]:
    """
    Load PCA components and reconstruct encoder/decoder functions.

    Args:
        cache_path: Directory containing components
        filename: Base filename (without extension)

    Returns:
        P, mu, encode, decode, eigenvalues
    """
    components_file = get_path(cache_path, f"{filename}_components.pkl")

    if not os.path.exists(components_file):
        raise FileNotFoundError(f"PCA components file not found: {components_file}")

    with open(components_file, "rb") as f:
        components = pickle.load(f)

    P = jnp.array(components["P"])
    mu = jnp.array(components["mu"])
    lams = components["eigenvalues"]
    whitening_scale = jnp.array(components["whitening_scale"])

    # Reconstruct encoder and decoder functions
    def encode(X_new: jnp.ndarray) -> jnp.ndarray:
        projected = (P @ (X_new - mu).T).T
        return projected * whitening_scale

    def decode(Z: jnp.ndarray) -> jnp.ndarray:
        Z_unwhitened = Z / whitening_scale
        reconstructed = (Z_unwhitened @ P) + mu
        return reconstructed

    logging.info(f"Loaded PCA components from: {components_file}")
    return P, mu, encode, decode, lams


# ------------------------------------------------------------------ #
#                 Symmetrisation of covariance matrix                #
# ------------------------------------------------------------------ #


def group_elements_from_ops(ops: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """Return stack (16, d, d) of all group elements from (I, R, Tx, Ty, Tz)."""
    I, R, Tx, Ty, Tz = ops
    elems = [
        I,
        R,
        Tx,
        Ty,
        Tz,
        R @ Tx,
        R @ Ty,
        R @ Tz,
        Tx @ Ty,
        Tx @ Tz,
        Ty @ Tz,
        R @ Tx @ Ty,
        R @ Tx @ Tz,
        R @ Ty @ Tz,
        Tx @ Ty @ Tz,
        R @ Tx @ Ty @ Tz,
    ]
    return jnp.stack(elems, axis=0)  # (16, d, d)


def symmetrize_covariance(S: jnp.ndarray, ops: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """
    S_sym = (1/16) * sum_{g in G} U(g) S U(g)^T
    """
    G = group_elements_from_ops(ops)  # (16, d, d)
    conj = jax.vmap(lambda g: g @ S @ g.T, in_axes=0, out_axes=0)
    return jnp.mean(conj(G), axis=0)  # (d, d)


def symmetrise_mean(mu: jnp.ndarray, ops: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """
    Project the mean onto the trivial character (+,+,+,+) so it is G-invariant.
    mu: (1, d) row vector (same shape you already use)
    """
    P_triv = projector(
        chR=+1, chTx=+1, chTy=+1, chTz=+1, ops=ops
    )  # (d,d), self-adjoint
    return mu @ P_triv  # (1,d)


# ------------------------------------------------------------------ #
#               Main data processing and saving routine              #
# ------------------------------------------------------------------ #


@hydra.main(version_base=None, config_path="config", config_name="polymer_dynamics_wi")
def main(config: DictConfig) -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Step 1: Process and sample data for PCA fitting
    logging.info("=" * 60)
    logging.info("STEP 1: DATA PROCESSING AND SAMPLING")
    logging.info("=" * 60)

    pca_filename = Path(config.data.filename + "_" + config.data.reduction.filename)
    X = sample_pca_data(
        config.data.reduction.pca_dataset,
        batch_size=config.data.reduction.batch_size,
        num_bins=config.data.reduction.num_bins,
        samples_per_batch=config.data.reduction.samples_per_batch,
        seed=config.data.reduction.seed,
        cache_path=config.data.cache_path,
        filename=pca_filename,
    )

    # Step 2: Build or load PCA components
    logging.info("=" * 60)
    logging.info("STEP 2: PCA FITTING")
    logging.info("=" * 60)

    # Check if we should load existing PCA components
    if config.data.reduction.get("load_existing_pca", False):
        try:
            P, mu, encode, decode, lams = load_pca_components(
                cache_path=config.data.cache_path,
                filename=pca_filename,
            )
            logging.info("Successfully loaded existing PCA components")
            # We can skip the X_jax creation and go directly to testing
            X_jax = jnp.array(X)
            del X
        except FileNotFoundError:
            logging.info("No existing PCA components found, computing new ones...")
            X_jax = jnp.array(X)
            del X
            P, mu, encode, decode, lams = build_two_PCs(X_jax)
    else:
        X_jax = jnp.array(X)
        del X
        P, mu, encode, decode, lams = build_two_PCs(X_jax)

    # PCA diagnostics
    logging.info("PCA diagnostics")
    logging.info("-" * 60)
    logging.info(f"PC-1 eigenvalue (variance): {lams[0]:.6f}")
    logging.info(f"PC-2 eigenvalue (variance): {lams[1]:.6f}")
    logging.info(f"Variance ratio (PC-1/PC-2): {lams[0]/lams[1]:.6f}")
    logging.info(
        f"Whitening scales: [{1.0/jnp.sqrt(lams[0]):.6f}, {1.0/jnp.sqrt(lams[1]):.6f}]"
    )

    # Test encoder on PCA fitting data
    logging.info("-" * 60)
    logging.info("Testing encoder on PCA fitting data")
    all_projected = encode(X_jax)
    logging.info(f"PCA data shape: {X_jax.shape}")
    logging.info(f"Projected shape: {all_projected.shape}")

    # Check if whitening actually works
    pc1_var = jnp.var(all_projected[:, 0])
    pc2_var = jnp.var(all_projected[:, 1])
    logging.info(f"PC-1 variance after whitening: {pc1_var:.6f} (should be ~1.0)")
    logging.info(f"PC-2 variance after whitening: {pc2_var:.6f} (should be ~1.0)")

    # Test decoder on a small subset
    logging.info("-" * 60)
    logging.info("Testing decoder on PCA fitting data")
    test_subset = all_projected[:5]  # Test on first 5 samples
    reconstructed = decode(test_subset)
    original_subset = X_jax[:5]
    reconstruction_error = jnp.mean(jnp.square(reconstructed - original_subset))
    logging.info(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    logging.info(
        f"Original data range: [{jnp.min(original_subset):.3f}, {jnp.max(original_subset):.3f}]"
    )
    logging.info(
        f"Reconstructed range: [{jnp.min(reconstructed):.3f}, {jnp.max(reconstructed):.3f}]"
    )

    # Save PCA components (encoder/decoder) if we computed new ones
    if not config.data.reduction.get("load_existing_pca", False):
        logging.info("-" * 60)
        logging.info("Saving PCA components")
        save_pca_components(
            cache_path=config.data.cache_path,
            filename=config.data.reduction.filename,
            P=P,
            mu=mu,
            encode=encode,
            decode=decode,
            lams=lams,
        )
    else:
        logging.info("-" * 60)
        logging.info("Using existing PCA components (not saving)")

    # Release PCA fitting data after validation
    del X_jax, all_projected

    # Step 3: Define transformation functions
    def get_extension_transform(x: jnp.ndarray) -> float:
        x = x.reshape(300, 3)
        ext_normalised = (jnp.max(x[:, 0]) - jnp.min(x[:, 0])) / 300.0
        return ext_normalised

    def pca_projection(x: jnp.ndarray) -> jnp.ndarray:
        return encode(x).ravel()

    # ------------------ Main transformation functions ----------------- #

    def transform_x(x: jnp.ndarray) -> jnp.ndarray:
        z_star = get_extension_transform(x)
        z_hat = pca_projection(x)
        return jnp.concatenate([jnp.array([z_star]), z_hat])

    def transform_args(args: jnp.ndarray) -> jnp.ndarray:
        if config.data.reduction.arg_transform == "log":
            return jnp.concatenate([args[0:1], jnp.log10(args[1:2])])
        elif config.data.reduction.arg_transform == "scale":
            return args * jnp.array([1.0, config.data.reduction.scale_factor])
        else:
            return args

    # Step 4: Transform train and test datasets
    logging.info("=" * 60)
    logging.info("STEP 3: DATASET TRANSFORMATION")
    logging.info("=" * 60)

    # Apply transformation
    transform_x_vmap = jax.vmap(transform_x)
    transform_args_vmap = jax.vmap(transform_args)

    # Initialize variables for final summary
    train_output_path = None
    test_output_path = None

    # Main transformation function
    def transform_dataset(dataset: Dataset) -> Dataset:
        return dataset.map(
            lambda batch: {
                "x": transform_x_vmap(batch["x"]),
                "args": transform_args_vmap(batch["args"]),
            }
        )

    # Process train dataset first (if not skipped)
    if config.data.reduction.skip_train:
        logging.info("Skipping training data processing (skip_train enabled)")
    else:
        logging.info(
            f"Loading and transforming train data from: {config.data.reduction.train_dataset}"
        )
        train_data = load_dataset(config.data.reduction.train_dataset).with_format(
            "numpy"
        )

        # Processing train data
        logging.info("Processing train data...")
        train_splits = config.data.reduction.train_splits
        train_data = concatenate_datasets([train_data[split] for split in train_splits])

        # Cache the transformed data
        transformed_cache_path = get_path(
            config.data.cache_path, f"{config.data.filename}_train_transformed"
        )
        if os.path.exists(transformed_cache_path):
            logging.info(
                f"Loading cached transformed train data from: {transformed_cache_path}"
            )
            train_data_pca_cached = Dataset.load_from_disk(
                transformed_cache_path
            ).with_format("jax")
            # Use cached data instead
            train_data_pca = train_data_pca_cached
        else:
            logging.info("No cached transformed data found. Transform dataset")
            train_data_pca = transform_dataset(train_data)
            logging.info("Saving transformed data to cache...")
            train_data_pca.save_to_disk(transformed_cache_path)
            logging.info(
                f"Saved transformed train data to cache: {transformed_cache_path}"
            )

        # Release original train data memory
        del train_data

        logging.info(
            f"Modify trajectory length to {config.data.reduction.train_traj_len}..."
        )
        train_data_pca = shrink_and_concatenate(
            train_data_pca, new_traj_len=config.data.reduction.train_traj_len
        )

        # Save train data immediately and release memory
        train_output_path = get_path(
            config.data.cache_path, f"{config.data.filename}_train"
        )
        logging.info(f"Saving train data to: {train_output_path}")
        train_data_pca.save_to_disk(train_output_path)

        # Release train data memory
        del train_data_pca
        logging.info("Train data processed and saved, memory released")

    # Now process test dataset (if not skipped)
    if config.data.reduction.skip_test:
        logging.info("Skipping test data processing (skip_test enabled)")
    else:
        logging.info(
            f"Loading and transforming test data from: {config.data.reduction.test_dataset}"
        )
        test_data = load_dataset(config.data.reduction.test_dataset).with_format(
            "numpy"
        )

        # Processing test data
        logging.info("Processing test data...")
        test_data_pca = transform_dataset(test_data)

        # Save test data
        test_output_path = get_path(
            config.data.cache_path, f"{config.data.filename}_test"
        )
        logging.info(f"Saving test data to: {test_output_path}")
        test_data_pca.save_to_disk(test_output_path)

        # Release test data memory
        del test_data, test_data_pca
        logging.info("Test data processed and saved, memory released")

    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"PCA fitted on sampled data from {config.data.reduction.pca_dataset}")

    if train_output_path:
        logging.info(f"Train data transformed and saved to: {train_output_path}")
    else:
        logging.info("Train data processing skipped")

    if test_output_path:
        logging.info(f"Test data transformed and saved to: {test_output_path}")
    else:
        logging.info("Test data processing skipped")


if __name__ == "__main__":
    main()
