from datasets import load_dataset, concatenate_datasets
import jax
import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import logging



# ------------------------------------------------------------------ #
#         Build two symmetry-constrained principal components        #
# ------------------------------------------------------------------ #


# ---- symmetry operators (inter-leaved layout) ----------------------
def make_ops(n=300):
    idx_x = jnp.arange(0, 3*n, 3)
    idx_y = jnp.arange(1, 3*n, 3)
    idx_z = jnp.arange(2, 3*n, 3)

    I  = jnp.eye(3*n)
    J  = jnp.flipud(jnp.eye(n))
    R  = jnp.kron(J, jnp.eye(3))          # head–tail reversal

    Tx = jnp.eye(3*n).at[idx_x, idx_x].set(-1)
    Ty = jnp.eye(3*n).at[idx_y, idx_y].set(-1)
    Tz = jnp.eye(3*n).at[idx_z, idx_z].set(-1)
    return I, R, Tx, Ty, Tz

# ---- projector for a given character --------------------------------
def projector(chR, chTx, chTy, chTz, ops=None):
    if ops is None:
        I, R, Tx, Ty, Tz = make_ops()
    else:
        I, R, Tx, Ty, Tz = ops

    ops_list = [I, R, Tx, Ty, Tz,
           R@Tx, R@Ty, R@Tz,
           Tx@Ty, Tx@Tz, Ty@Tz,
           R@Tx@Ty, R@Tx@Tz, R@Ty@Tz, Tx@Ty@Tz,
           R@Tx@Ty@Tz]
    signs = [
        1,
        chR, chTx, chTy, chTz,
        chR*chTx, chR*chTy, chR*chTz,
        chTx*chTy, chTx*chTz, chTy*chTz,
        chR*chTx*chTy, chR*chTx*chTz, chR*chTy*chTz, chTx*chTy*chTz,
        chR*chTx*chTy*chTz
    ]
    return 0.0625 * sum(s*o for s, o in zip(signs, ops_list))  # 1/16 factor

# ---- dominant eigenvector of a symmetric matrix ---------------------
def top_eigvec(S):
    vals, vecs = jnp.linalg.eigh(S)          # ascending order
    v = vecs[:, -1]
    lam = vals[-1]
    return v, lam

# ---- main builder ----------------------------------------------------
def build_two_PCs(X, eps=1e-12):
    """
    Original PCA method without manual scaling
    X  : (N, 900) flattened chains, inter-leaved (x1,y1,z1,…)
    returns  P (2×900), mu (1×900), encode callable, eigenvalues
    """
    # Create projectors for the two sectors
    ops = make_ops()
    P1 = projector(chR=-1, chTx=-1, chTy=+1, chTz=+1, ops=ops)   # PC-1
    P2 = projector(chR=+1, chTx=-1, chTy=+1, chTz=+1, ops=ops)   # PC-2

    # centre with Tx-even mean  (zero x-mean)
    mu = X.mean(axis=0, keepdims=True)
    mu = mu.at[:, 0::3].set(0.0)
    Xc = X - mu
    Sigma = Xc.T @ Xc / (len(X) - 1)        # (900,900)

    # PC-1  (odd R & Tx)
    v1, _ = top_eigvec(P1 @ Sigma @ P1)
    v1 = P1 @ v1
    v1 = v1 / jnp.linalg.norm(v1)

    # PC-2  (odd Tx only)
    v2, _ = top_eigvec(P2 @ Sigma @ P2)
    v2 = P2 @ v2
    v2 = v2 / jnp.linalg.norm(v2)  # already orthogonal to v1

    P = jnp.vstack([v1, v2])                 # (2,900)

    # Compute actual variances by projecting the data
    raw_projected = (P @ Xc.T).T  # Project centered data
    lam1 = jnp.var(raw_projected[:, 0])  # Actual variance of PC-1
    lam2 = jnp.var(raw_projected[:, 1])  # Actual variance of PC-2

    # Whitening: divide by square root of variances
    whitening_scale = 1.0 / jnp.sqrt(jnp.array([lam1, lam2]))

    def encode(X_new):
        projected = (P @ (X_new - mu).T).T       # (N,2)
        return projected * whitening_scale       # Apply whitening

    return P, mu, encode, (lam1, lam2)


# --------------------------------------------------------------------
#  Data processing and stratified sampling functions
# --------------------------------------------------------------------

@jax.jit
def get_extension(x):
    x = x.reshape(-1, 3)
    extension = jnp.max(x[:, 0]) - jnp.min(x[:, 0])
    return extension


def process_and_sample_data(dataset_name, batch_size=32, num_bins=32, samples_per_batch=64, seed=0):
    """
    Load dataset and perform stratified sampling by extension lengths
    Memory-optimized version that processes data in streaming fashion
    """
    logging.info(f"Loading and processing data from: {dataset_name}")

    # Set random seed for reproducibility
    np.random.seed(seed)
    logging.info(f"Set random seed to {seed} for reproducible sampling")

    # Load the specified dataset
    data = load_dataset(dataset_name).with_format("numpy")
    # Concatenate all available splits
    all_splits = []
    for split_name in data.keys():
        all_splits.append(data[split_name])
    data = concatenate_datasets(all_splits)

    x_data = []
    total_batches = len(data)//batch_size + (1 if len(data) % batch_size != 0 else 0)

    # Process data in streaming fashion to minimize memory usage
    for batch_idx, d in enumerate(tqdm(data.iter(batch_size), total=total_batches)):
        # Process current batch
        x = d["x"].reshape(-1, 900)
        extensions = jax.vmap(get_extension)(x)

        # Stratified sampling by extension lengths
        # Bin the extensions into bins
        bins = jnp.linspace(extensions.min(), extensions.max(), num_bins + 1)
        bin_indices = jnp.digitize(extensions, bins) - 1  # bin_indices in [0, num_bins-1]
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
                extra_idx = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                x_chosen.extend([x[i:i+1] for i in extra_idx])

        # Convert to jnp array and add to collection
        if x_chosen:
            x_chosen = jnp.concatenate(x_chosen, axis=0)
            x_data.append(x_chosen)

        # Explicitly delete large objects to free memory
        del x, extensions, bins, bin_indices, d

        # Periodic memory cleanup and progress reporting
        if (batch_idx + 1) % 100 == 0:
            current_samples = sum(len(chunk) for chunk in x_data)
            logging.info(f"  Processed {batch_idx + 1}/{total_batches} batches, collected {current_samples} samples")

    # Final concatenation
    x_data = jnp.concatenate(x_data, axis=0)
    logging.info(f"Processed and sampled {len(x_data)} samples for PCA fitting")
    return x_data


@hydra.main(version_base=None, config_path="config", config_name="polymer_dynamics_wi")
def main(config: DictConfig) -> None:
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Step 1: Process and sample data for PCA fitting
    logging.info("=" * 60)
    logging.info("STEP 1: DATA PROCESSING AND SAMPLING")
    logging.info("=" * 60)

    X = process_and_sample_data(
        config.data.generation.pca_source,
        batch_size=config.data.generation.batch_size,
        num_bins=config.data.generation.num_bins,
        samples_per_batch=config.data.generation.samples_per_batch,
        seed=config.data.generation.seed
    )

    # Step 2: Build PCA components
    logging.info("\n" + "=" * 60)
    logging.info("STEP 2: PCA FITTING")
    logging.info("=" * 60)

    X_jax = jnp.array(X)
    # Release the numpy array to save memory
    del X

    P, mu, encode, lams = build_two_PCs(X_jax)

    # PCA diagnostics
    logging.info("PCA DIAGNOSTICS")
    logging.info("-" * 40)
    logging.info(f"PC-1 eigenvalue (variance): {lams[0]:.6f}")
    logging.info(f"PC-2 eigenvalue (variance): {lams[1]:.6f}")
    logging.info(f"Variance ratio (PC-1/PC-2): {lams[0]/lams[1]:.6f}")
    logging.info(f"Whitening scales: [{1.0/jnp.sqrt(lams[0]):.6f}, {1.0/jnp.sqrt(lams[1]):.6f}]")

    # Test encoder on PCA fitting data
    logging.info("\n--- Testing encoder on PCA fitting data ---")
    all_projected = encode(X_jax)
    logging.info(f"PCA data shape: {X_jax.shape}")
    logging.info(f"Projected shape: {all_projected.shape}")

    # Check if whitening actually works
    pc1_var = jnp.var(all_projected[:, 0])
    pc2_var = jnp.var(all_projected[:, 1])
    logging.info(f"PC-1 variance after whitening: {pc1_var:.6f} (should be ~1.0)")
    logging.info(f"PC-2 variance after whitening: {pc2_var:.6f} (should be ~1.0)")

    # Release PCA fitting data after validation
    del X_jax, all_projected

    # Step 3: Define transformation functions
    def get_extension_transform(x):
        x = x.reshape(300, 3)
        ext_normalised = (jnp.max(x[:, 0]) - jnp.min(x[:, 0])) / 300.0
        return ext_normalised

    def pca_projection(x):
        return encode(x).ravel()

    def transform(x):
        z_star = get_extension_transform(x)
        z_hat = pca_projection(x)
        return jnp.concatenate([jnp.array([z_star]), z_hat])

    # Step 4: Transform train and test datasets
    logging.info("\n" + "=" * 60)
    logging.info("STEP 3: DATASET TRANSFORMATION")
    logging.info("=" * 60)

    # Apply transformation
    transform_vmap = jax.vmap(transform)

    # Initialize variables for final summary
    train_output_path = None
    test_output_path = None

    # Main transformation function
    def transform_dataset(dataset):
        return dataset.map(lambda x: {"x": transform_vmap(x["x"])})

    # Process train dataset first (if not skipped)
    if config.data.generation.skip_train:
        logging.info("Skipping training data processing (skip_train enabled)")
    else:
        logging.info(f"Loading and transforming train data from: {config.data.generation.train_dataset}")
        train_data = load_dataset(config.data.generation.train_dataset).with_format("numpy")

        # Processing train data
        logging.info("Processing train data...")
        train_data_pca = transform_dataset(train_data)

        # Save train data immediately and release memory
        train_output_path = f"{config.data.cache_path}_train"
        logging.info(f"Saving train data to: {train_output_path}")
        train_data_pca.save_to_disk(train_output_path)

        # Release train data memory
        del train_data, train_data_pca
        logging.info("Train data processed and saved, memory released")

    # Now process test dataset (if not skipped)
    if config.data.generation.skip_test:
        logging.info("\nSkipping test data processing (skip_test enabled)")
    else:
        logging.info(f"\nLoading and transforming test data from: {config.data.generation.test_dataset}")
        test_data = load_dataset(config.data.generation.test_dataset).with_format("numpy")

        # Processing test data
        logging.info("Processing test data...")
        test_data_pca = transform_dataset(test_data)

        # Save test data
        test_output_path = f"{config.data.cache_path}_test"
        logging.info(f"Saving test data to: {test_output_path}")
        test_data_pca.save_to_disk(test_output_path)

        # Release test data memory
        del test_data, test_data_pca
        logging.info("Test data processed and saved, memory released")

    logging.info("\n" + "=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"PCA fitted on sampled data from {config.data.generation.pca_source}")

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
