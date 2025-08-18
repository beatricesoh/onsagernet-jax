from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
import jax
import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------
#  Log transformation for args data
# --------------------------------------------------------------------

def log_transform_dataset(data: Dataset) -> Dataset:
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


# --------------------------------------------------------------------
#  Build two symmetry-constrained principal components
# --------------------------------------------------------------------

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

I, R, Tx, Ty, Tz = make_ops()

# ---- projector for a given character --------------------------------
def projector(chR, chTx, chTy, chTz):
    ops = [I, R, Tx, Ty, Tz,
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
    return 0.0625 * sum(s*o for s, o in zip(signs, ops))  # 1/16 factor

# projectors for the two sectors
P1 = projector(chR=-1, chTx=-1, chTy=+1, chTz=+1)   # PC-1
P2 = projector(chR=+1, chTx=-1, chTy=+1, chTz=+1)   # PC-2

# ---- dominant eigenvector of a symmetric matrix ---------------------
def top_eigvec(S):
    vals, vecs = jnp.linalg.eigh(S)          # ascending order
    v = vecs[:, -1]
    λ = vals[-1]
    return v, λ

# ---- main builder ----------------------------------------------------
def build_two_PCs(X, eps=1e-12):
    """
    Original PCA method without manual scaling
    X  : (N, 900) flattened chains, inter-leaved (x1,y1,z1,…)
    returns  P (2×900), mu (1×900), encode callable, eigenvalues
    """
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
    λ1 = jnp.var(raw_projected[:, 0])  # Actual variance of PC-1
    λ2 = jnp.var(raw_projected[:, 1])  # Actual variance of PC-2

    # Whitening: divide by square root of variances
    whitening_scale = 1.0 / jnp.sqrt(jnp.array([λ1, λ2]))

    def encode(X_new):
        projected = (P @ (X_new - mu).T).T       # (N,2)
        return projected * whitening_scale       # Apply whitening

    return P, mu, encode, (λ1, λ2)


# --------------------------------------------------------------------
#  Data processing and stratified sampling functions
# --------------------------------------------------------------------

@jax.jit
def get_extension(x):
    x = x.reshape(-1, 3)
    extension = jnp.max(x[:, 0]) - jnp.min(x[:, 0])
    return extension


def process_and_sample_data(dataset_name, batch_size=32, num_bins=32, samples_per_batch=64):
    """
    Load dataset and perform stratified sampling by extension lengths
    Memory-optimized version that processes data in streaming fashion
    """
    print(f"Loading and processing data from: {dataset_name}")

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
        bins = np.linspace(extensions.min(), extensions.max(), num_bins + 1)
        bin_indices = np.digitize(extensions, bins) - 1  # bin_indices in [0, num_bins-1]
        x_chosen = []
        for i in range(num_bins):
            idx_in_bin = np.where(bin_indices == i)[0]
            if len(idx_in_bin) > 0:
                # Randomly pick one sample from this bin
                chosen_idx = np.random.choice(idx_in_bin, size=1)
                x_chosen.append(x[chosen_idx])

        # If less than samples_per_batch bins have samples, randomly fill up to samples_per_batch
        if len(x_chosen) < samples_per_batch:
            remaining = samples_per_batch - len(x_chosen)
            all_indices = np.arange(x.shape[0])
            already_chosen = np.concatenate(x_chosen).reshape(-1, x.shape[1])
            mask = np.ones(x.shape[0], dtype=bool)
            for arr in x_chosen:
                mask[np.where((x == arr).all(axis=1))[0][0]] = False
            remaining_indices = all_indices[mask]
            if len(remaining_indices) > 0:
                extra_idx = np.random.choice(remaining_indices, size=min(remaining, len(remaining_indices)), replace=False)
                x_chosen.extend([x[i:i+1] for i in extra_idx])

        # Convert to numpy and add to collection
        if x_chosen:
            x_chosen = np.concatenate(x_chosen, axis=0)
            x_data.append(x_chosen)

        # Explicitly delete large objects to free memory
        del x, extensions, bins, bin_indices, d

        # Periodic memory cleanup and progress reporting
        if (batch_idx + 1) % 100 == 0:
            current_samples = sum(len(chunk) for chunk in x_data)
            print(f"  Processed {batch_idx + 1}/{total_batches} batches, collected {current_samples} samples")

    # Final concatenation
    x_data = np.concatenate(x_data, axis=0)
    print(f"Processed and sampled {len(x_data)} samples for PCA fitting")
    return x_data


@hydra.main(version_base=None, config_path="config", config_name="polymer_dynamics_wi")
def main(cfg: DictConfig) -> None:
    # Step 1: Process and sample data for PCA fitting
    print("=" * 60)
    print("STEP 1: DATA PROCESSING AND SAMPLING")
    print("=" * 60)

    X = process_and_sample_data(
        cfg.data.generation.pca_source,
        batch_size=cfg.data.generation.batch_size,
        num_bins=cfg.data.generation.num_bins,
        samples_per_batch=cfg.data.generation.samples_per_batch
    )

    # Step 2: Build PCA components
    print("\n" + "=" * 60)
    print("STEP 2: PCA FITTING")
    print("=" * 60)

    X_jax = jnp.array(X)
    # Release the numpy array to save memory
    del X

    P, mu, encode, lams = build_two_PCs(X_jax)

    print("PCA DIAGNOSTICS")
    print("-" * 40)
    print(f"PC-1 eigenvalue (variance): {lams[0]:.6f}")
    print(f"PC-2 eigenvalue (variance): {lams[1]:.6f}")
    print(f"Variance ratio (PC-1/PC-2): {lams[0]/lams[1]:.6f}")
    print(f"Whitening scales: [{1.0/jnp.sqrt(lams[0]):.6f}, {1.0/jnp.sqrt(lams[1]):.6f}]")

    # Test encoder on PCA fitting data
    print("\n--- Testing encoder on PCA fitting data ---")
    all_projected = encode(X_jax)
    print(f"PCA data shape: {X_jax.shape}")
    print(f"Projected shape: {all_projected.shape}")

    # Check if whitening actually works
    pc1_var = jnp.var(all_projected[:, 0])
    pc2_var = jnp.var(all_projected[:, 1])
    print(f"PC-1 variance after whitening: {pc1_var:.6f} (should be ~1.0)")
    print(f"PC-2 variance after whitening: {pc2_var:.6f} (should be ~1.0)")

    # Release PCA fitting data after validation
    del X_jax, all_projected

    # Step 3: Define transformation functions
    def get_extension_transform(x):
        x = x.reshape(300, 3)
        ext_normalised = (jnp.max(x[:, 0]) - jnp.min(x[:, 0])) / 300.0
        return ext_normalised

    def pca_projection(x):
        return encode(x).ravel()

    def flip_test_data_ordering(x):
        """
        Flip test data dimensions:
        [NUM_STEPS, 900] -> [NUM_STEPS, 3, 300] -> transpose last 2 dims -> [NUM_STEPS, 300, 3] -> [NUM_STEPS, 900]
        """
        # x shape: [NUM_STEPS, 900]
        num_steps = x.shape[0]

        # Reshape to [NUM_STEPS, 3, 300]
        x_reshaped = x.reshape(num_steps, 3, 300)

        # Transpose last 2 dimensions to get [NUM_STEPS, 300, 3]
        x_transposed = jnp.transpose(x_reshaped, (0, 2, 1))

        # Reshape back to [NUM_STEPS, 900]
        x_flipped = x_transposed.reshape(num_steps, 900)

        return x_flipped

    def transform(x):
        z_star = get_extension_transform(x)
        z_hat = pca_projection(x)
        return jnp.concatenate([jnp.array([z_star]), z_hat])

    # Step 4: Transform train and test datasets
    print("\n" + "=" * 60)
    print("STEP 3: DATASET TRANSFORMATION")
    print("=" * 60)

    # Apply transformation
    transform_vmap = jax.vmap(transform)

    # Initialize variables for final summary
    train_output_path = None
    test_output_path = None

    # Process train dataset first (if not skipped)
    if cfg.data.generation.skip_train:
        print("⏭️  Skipping training data processing (skip_train enabled)")
    else:
        print(f"Loading and transforming train data from: {cfg.data.generation.train_dataset}")
        train_data = load_dataset(cfg.data.generation.train_dataset).with_format("numpy")

        # Apply log transformation if enabled
        if cfg.data.generation.log_transform:
            print("Applying log transformation to train data...")
            train_data = log_transform_dataset(train_data)

        # Processing train data
        print("Processing train data...")
        train_data_pca = train_data.map(lambda x: {"x": transform_vmap(x["x"])})

        # Save train data immediately and release memory
        train_output_path = f"{cfg.data.cache_path}_train"
        print(f"Saving train data to: {train_output_path}")
        train_data_pca.save_to_disk(train_output_path)

        # Release train data memory
        del train_data, train_data_pca
        print("✓ Train data processed and saved, memory released")

    # Now process test dataset (if not skipped)
    if cfg.data.generation.skip_test:
        print("\n⏭️  Skipping test data processing (skip_test enabled)")
    else:
        print(f"\nLoading and transforming test data from: {cfg.data.generation.test_dataset}")
        test_data = load_dataset(cfg.data.generation.test_dataset).with_format("numpy")

        # Apply log transformation if enabled
        if cfg.data.generation.log_transform:
            print("Applying log transformation to test data...")
            test_data = log_transform_dataset(test_data)

        # Define test data transformation function based on flip flag
        if cfg.data.generation.flip_test:
            print("🔄 Flip mode enabled: Will reorder test data dimensions before transformation")
            print("   [NUM_STEPS, 900] -> [NUM_STEPS, 3, 300] -> transpose -> [NUM_STEPS, 300, 3] -> [NUM_STEPS, 900]")

            def transform_test_data(batch):
                # First flip the data ordering
                x_flipped = flip_test_data_ordering(batch["x"])
                # Then apply the usual transformation
                return {"x": transform_vmap(x_flipped)}
        else:
            print("📋 Standard mode: Applying transformation without dimension reordering")

            def transform_test_data(batch):
                return {"x": transform_vmap(batch["x"])}

        # Processing test data
        print("Processing test data...")
        test_data_pca = test_data.map(transform_test_data)

        # Save test data
        test_output_path = f"{cfg.data.cache_path}_test"
        print(f"Saving test data to: {test_output_path}")
        test_data_pca.save_to_disk(test_output_path)

        # Release test data memory
        del test_data, test_data_pca
        print("✓ Test data processed and saved, memory released")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"PCA fitted on sampled data from {cfg.data.generation.pca_source}")

    if train_output_path:
        print(f"Train data transformed and saved to: {train_output_path}")
    else:
        print("Train data processing skipped")

    if test_output_path:
        print(f"Test data transformed and saved to: {test_output_path}")
        if cfg.data.generation.flip_test:
            print("🔄 Test data processed with dimension reordering enabled")
    else:
        print("Test data processing skipped")

    print(f"Transform: [extension, PC-1_whitened, PC-2_whitened]")
    print(f"PC variance ratio: {lams[0]/lams[1]:.3f}")
    print("Memory optimized: datasets processed sequentially to minimize peak usage")


if __name__ == "__main__":
    main()
