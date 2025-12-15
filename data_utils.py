import types
import torch
import random
import math
from typing import Sequence, List
from torch.utils.data import Sampler
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np

def compute_proportions(labels, num_classes):
    """Compute normalized label counts for a batch."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts / counts.sum()

class FixedBatchSampler(Sampler[List[int]]):
    """Yield predefined lists of indices as batches."""

    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def create_fixed_proportion_batches(dataset, teacher_probs_list, bag_size, num_classes, seed: int | None = None):
    """Return a ``FixedBatchSampler`` matching the specified class proportions.

    Parameters
    ----------
    dataset : Dataset
        Dataset from which to draw samples.
    teacher_probs_list : Sequence[Sequence[float]]
        Desired class proportions for each bag.
    bag_size : int
        Number of samples per bag.
    num_classes : int
        Total number of classes.
    seed : int | None, default None
        Random seed controlling the shuffling of class indices. Using the same
        seed will yield identical bag composition.
    """
    dataset_indices = list(range(len(dataset)))

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in dataset_indices:
        root_idx = idx
        ds = dataset
        # Resolve the index through potentially nested Subset objects
        while hasattr(ds, "indices"):
            root_idx = ds.indices[root_idx]
            ds = ds.dataset
        label = int(targets[root_idx])
        if label < num_classes:
            # store dataset-relative index
            class_to_indices[label].append(idx)

    rng = random.Random(seed) if seed is not None else random
    for idx_list in class_to_indices.values():
        rng.shuffle(idx_list)

    batches = []
    for probs in teacher_probs_list:
        raw = [p * bag_size for p in probs]
        counts = [math.floor(c) for c in raw]
        remaining = bag_size - sum(counts)
        fractions = [r - math.floor(r) for r in raw]
        for cls in sorted(range(num_classes), key=lambda i: fractions[i], reverse=True)[:remaining]:
            counts[cls] += 1

        batch = []
        for cls, count in enumerate(counts):
            batch.extend(class_to_indices[cls][:count])
            class_to_indices[cls] = class_to_indices[cls][count:]
        batches.append(batch)

    return FixedBatchSampler(batches)


def create_random_bags(dataset, bag_size, num_classes, shuffle=True, seed: int | None = None):
    """Create random bags and return a sampler and teacher label proportions.

    Parameters
    ----------
    dataset : Dataset
        Dataset from which to draw samples.
    bag_size : int
        Number of samples in each bag.
    num_classes : int
        Total number of classes.
    shuffle : bool, default True
        Whether to shuffle dataset indices before grouping into bags.
    seed : int | None, default None
        Seed for the shuffling process. Supplying a seed makes the bag
        formation deterministic.
    """
    dataset_indices = list(range(len(dataset)))
    rng = random.Random(seed) if seed is not None else random
    if shuffle:
        rng.shuffle(dataset_indices)

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    batches = []
    teacher_props = []
    # ignore last incomplete batch
    full_len = len(dataset_indices) - len(dataset_indices) % bag_size
    for start in range(0, full_len, bag_size):
        batch_indices = dataset_indices[start : start + bag_size]
        batches.append(batch_indices)

        labels = []
        for idx in batch_indices:
            root_idx = idx
            ds = dataset
            while hasattr(ds, "indices"):
                root_idx = ds.indices[root_idx]
                ds = ds.dataset
            label = int(targets[root_idx])
            if label < num_classes:
                labels.append(label)
        teacher_props.append(compute_proportions(torch.tensor(labels), num_classes))

    sampler = FixedBatchSampler(batches)
    teacher_tensor = torch.stack(teacher_props)
    return sampler, teacher_tensor


def load_pt_features(
    train_path: str,
    test_path: str,
    pca_dim: int | None = None,
    *,
    fraction: float | None = None,
    fraction_seed: int | None = None,
):
    """Load feature tensors from ``.pt`` files and optionally apply PCA.

    Parameters
    ----------
    train_path : str
        Path to the training ``.pt`` file containing ``features`` and ``labels`` tensors.
    test_path : str
        Path to the test ``.pt`` file containing ``features`` and ``labels`` tensors.
    pca_dim : int | None, optional
        If provided, PCA is applied to reduce the feature dimensionality to ``pca_dim``.
    fraction : float | None, optional
        When set, subsample this fraction of each class from the training set. The value
        must satisfy ``0 < fraction <= 1``.
    fraction_seed : int | None, optional
        Seed passed to ``np.random.default_rng`` when selecting the subsample indices.
    """
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    X_train = train_data["features"].cpu().numpy()
    y_train = train_data["labels"].cpu().numpy()
    X_test = test_data["features"].cpu().numpy()
    y_test = test_data["labels"].cpu().numpy()

    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must satisfy 0 < fraction <= 1")

        classes, counts = np.unique(y_train, return_counts=True)
        rng = np.random.default_rng(fraction_seed)
        per_class_indices: List[np.ndarray] = []
        for cls, count in zip(classes, counts):
            class_indices = np.flatnonzero(y_train == cls)
            target_size = min(count, max(1, int(count * fraction)))
            chosen = rng.choice(class_indices, size=target_size, replace=False)
            per_class_indices.append(chosen)

        selected_indices = np.concatenate(per_class_indices)
        rng.shuffle(selected_indices)
        X_train = X_train[selected_indices]
        y_train = y_train[selected_indices]

    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"PCA reduced shape: {X_train.shape}")

    return X_train, X_test, y_train, y_test
