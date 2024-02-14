import numpy as np


def create_random_splits(reference_matchings):
    # Train/val/test mask calculations
    n_edges = len(reference_matchings)
    n_train = int(n_edges * 0.7)
    n_val = int(n_edges * 0.2)
    # leaves 0.1 for the test set

    train = np.zeros(n_edges, dtype=bool)
    val = np.zeros(n_edges, dtype=bool)
    test = np.zeros(n_edges, dtype=bool)

    train[:n_train] = True
    val[n_train : n_train + n_val] = True
    test[n_val + n_train :] = True

    # shuffling the reference matchings
    reference_matchings = reference_matchings.sample(frac=1).reset_index(drop=True)

    return (
        reference_matchings[train],
        reference_matchings[val],
        reference_matchings[test],
    )
