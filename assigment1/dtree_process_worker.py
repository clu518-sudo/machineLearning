import numpy as np

def countLabels(labelList):
    labels = np.asarray(labelList, dtype=object)
    values, counts = np.unique(labels, return_counts=True)
    return {value: int(count) for value, count in zip(values.tolist(), counts.tolist())}

def splitDataSet(dataset, recentFeatureIndex, oneValueInRecentFeature):
    data_arr = np.asarray(dataset, dtype=object)
    if data_arr.size == 0:
        return data_arr

    match_mask = data_arr[:, recentFeatureIndex] == oneValueInRecentFeature
    filtered = data_arr[match_mask]
    if filtered.size == 0:
        return filtered
    return np.delete(filtered, recentFeatureIndex, axis=1)

def calEntropy(dataset):
    data_arr = np.asarray(dataset, dtype=object)
    if data_arr.size == 0:
        return 0.0

    labels = data_arr[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))
    
def compute_feature_gain(args):
    """Must stay at module top-level for multiprocessing pickling."""
    dataset, i, basicEntropy = args
    data_arr = np.asarray(dataset, dtype=object)
    uniqueFeatureValue = np.unique(data_arr[:, i])
    localEntropy = 0.0
    total_rows = float(len(data_arr))
    for fValue in uniqueFeatureValue:
        subDataSet = splitDataSet(data_arr, i, fValue)
        percentage = len(subDataSet) / total_rows
        localEntropy += percentage * calEntropy(subDataSet)
    return i, basicEntropy - localEntropy


def build_class_folds_worker(args):
    """Create per-class fold index chunks in a subprocess-safe helper."""
    class_value, class_indices, k, seed = args
    indices = np.asarray(class_indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_indices = np.array_split(indices, int(k))
    return class_value, [chunk.astype(np.int64) for chunk in split_indices]