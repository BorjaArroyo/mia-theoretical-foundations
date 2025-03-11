from torch.utils.data import Subset

def get_denominator_dataset(target_class, full_dataset, samples_per_class):
    indices = []
    counts = {}
    for i, (_, label) in enumerate(full_dataset):
        if label == target_class:
            continue
        if label not in counts:
            counts[label] = 0
        if counts[label] < samples_per_class:
            indices.append(i)
            counts[label] += 1
    return Subset(full_dataset, indices)

def get_numerator_dataset(target_class, query_index, full_dataset, samples_per_class):
    indices = []
    counts = {}
    for i, (_, label) in enumerate(full_dataset):
        if label == target_class:
            if i == query_index:
                indices.append(i)
        else:
            if label not in counts:
                counts[label] = 0
            if counts[label] < samples_per_class:
                indices.append(i)
                counts[label] += 1
    return Subset(full_dataset, indices)