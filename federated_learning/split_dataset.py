import random
import numpy as np

def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    elif fed_args.split_strategy == 'quantity_skew':
        local_datasets = partition_dataset_with_quantity_skew(fed_args, dataset)
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round

def partition_dataset_with_quantity_skew(fed_args, raw_dataset):
    total_size = len(raw_dataset)
    partition_proportions = np.random.dirichlet(alpha=[fed_args.concentration] * fed_args.num_clients)
    cum_indices = np.cumsum(np.floor(partition_proportions * total_size).astype(int))

    cum_indices[-1] = total_size
    partitions = [
        raw_dataset.select(range(start, end))
        for start, end in zip([0] + cum_indices[:-1].tolist(), cum_indices.tolist())
    ]
    
    return partitions