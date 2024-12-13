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

def partition_dataset_with_disease_classes(fed_args, raw_dataset, Categories):
   # Initialize empty partitions for each category
    partitions = {category: [] for category in Categories}

    # Iterate over the dataset and assign records to categories
    for idx in range(len(raw_dataset)):
        query = raw_dataset[idx]["instruction"]
        response = raw_dataset[idx]["response"]

        # Assign to categories based on keywords
        assigned = False
        for category, keywords in Categories.items():
            if any(keyword in query or keyword in response for keyword in keywords):
                partitions[category].append(idx)
                assigned = True
                break
        
        # If no category matches, classify as "Others"
        if not assigned:
            partitions["Others"].append(idx)

    # Convert indices to subsets of the raw dataset
    client_partitions = [
        raw_dataset.select(indices) for indices in partitions.values()
    ]

    return client_partitions