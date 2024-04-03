import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import random

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'  # Set any free port here.

def run(rank, size):
    recall_k = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_recall_k = [0.1, 0.2, 0.3, 0.4, 0.5]
    """ Distributed function to be implemented later. """
    torch.cuda.set_device(rank)  # Set the GPU device to the rank of the current process
    # Generate random dataset_result
    dataset_result = {'mean_recall_k': {}, 'recall_k': {}}
    for r in recall_k:
        dataset_result['recall_k']['recall_{}'.format(r)] = [random.random() for _ in range(10)]
    for r in mean_recall_k:
        dataset_result['mean_recall_k']['mean_recall_{}'.format(r)] = [random.random() for _ in range(10)]
    print(f"Rank {rank} dataset_result: {dataset_result}")
        
    for k1, v1 in dataset_result.items():
        tensor_list = [torch.tensor(v, device='cuda') for v in v1.values()]
        gathered_tensors = []
        for tensor in tensor_list:
            output_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output_tensor, tensor)
            gathered_tensors.append(torch.stack(output_tensor))
        for k2, tensor in zip(v1.keys(), gathered_tensors):
            dataset_result[k1][k2] = tensor.cpu().numpy().mean()

    if rank == 0:
        print(f"Rank {rank} dataset_result: {dataset_result}")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()