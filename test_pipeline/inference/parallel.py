"""
Parallel execution utilities for batch processing.
"""
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from typing import Dict, List, Any, Callable, Optional, Tuple
from tqdm import tqdm
from omegaconf import DictConfig

from utils import dist_util


class ParallelExecutor:
    """
    Orchestrates batch processing in single or multi-process mode.
    
    Provides a unified interface for:
    - Single-process sequential execution
    - Multi-process parallel execution with GPU workers
    """
    
    def __init__(
        self,
        num_workers: int,
        device: torch.device,
    ):
        self.num_workers = num_workers
        self.device = device
    
    def execute(
        self,
        dataloader,
        process_fn: Callable,
        process_fn_kwargs: Dict[str, Any],
        get_seq_name_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Execute batch processing across all batches.
        
        Args:
            dataloader: PyTorch DataLoader
            process_fn: Function to process each batch
            process_fn_kwargs: Additional kwargs for process_fn
            get_seq_name_fn: Function to extract sequence name from batch
            
        Returns:
            Dictionary mapping sequence names to results
        """
        if self.num_workers == 0:
            return self._execute_single_process(
                dataloader, process_fn, process_fn_kwargs, get_seq_name_fn
            )
        else:
            return self._execute_multi_process(
                dataloader, process_fn, process_fn_kwargs, get_seq_name_fn
            )
    
    def _execute_single_process(
        self,
        dataloader,
        process_fn: Callable,
        process_fn_kwargs: Dict[str, Any],
        get_seq_name_fn: Callable,
    ) -> Dict[str, Any]:
        """Execute in single process (main thread)."""
        results = {}
        
        pbar = tqdm(total=len(dataloader), desc="Processing batches in single process")
        for batch_idx, batch in enumerate(dataloader):
            seq_name = get_seq_name_fn(batch, batch_idx)
            
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Process batch
            result = process_fn(batch, **process_fn_kwargs)
            
            if result is not None:
                results[seq_name] = result
            
            del batch, result
            pbar.update(1)
        
        pbar.close()
        return results
    
    def _execute_multi_process(
        self,
        dataloader,
        process_fn: Callable,
        process_fn_kwargs: Dict[str, Any],
        get_seq_name_fn: Callable,
    ) -> Dict[str, Any]:
        """Execute using multiple worker processes."""
        # Setup queues
        input_queue = Queue()
        output_queue = Queue()
        
        # Determine devices for workers
        devices = [f'cuda:{0}' for _ in range(self.num_workers)]
        
        # Start worker processes
        workers = []
        for worker_id, device in enumerate(devices):
            p = Process(
                target=self._worker_fn,
                args=(worker_id, input_queue, output_queue, process_fn, process_fn_kwargs, device)
            )
            p.start()
            workers.append(p)
        
        # Enqueue all batches
        for batch_idx, batch in enumerate(dataloader):
            seq_name = get_seq_name_fn(batch, batch_idx)
            batch = self._move_to_device(batch)
            input_queue.put((seq_name, batch))
        
        # Send termination signals
        for _ in range(self.num_workers):
            input_queue.put(None)
        
        # Collect results
        results = {}
        pbar = tqdm(total=len(dataloader), desc="Processing batches in multi process")
        
        received = 0
        while received < len(dataloader):
            seq_name, result = output_queue.get()
            if result is not None:
                results[seq_name] = result
            received += 1
            pbar.update(1)
        
        pbar.close()
        
        # Wait for workers to finish
        for p in workers:
            p.join()
        
        return results
    
    @staticmethod
    def _worker_fn(
        worker_id: int,
        input_queue: Queue,
        output_queue: Queue,
        process_fn: Callable,
        process_fn_kwargs: Dict[str, Any],
        device: str,
    ):
        """Worker process function."""
        torch.cuda.set_device(device)
        print(f"Worker {worker_id} started on {device}")
        
        while True:
            task = input_queue.get()
            if task is None:
                break
            
            seq_name, batch = task
            result = process_fn(batch, **process_fn_kwargs)
            output_queue.put((seq_name, result))
            print(f"Worker {worker_id} processed {seq_name}")
        
        print(f"Worker {worker_id} exiting")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


def get_seq_name_from_batch(
    batch: Dict[str, Any],
    batch_idx: int,
    slice_strategy: str,
    dataset_name: str,
) -> Any:
    """
    Extract sequence name from batch.
    
    For full_sequence mode with AMASS, uses the sequence path.
    Otherwise, uses batch index.
    """
    if slice_strategy == 'full_sequence' and dataset_name.lower() == 'amass':
        return batch['meta']['name'][0]
    return batch_idx
