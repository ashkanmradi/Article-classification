from timeit import default_timer
import torch
from torch.utils.data import DataLoader

class Timer:
    """
    Timer utilities for benchmarking running times of executions
    Provides a context manager as well. 

    Examples:
        >>> import time
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> with Timer() as t:
        ...   time.sleep(1)
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 1...'
    """

    def __init__(self):
        self._timer = default_timer
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = self._timer()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() "
                "as t:"
            )

    @property
    def interval(self):
        if self.running:
            raise ValueError("Trime has not been stopped, please use stop().")
        else:
            return self._interval


class PyTorchUtils:
    @staticmethod
    def get_amp(fp16: bool = False):
        """This function ensures that fp16 execution of torch.einsum is enabled
            if fp16 is set. Otherwise, it'll default to "promote" mode,
            where the operations are in fp32.
            Note that setting `fp16_opt_level="O2"` will remove the need for this code.
        """
        if fp16:
            try:
                # from apex import amp

                amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex"
                )
        else:
            amp = None
        return amp

    @staticmethod
    def get_device(num_gpus: int = None, gpu_ids: list = None, local_rank: int = -1):
        if gpu_ids is not None:
            num_gpus = len(gpu_ids)
        if local_rank == -1:
            num_gpus = (
                min(num_gpus, torch.cuda.device_count())
                if num_gpus is not None
                else torch.cuda.device_count()
            )
            device = torch.device(
                "cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
            )
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            num_gpus = 1
        return device, num_gpus

    @staticmethod
    def move_model_to_device(model, device):
        if not isinstance(device, torch.device):
            raise ValueError("device must be of type torch.device.")

        # unwrap model
        # if isinstance(model, torch.nn.DataParallel):
        model = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training

        # move to device
        return model.to(device)

    @staticmethod
    def parallelize_model(
        model, device, num_gpus: int = None, gpu_ids: list = None, local_rank: int = -1
    ):
        """Moves a model to the specified device (cpu or gpu/s)
        and implements data parallelism when multiple gpus are specified.
        Args:
            model (Module): A PyTorch model.
            device (torch.device): A PyTorch device.
            num_gpus (int): The number of GPUs to be used.
                If set to None, all available GPUs will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If None, the first num_gpus GPUs will be used.
                If not None, overrides num_gpus. if gpu_ids is an empty list
                or there is no valid gpu devices are specified,
                and device is "cuda", model will not be moved or parallelized.
                Defaults to None.
            local_rank (int): Local GPU ID within a node. Used in distributed environments.
                If not -1, num_gpus and gpu_ids are ignored.
                Defaults to -1.
        Returns:
            Module, DataParallel, DistributedDataParallel: A PyTorch Module or
                a DataParallel/DistributedDataParallel wrapper,
                when one or multiple gpus are used.
        """
        if not isinstance(device, torch.device):
            raise ValueError("device must be of type torch.device.")

        model_module = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training

        if local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model_module,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            if device.type == "cuda":
                if num_gpus is not None:
                    if num_gpus < 1:
                        raise ValueError("num_gpus must be at least 1 or None")
                num_cuda_devices = torch.cuda.device_count()
                if num_cuda_devices < 1:
                    raise Exception("CUDA devices are not available.")
                if gpu_ids is None:
                    num_gpus = (
                        num_cuda_devices
                        if num_gpus is None
                        else min(num_gpus, num_cuda_devices)
                    )
                    gpu_ids = list(range(num_gpus))
                else:
                    gpu_ids = list(
                        set(list(range(num_cuda_devices))).intersection(gpu_ids)
                    )
                if len(gpu_ids) > 0:
                    model = torch.nn.DataParallel(model_module, device_ids=gpu_ids)
        return model

    @staticmethod
    def compute_training_steps(
        dataloader: DataLoader,
        num_epochs: int = 1,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
    ):
        """Computes the max training steps given a dataloader.

            Args:
                dataloader (Dataloader): A PyTorch DataLoader.
                num_epochs (int, optional): Number of training epochs. Defaults to 1.
                max_steps (int, optional): Total number of training steps.
                    If set to a positive value, it overrides num_epochs.
                    Otherwise, it's determined by the dataset length,
                    gradient_accumulation_steps, and num_epochs.
                    Defaults to -1.
                gradient_accumulation_steps (int, optional): Number of steps to accumulate
                    before performing a backward/update pass.
                    Default to 1.

            Returns:
                int: The max number of steps to be used in a training loop.
            """
        try:
            dataset_length = len(dataloader)
        except Exception:
            dataset_length = -1
        if max_steps <= 0:
            if dataset_length != -1 and num_epochs > 0:
                max_steps = dataset_length // gradient_accumulation_steps * num_epochs
        if max_steps <= 0:
            raise Exception("Max steps cannot be determined.")
        return max_steps
    