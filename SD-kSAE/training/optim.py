from typing import Optional
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(
    scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs
):
    
    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")