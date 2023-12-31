from torch.optim import RAdam, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup


def get_optimizer(model, lr: str, warmupSteps: int, total_train_steps: int,
                  scheduler: str = "cosine", optimizer: str = "adamw",
                  weight_decay: float = 0.01, hard_rr: int = 0):
    if optimizer == "radam":
        optimizer = RAdam(params=model.parameters(), lr=float(lr), weight_decay=weight_decay)
    else:
        optimizer = AdamW(params=model.parameters(), lr=float(lr), weight_decay=weight_decay)

    # Instantiate scheduler
    if scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmupSteps,
            num_training_steps=total_train_steps
        )
    elif scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmupSteps,
            num_training_steps=total_train_steps
        )
    elif scheduler == "cosine_hard":
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmupSteps,
            num_training_steps=total_train_steps,
            num_cycles=hard_rr
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmupSteps
        )

    return optimizer, lr_scheduler
