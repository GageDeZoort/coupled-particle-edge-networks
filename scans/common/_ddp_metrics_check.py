"""Throwaway DDP correctness check for running_train_metrics (no compute())."""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from cpen.lit_models.base_lit_cpen import BaseLitCPEN


class DummyModel(nn.Module):
    out_dim = 2
    operator_names: list[str] = []

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, **kwargs):
        return self.lin(x)


class Lit(BaseLitCPEN):
    def __init__(self):
        super().__init__(DummyModel(), eta_0=0.1)


def worker(rank: int, world: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    lit = Lit()
    lit.on_train_epoch_start()

    # Rank 0 sees 4 correct preds, rank 1 sees 2 correct / 2 wrong.
    if rank == 0:
        probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        y = torch.tensor([0, 0, 1, 1])
        loss = torch.tensor(1.0)
    else:
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        y = torch.tensor([0, 0, 0, 1])  # 3 correct actually: idx0 ok, idx1 pred1 true0 wrong, idx2 pred0 true0 ok, idx3 ok
        loss = torch.tensor(3.0)

    lit.train_acc(probs, y)
    lit.train_loss_running.update(loss, weight=torch.tensor(float(y.numel())))

    m = lit.running_train_metrics(sync=True)

    # Gather each rank's result to confirm they agree.
    payload = torch.tensor([m["train_loss"], m["train_acc"]])
    gathered = [torch.zeros_like(payload) for _ in range(world)]
    dist.all_gather(gathered, payload)

    if rank == 0:
        r0, r1 = gathered[0], gathered[1]
        agree = torch.allclose(r0, r1, atol=1e-6)
        # Expected loss: (1*4 + 3*4)/8 = 2.0
        # Expected acc (micro): correct = rank0 4 + rank1 3 = 7 of 8 = 0.875
        exp_loss, exp_acc = 2.0, 0.875
        ok_loss = abs(float(r0[0]) - exp_loss) < 1e-6
        ok_acc = abs(float(r0[1]) - exp_acc) < 1e-6
        print(f"rank0={r0.tolist()} rank1={r1.tolist()} agree={agree} "
              f"ok_loss={ok_loss} ok_acc={ok_acc}")
        print("RESULT:", "PASS" if (agree and ok_loss and ok_acc) else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
