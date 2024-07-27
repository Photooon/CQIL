import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class SyncInWrapper(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, x):
        # print(self.rank, "[SyncIn Broadcast] Start", x)
        dist.broadcast(x, src=0)
        # print(self.rank, "[SyncIn Broadcast] Finished", x)
        return (x, )

class SyncOutWrapper(nn.Module):
    def __init__(self, rank, model):
        super().__init__()
        self.rank = rank
        self.model = model
    
    def forward(self, x):
        ops = []
        # Bypass
        d = 0
        world_size = 2
        if d != 0:
            for i in range(d):
                if self.rank+i+1 < world_size:
                    send_op = dist.P2POp(dist.isend, x, self.rank+i+1)
                    ops.append(send_op)
                if self.rank-i-1 >= 0:
                    recv_tensor = torch.zeros_like(x)
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, self.rank-i-1)
                    ops.append(recv_op)
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Calculate
        out = self.model(x)[0] - x
        # all_reduce output
        # print(self.rank, "[SyncOut AllReduce] Start", out)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out += x
        # print(self.rank, "[SyncOut AllReduce] Finished", out)
        return (out, )

class JustPrint(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, x):
        print(self.rank, "Just Print")
        return x

def worker(rank, device_count, sub_model, input_shape_queue):
    dist.init_process_group("nccl", rank=rank, world_size=device_count)
    torch.cuda.set_device(rank)
    sub_model.cuda()

    while True:
        x = input_shape_queue.get()
        if x == None:
            break
        x = torch.empty(x, dtype=torch.bfloat16).cuda()
        dist.barrier()

        with torch.no_grad():
            for layer in sub_model:
                x = layer(x)[0]

class CQILWrapper(nn.Module):
    def __init__(self, model, start_l, end_l, device_count=2, port="29501"):
        super().__init__()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("Warning: CQILWrapper only support evaluation now.")
        model.eval()

        self.workers = []
        self.input_shape_queue = mp.Queue()
        self.device_count = device_count

        # Model split and reallocate
        l_ids_pre = list(range(0, start_l))
        l_ids_post = list(range(end_l, len(model.model.layers)))
        self.layers_pre = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i in l_ids_pre])
        self.layers_post = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i in l_ids_post])

        l_ids_ranks = [[] for _ in range(device_count)]
        for i in range(start_l, end_l):
            l_ids_ranks[(i - start_l)%self.device_count].append(i)
        self.layers_rank0 = nn.ModuleList([SyncInWrapper(0)] + [SyncOutWrapper(0, layer) for i, layer in enumerate(model.model.layers) if i in l_ids_ranks[0]])
        for rank in range(1, device_count):
            layers = nn.ModuleList([SyncInWrapper(rank)] + [SyncOutWrapper(rank, layer) for i, layer in enumerate(model.model.layers) if i in l_ids_ranks[rank]])
            proc = mp.Process(target=worker, args=(rank, device_count, layers, self.input_shape_queue))
            self.workers.append(proc)
            proc.start()
        
        dist.init_process_group("nccl", rank=0, world_size=device_count)
        torch.cuda.set_device(0)

        print("Info: CQIL workers initialized.")

        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, x):
        x = self.embed_tokens(x.to(self.embed_tokens.weight.device))
        for layer in self.layers_pre:
            x = layer(x)[0]

        for _ in range(1, self.device_count):
            self.input_shape_queue.put(list(x.shape))
        dist.barrier()
        for layer in self.layers_rank0:
            x = layer(x)[0]

        for layer in self.layers_post:
            x = layer(x)[0]

        x = self.norm(x)
        outputs = self.lm_head(x)

        return (outputs, )

    def close(self):
        for _ in range(self.device_count):
            self.input_shape_queue.put(None)
        for worker in self.workers:
            worker.join()
        print("Info: CQIL closed")
