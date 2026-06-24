import torch.profiler as profiler
import torch
import time
from collections import defaultdict

def is_leaf_module(module):
    return len(list(module.children())) == 0

class ModuleProfiler:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.stats = defaultdict(lambda: {
            "time": 0.0,
            "cuda_mem": 0,
            "params": 0
        })

    def _register_params(self):
        for name, module in self.model.named_modules():
            if name == "":
                continue
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.stats[name]["params"] = params

    def _make_hook(self, name):
        def hook(module, inputs, outputs):
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
            start_time = time.time()

            # forward 已经执行完，这里是“after hook”
            torch.cuda.synchronize()
            end_time = time.time()
            end_mem = torch.cuda.memory_allocated()

            self.stats[name]["time"] += (end_time - start_time)
            self.stats[name]["cuda_mem"] += max(end_mem - start_mem, 0)

        return hook

    def start(self):
        self._register_params()
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if not is_leaf_module(module):
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def stop(self):
        for h in self.handles:
            h.remove()

    def summary(self):
        total_time = sum(v["time"] for v in self.stats.values())
        total_mem = sum(v["cuda_mem"] for v in self.stats.values())

        print("\n[ Module-wise Profiling ]")
        print(
            f"{'Module':40s} | {'Params':>10s} | "
            f"{'Time(ms)':>10s} | {'Time%':>6s} | "
            f"{'Mem(MB)':>10s} | {'Mem%':>6s}"
        )
        print("-" * 95)

        for name, v in sorted(
            self.stats.items(),
            key=lambda x: x[1]["cuda_mem"],
            reverse=True
        ):
            if v["time"] == 0 and v["cuda_mem"] == 0:
                continue

            time_ms = v["time"] * 1000
            mem_mb = v["cuda_mem"] / 1024**2

            print(
                f"{name:40s} | "
                f"{v['params']:10d} | "
                f"{time_ms:10.2f} | "
                f"{(time_ms/total_time*100 if total_time else 0):6.2f} | "
                f"{mem_mb:10.2f} | "
                f"{(mem_mb/total_mem*100 if total_mem else 0):6.2f}"
            )

def get_one_batch(exp, flag='train'):
    """
    从你现有的 Experiment / Exp 类里拿一个 batch
    """
    _, loader = exp._get_data(flag=flag)
    return next(iter(loader))


def prepare_batch(batch, device, args):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    if batch_x_mark is not None:
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    dec_inp = torch.cat(
        [batch_y[:, :args.label_len, :], dec_inp],
        dim=1
    ).to(device)

    return batch_x, batch_x_mark, dec_inp, batch_y_mark


def profile_one_forward(exp):
    model = exp.model
    args = exp.args
    device = exp.device

    model.eval()

    batch = get_one_batch(exp)
    batch_x, batch_x_mark, dec_inp, batch_y_mark = prepare_batch(
        batch, device, args
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ===== 模块 profiler =====
    mod_profiler = ModuleProfiler(model)
    mod_profiler.start()

    with torch.no_grad():
        model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    torch.cuda.synchronize()
    mod_profiler.stop()

    # ===== 输出 =====
    print("\n[ Peak CUDA Memory ]")
    print(f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    mod_profiler.summary()
