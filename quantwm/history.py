"""Historical channel energy and latest Q16; no metric evaluation."""
from collections import defaultdict
import torch

class HistoricalQueryStatsTracker:
    def __init__(self, *, layer_count, query_samples=16):
        self.layer_count = int(layer_count)
        self.query_samples = int(query_samples)
        self.call_index = 0
        self.pending_sum = {}
        self.pending_count = defaultdict(int)
        self.deferred_sum = {}
        self.deferred_count = defaultdict(int)
        self.history_sum = {}
        self.history_count = defaultdict(int)
        self.recent_queries = {}

    def wrap(self, original):
        def wrapped(q, k, v, *args, **kwargs):
            self.observe(q)
            return original(q, k, v, *args, **kwargs)
        return wrapped

    @torch.no_grad()
    def observe(self, q):
        layer = self.call_index % self.layer_count
        self.call_index += 1
        if q.ndim != 4:
            return
        energy = q.float().square().sum(dim=(0, 1))
        if layer in self.pending_sum:
            self.pending_sum[layer].add_(energy)
        else:
            self.pending_sum[layer] = energy.detach().clone()
        self.pending_count[layer] += int(q.shape[0] * q.shape[1])
        count = min(self.query_samples, q.shape[1])
        indices = torch.linspace(0, q.shape[1]-1, count, device=q.device).round().long()
        self.recent_queries[layer] = q.index_select(1, indices).float().permute(0, 2, 1, 3).detach()

    def drain_pending(self):
        batch = self.pending_sum, dict(self.pending_count)
        self.pending_sum = {}
        self.pending_count = defaultdict(int)
        return batch

    @torch.no_grad()
    def defer(self, batch):
        sums, counts = batch
        for layer, energy in sums.items():
            if layer in self.deferred_sum:
                self.deferred_sum[layer].add_(energy)
            else:
                self.deferred_sum[layer] = energy.detach().clone()
            self.deferred_count[layer] += int(counts[layer])

    @torch.no_grad()
    def commit_deferred(self):
        for layer, energy in self.deferred_sum.items():
            old = self.history_sum.get(layer)
            self.history_sum[layer] = old + energy if old is not None else energy.detach().clone()
            self.history_count[layer] += self.deferred_count[layer]
        self.deferred_sum = {}
        self.deferred_count = defaultdict(int)

    def get_weight(self, layer, *, heads, dim, device):
        value = self.history_sum.get(int(layer))
        count = self.history_count[int(layer)]
        if value is None or value.shape != (heads, dim) or count <= 0:
            return torch.ones((heads, dim), device=device, dtype=torch.float32)
        weight = value.to(device).float() / float(count)
        return weight / weight.mean(dim=-1, keepdim=True).clamp_min(1e-8)

    def get_queries(self, layer):
        return self.recent_queries.get(int(layer))
