# Tensor.index_put_ Dump And CUDA Replay

This note is for whole-network NPU failures around `x.index_put_(indices, value, accumulate=...)`.
The dump preserves tensor values plus view/alias metadata, so CUDA replay can reproduce overlap behavior.

## 1. Add Dump At The Call Site

Place this helper near the failing call site, or import it from a local debug file.

```python
import torch


def dump_index_put_case(path, self, indices, value, accumulate=False, unsafe=False):
    groups = []
    key_to_gid = {}

    def pack(t):
        if t is None:
            return None

        key = (str(t.device), t.untyped_storage().data_ptr())
        if key not in key_to_gid:
            gid = len(groups)
            key_to_gid[key] = gid

            # Save the whole storage as a 1D tensor. CUDA replay rebuilds the
            # original views with size/stride/storage_offset.
            n_elem = t.untyped_storage().nbytes() // t.element_size()
            base = torch.as_strided(t, (n_elem,), (1,), 0).detach().cpu().clone()
            groups.append({
                "base": base,
                "orig_device": str(t.device),
                "dtype": str(t.dtype),
            })

        return {
            "gid": key_to_gid[key],
            "size": tuple(t.size()),
            "stride": tuple(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": t.numel(),
        }

    torch.save(
        {
            "self": pack(self),
            "indices": [pack(i) for i in indices],
            "value": pack(value),
            "accumulate": bool(accumulate),
            "unsafe": bool(unsafe),
            "groups": groups,
        },
        path,
    )
    print(f"[dump index_put_] saved: {path}")
```

Use it directly before `x.index_put_`:

```python
dump_index_put_case(
    "/tmp/index_put_case.pt",
    x,
    indices,
    value,
    accumulate=accumulate,
    unsafe=False,
)

x.index_put_(indices, value, accumulate=accumulate)
```

If the call uses a literal accumulate value:

```python
dump_index_put_case("/tmp/index_put_case.pt", x, indices, value, accumulate=True)
x.index_put_(indices, value, accumulate=True)
```

## 2. CUDA Replay Script

Save this as `replay_index_put_cuda.py` and run it on a CUDA machine.

```python
import sys
import torch


case = torch.load(sys.argv[1], map_location="cpu")
cache = {}


def base(gid, device):
    key = (gid, device)
    if key not in cache:
        cache[key] = case["groups"][gid]["base"].to(device)
    return cache[key]


def rebuild(meta):
    if meta is None:
        return None

    # Preserve CPU tensors as CPU tensors. This matters for PyTorch's
    # index_put_ -> masked_fill_ fast path with CPU scalar values.
    device = "cpu" if meta["device"].startswith("cpu") else "cuda"
    return torch.as_strided(
        base(meta["gid"], device),
        tuple(meta["size"]),
        tuple(meta["stride"]),
        meta["storage_offset"],
    )


x = rebuild(case["self"])
indices = [rebuild(i) for i in case["indices"]]
value = rebuild(case["value"])

print("accumulate:", case["accumulate"], "unsafe:", case["unsafe"])
print("x:", x.device, x.dtype, tuple(x.shape), x.stride(), x.storage_offset())
print("value:", value.device, value.dtype, tuple(value.shape), value.stride(), value.storage_offset())
for n, idx in enumerate(indices):
    if idx is None:
        print(f"index[{n}]: None")
    else:
        print(f"index[{n}]:", idx.device, idx.dtype, tuple(idx.shape), idx.stride(), idx.storage_offset())

try:
    with torch.no_grad():
        x.index_put_(indices, value, accumulate=case["accumulate"])
    torch.cuda.synchronize()
    print("CUDA: PASS")
except Exception as e:
    print("CUDA: RAISED")
    print(type(e).__name__, e)
```

Run:

```bash
python replay_index_put_cuda.py /tmp/index_put_case.pt
```

## 3. Quick Print-Only Probe

Use this when you only need to inspect overlap metadata at the call site.
This is not enough to replay exact CUDA behavior because it does not save index values.

```python
def print_tensor_meta(name, t):
    if t is None:
        print(name, None)
        return
    print(
        name,
        "device=", t.device,
        "dtype=", t.dtype,
        "size=", tuple(t.size()),
        "stride=", tuple(t.stride()),
        "storage_offset=", t.storage_offset(),
        "data_ptr=", t.data_ptr(),
        "storage_ptr=", t.untyped_storage().data_ptr(),
        "numel=", t.numel(),
    )


print_tensor_meta("x", x)
print_tensor_meta("value", value)
for i, idx in enumerate(indices):
    print_tensor_meta(f"index[{i}]", idx)
print("accumulate=", accumulate)
```

## 4. How To Interpret Replay

- `CUDA: RAISED` with an overlap error: CUDA also rejects the case.
- `CUDA: PASS` but NPU raises: NPU is stricter than CUDA, or NPU misses CUDA's masked-fill fast path.
- NPU passes but `CUDA: RAISED`: NPU is looser than CUDA, especially suspicious for `accumulate=True` overlap cases.

Important details:

- Dumping each tensor independently with `.cpu().clone()` destroys shared-storage alias relationships.
- The helper saves storage groups plus `size`, `stride`, and `storage_offset` so replay can rebuild views.
- For `x.index_put_`, replay intentionally calls `x.index_put_` rather than `torch.ops.aten._index_put_impl_`, matching the original call site.
