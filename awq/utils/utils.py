import gc
import importlib
import torch
import accelerate


ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
try:
    import triton as tl
    triton_available = True
except ImportError:
    triton_available = False



def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def set_module_name(model, name, value):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, value)


def clear_memory(weight=None, force=False):
    if weight is not None:
        del weight

    if force:
        gc.collect()
        if torch.cuda.is_available():
            # 清理所有 CUDA 设备的缓存
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
        return

    if torch.cuda.is_available():
        need_clear = False
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_pct = (free_bytes / total_bytes * 100) if total_bytes > 0 else 0
                if free_pct < 5:
                    need_clear = True
        if need_clear:
            gc.collect()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()


def compute_memory_used_pct(device):
    memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    memory_pct = (
        memory_used
        / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
        * 100
    )
    return memory_pct


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"


def get_lowest_memory_device_index():
    device = None
    curr_device_memory_pct = 0
    for device_index in range(torch.cuda.device_count()):
        device_memory_pct = compute_memory_used_pct(device_index)
        if device is None or device_memory_pct < curr_device_memory_pct:
            device = device_index
            curr_device_memory_pct = device_memory_pct

    return device
