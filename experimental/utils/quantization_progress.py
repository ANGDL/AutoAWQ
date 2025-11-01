import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _sanitize_layer_name(layer_name: str) -> str:
    sanitized = layer_name.replace("/", "_")
    sanitized = sanitized.replace(" ", "_")
    return sanitized.replace(".", "_")


class QuantizationProgressManager:
    """Persist per-layer quantization artifacts and progress information.

    The manager stores scale and clip tensors for each layer on disk so the
    quantization process can recover gracefully after interruptions. All writes
    are scheduled on a background thread to avoid blocking the main quantization
    loop.
    """

    def __init__(
        self,
        save_dir: Path,
        logger: Optional[Any] = None,
        max_workers: int = 1,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir = self.save_dir / "layers"
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.save_dir / "config.json"
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: List[Any] = []
        self._lock = threading.Lock()
        self._config_cache: Optional[Dict[str, Any]] = None
        self._extend_info: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def ensure_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the initial configuration if missing and merge defaults."""
        with self._lock:
            current = self._load_config_locked()
            if current is None:
                new_config = dict(config)
                new_config.setdefault("completed_layers", [])
                self._write_config_locked(new_config)
                self._config_cache = new_config
                return new_config

            updated = dict(current)
            for key, value in config.items():
                if key not in updated:
                    updated[key] = value
                elif key not in {"completed_layers"} and updated[key] != value:
                    self._log_warning(
                        "Config mismatch for %s: existing=%s, requested=%s",
                        key,
                        updated[key],
                        value,
                    )
            if "completed_layers" not in updated:
                updated["completed_layers"] = []
            if updated != current:
                self._write_config_locked(updated)
            self._config_cache = updated
            return updated

    def mark_layer_done(self, layer_name: str) -> None:
        """Record the layer as processed in the configuration file."""
        with self._lock:
            config = self._load_config_locked() or {}
            completed: List[str] = list(config.get("completed_layers", []))
            if layer_name not in completed:
                completed.append(layer_name)
                config["completed_layers"] = completed
                self._write_config_locked(config)
            self._config_cache = config

    def get_completed_layers(self) -> Sequence[str]:
        with self._lock:
            if self._config_cache is None:
                self._config_cache = self._load_config_locked()
            if not self._config_cache:
                return []
            return tuple(self._config_cache.get("completed_layers", []))

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            if self._config_cache is None:
                self._config_cache = self._load_config_locked() or {}
            return dict(self._config_cache)

    # ------------------------------------------------------------------
    # Layer state helpers
    # ------------------------------------------------------------------
    def layer_state_exists(self, layer_name: str, layer_index: int) -> bool:
        return self._layer_state_path(layer_name, layer_index).exists()

    def load_layer_state(self, layer_name: str, layer_index: int) -> Optional[Dict[str, Any]]:
        path = self._layer_state_path(layer_name, layer_index)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_warning(
                "Failed to load layer state %s: %s",
                path,
                exc,
            )
            return None

    def save_layer_async(
        self,
        layer_name: str,
        layer_index: int,
        scales_list: Iterable[Tuple[str, Sequence[str], torch.Tensor]],
        clip_list: Iterable[Tuple[str, torch.Tensor]],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            extend_info = dict()
            keys = tuple(self._extend_info.keys())
            for key in keys:
                if key.startswith(layer_name):
                    extend_info[key] = self._extend_info.pop(key)

        payload = {
            "layer_name": layer_name,
            "layer_index": layer_index,
            "scales_list": self._serialize_scales(scales_list),
            "clip_list": self._serialize_clips(clip_list),
            "metadata": extra_metadata or {},
        }

        if extend_info:
            payload.update(extend_info)

        path = self._layer_state_path(layer_name, layer_index)
        future = self._executor.submit(self._write_layer_state, path, payload)
        with self._lock:
            self._pending.append(future)

    def push_current_extend_info(
        self,
        layer_name: str,
        extend_info: Dict[str, Any],
        ) -> None:
        with self._lock:
            existing = self._extend_info.setdefault(layer_name, {})
            overlapping = set(existing).intersection(extend_info)
            if overlapping:
                raise ValueError(
                    "Extend info keys already exist for layer "
                    f"{layer_name}: {sorted(overlapping)}"
                )
            existing.update(extend_info)


    def flush(self) -> None:
        self.wait_for_pending_tasks()

    def wait_for_pending_tasks(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    break
                future = self._pending.pop(0)
            future.result()

    def close(self) -> None:
        self.wait_for_pending_tasks()
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _layer_state_path(self, layer_name: str, layer_index: int) -> Path:
        safe_name = _sanitize_layer_name(layer_name)
        filename = f"{layer_index:04d}_{safe_name}.pt"
        return self.layers_dir / filename

    def _serialize_scales(
        self,
        scales_list: Iterable[Tuple[str, Sequence[str], torch.Tensor]],
    ) -> List[Tuple[str, Tuple[str, ...], torch.Tensor]]:
        serialized: List[Tuple[str, Tuple[str, ...], torch.Tensor]] = []
        for prev_op, layer_names, scales in scales_list or []:
            tensor = scales.detach().to("cpu")
            serialized.append((prev_op, tuple(layer_names), tensor))
        return serialized

    def _serialize_clips(
        self,
        clip_list: Iterable[Tuple[str, torch.Tensor]],
    ) -> List[Tuple[str, torch.Tensor]]:
        serialized: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in clip_list or []:
            serialized.append((name, tensor.detach().to("cpu")))
        return serialized

    def _write_layer_state(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _load_config_locked(self) -> Optional[Dict[str, Any]]:
        if not self.config_path.exists():
            return None
        with self.config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_config_locked(self, config: Dict[str, Any]) -> None:
        tmp_path = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=True)
        tmp_path.replace(self.config_path)

    def _log_warning(self, message: str, *args: Any) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, "warning", None)
        if log_fn is not None:
            log_fn(message, *args)

