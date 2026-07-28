"""
FP8 Cache Compression for KV-Cache Optimization

Provides:
- FP8 quantization/dequantization with calibration
- Compressed KV-cache storage and retrieval
- Compression accuracy validation
"""

import torch
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any


@dataclass
class ScaleParams:
    """Calibrated scale parameters for a (layer, head) pair."""
    calibrated: bool = False
    scale_k: float = 1.0
    scale_v: float = 1.0


class FP8Compressor:
    """
    FP8 compression with per-layer/head calibration.

    Simulates FP8 quantization using int8 storage with calibrated scales.
    """

    def __init__(self, calibration_samples: int = 100,
                 device: torch.device = torch.device("cpu")):
        self.calibration_samples = calibration_samples
        self.device = device

        # Scale parameters keyed by (layer_id, head_id)
        self.scale_params: Dict[Tuple[int, int], ScaleParams] = {}

        # Calibration sample buffers
        self._calibration_k: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._calibration_v: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._total_samples = 0

    def collect_calibration_sample(self, layer_id: int, head_id: int,
                                   k: torch.Tensor, v: torch.Tensor) -> None:
        """Collect a calibration sample for scale estimation."""
        key = (layer_id, head_id)
        if key not in self._calibration_k:
            self._calibration_k[key] = []
            self._calibration_v[key] = []

        self._calibration_k[key].append(k.detach().float())
        self._calibration_v[key].append(v.detach().float())
        self._total_samples += 1

    def calibrate_scales(self) -> None:
        """Compute quantization scales from collected samples."""
        for key in self._calibration_k:
            k_samples = self._calibration_k[key]
            v_samples = self._calibration_v[key]

            # Compute max absolute values for scale
            k_abs_max = max(s.abs().max().item() for s in k_samples) if k_samples else 1.0
            v_abs_max = max(s.abs().max().item() for s in v_samples) if v_samples else 1.0

            # Scale to map to int8 range [-127, 127]
            scale_k = k_abs_max / 127.0 if k_abs_max > 0 else 1.0
            scale_v = v_abs_max / 127.0 if v_abs_max > 0 else 1.0

            self.scale_params[key] = ScaleParams(
                calibrated=True,
                scale_k=scale_k,
                scale_v=scale_v,
            )

    def quantize_kv(self, layer_id: int, head_id: int,
                    k: torch.Tensor, v: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV tensors to simulated FP8 (int8 with scale)."""
        key = (layer_id, head_id)
        params = self.scale_params.get(key)

        if params is None or not params.calibrated:
            # Auto-calibrate with current data
            self.collect_calibration_sample(layer_id, head_id, k, v)
            self.calibrate_scales()
            params = self.scale_params[key]

        k_float = k.float()
        v_float = v.float()

        # Quantize to int8 range
        k_fp8 = torch.clamp(torch.round(k_float / params.scale_k), -127, 127).to(torch.int8)
        v_fp8 = torch.clamp(torch.round(v_float / params.scale_v), -127, 127).to(torch.int8)

        return k_fp8, v_fp8

    def dequantize_kv(self, layer_id: int, head_id: int,
                      k_fp8: torch.Tensor, v_fp8: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize FP8 tensors back to float16."""
        key = (layer_id, head_id)
        params = self.scale_params[key]

        k_restored = (k_fp8.float() * params.scale_k).to(torch.float16)
        v_restored = (v_fp8.float() * params.scale_v).to(torch.float16)

        return k_restored, v_restored

    def get_compression_stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        calibrated = sum(1 for p in self.scale_params.values() if p.calibrated)
        return {
            "calibrated_layers": calibrated,
            "total_layers": len(self.scale_params),
            "calibration_samples": self._total_samples,
            "memory_reduction_percent": 50.0,  # FP8 (1 byte) vs FP16 (2 bytes)
        }


class CompressedKVCache:
    """
    Compressed KV-cache using FP8 quantization.

    Stores KV pairs in compressed form and decompresses on retrieval.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int, device: torch.device,
                 enable_compression: bool = True):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.enable_compression = enable_compression

        # Compressor
        self.compressor = FP8Compressor(device=device) if enable_compression else None

        # Compressed storage: dict of dicts, values are tensors or None
        self.compressed_k: Dict[int, Dict[int, Optional[torch.Tensor]]] = {}
        self.compressed_v: Dict[int, Dict[int, Optional[torch.Tensor]]] = {}
        for layer_id in range(num_layers):
            self.compressed_k[layer_id] = {}
            self.compressed_v[layer_id] = {}
            for head_id in range(num_heads):
                self.compressed_k[layer_id][head_id] = None
                self.compressed_v[layer_id][head_id] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        # Track uncompressed size for stats
        self._uncompressed_bytes = 0
        self._compressed_bytes = 0

    def calibrate_compression(self, samples: List[Tuple[int, int, torch.Tensor, torch.Tensor]]) -> None:
        """Calibrate the compressor with sample data."""
        if self.compressor is None:
            return

        for layer_id, head_id, k, v in samples:
            self.compressor.collect_calibration_sample(layer_id, head_id, k, v)

        self.compressor.calibrate_scales()

    def store_compressed(self, layer_id: int, head_id: int,
                         seq_pos: int, k: torch.Tensor,
                         v: torch.Tensor) -> None:
        """Store KV pair, optionally compressing it."""
        with self._lock:
            k_2d = k.float()
            v_2d = v.float()

            # Ensure 3D: (batch, seq, head_dim)
            if k_2d.dim() == 2:
                k_3d = k_2d.unsqueeze(1)  # (batch, 1, head_dim)
                v_3d = v_2d.unsqueeze(1)
            else:
                k_3d = k_2d
                v_3d = v_2d

            # Track uncompressed size
            elem_size = 2  # FP16
            self._uncompressed_bytes += k_3d.nelement() * elem_size + v_3d.nelement() * elem_size

            if self.enable_compression and self.compressor is not None:
                k_q, v_q = self.compressor.quantize_kv(layer_id, head_id, k_3d, v_3d)
            else:
                k_q = k_3d
                v_q = v_3d

            # Track compressed size
            c_elem_size = k_q.element_size()
            self._compressed_bytes += k_q.nelement() * c_elem_size + v_q.nelement() * c_elem_size

            # Append to existing storage or create new
            existing_k = self.compressed_k[layer_id][head_id]
            existing_v = self.compressed_v[layer_id][head_id]

            if existing_k is None:
                # Initialize storage: allocate max_seq_len slots
                storage_k = torch.zeros(1, self.max_seq_len, self.head_dim, dtype=k_q.dtype, device=self.device)
                storage_v = torch.zeros(1, self.max_seq_len, self.head_dim, dtype=v_q.dtype, device=self.device)
                storage_k[:, seq_pos:seq_pos + k_q.shape[1], :] = k_q
                storage_v[:, seq_pos:seq_pos + v_q.shape[1], :] = v_q
                self.compressed_k[layer_id][head_id] = storage_k
                self.compressed_v[layer_id][head_id] = storage_v
            else:
                existing_k[:, seq_pos:seq_pos + k_q.shape[1], :] = k_q
                existing_v[:, seq_pos:seq_pos + v_q.shape[1], :] = v_q

    def retrieve_decompressed(self, layer_id: int, head_id: int,
                              start: int, end: int
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and decompress KV for a range of positions."""
        with self._lock:
            stored_k = self.compressed_k[layer_id][head_id]
            stored_v = self.compressed_v[layer_id][head_id]

            if stored_k is None:
                # Return zeros if nothing stored
                seq_len = end - start
                return (
                    torch.zeros(1, seq_len, self.head_dim, dtype=torch.float16, device=self.device),
                    torch.zeros(1, seq_len, self.head_dim, dtype=torch.float16, device=self.device),
                )

            k_slice = stored_k[:, start:end, :]
            v_slice = stored_v[:, start:end, :]

            if self.enable_compression and self.compressor is not None:
                k_out, v_out = self.compressor.dequantize_kv(layer_id, head_id, k_slice, v_slice)
            else:
                k_out = k_slice.to(torch.float16)
                v_out = v_slice.to(torch.float16)

            return k_out, v_out

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage statistics."""
        compressed_bytes = 0
        for layer_id in range(self.num_layers):
            for head_id in range(self.num_heads):
                t = self.compressed_k[layer_id][head_id]
                if t is not None:
                    compressed_bytes += t.nelement() * t.element_size()
                t = self.compressed_v[layer_id][head_id]
                if t is not None:
                    compressed_bytes += t.nelement() * t.element_size()

        compressed_mb = compressed_bytes / (1024 * 1024)
        # Uncompressed equivalent (FP16 = 2 bytes per element vs int8 = 1 byte)
        uncompressed_mb = compressed_mb * 2 if self.enable_compression else compressed_mb
        compression_ratio = 2.0 if (self.enable_compression and compressed_mb > 0) else 1.0
        savings = (1.0 - 1.0 / compression_ratio) * 100 if compression_ratio > 0 else 0.0

        return {
            "compressed_mb": compressed_mb,
            "uncompressed_mb": uncompressed_mb,
            "compression_ratio": compression_ratio,
            "memory_savings_percent": savings,
        }

    def clear_cache(self) -> None:
        """Clear all compressed cache data."""
        with self._lock:
            for layer_id in range(self.num_layers):
                for head_id in range(self.num_heads):
                    self.compressed_k[layer_id][head_id] = None
                    self.compressed_v[layer_id][head_id] = None
            self._compressed_bytes = 0
            self._uncompressed_bytes = 0


class CompressionAccuracyValidator:
    """Validates compression accuracy by comparing original vs compressed tensors."""

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self._losses: List[float] = []

    def measure_accuracy_loss(self, orig_k: torch.Tensor, orig_v: torch.Tensor,
                              comp_k: torch.Tensor, comp_v: torch.Tensor) -> float:
        """Measure accuracy loss between original and compressed KV pairs."""
        k_mse = torch.mean((orig_k.float() - comp_k.float()) ** 2).item()
        v_mse = torch.mean((orig_v.float() - comp_v.float()) ** 2).item()
        loss = (k_mse + v_mse) / 2.0

        self._losses.append(loss)
        return loss

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Return accuracy statistics from all measurements."""
        if not self._losses:
            return {
                "samples": 0,
                "avg_loss": 0.0,
                "max_loss": 0.0,
                "relative_loss_percent": 0.0,
            }

        avg_loss = sum(self._losses) / len(self._losses)
        max_loss = max(self._losses)

        return {
            "samples": len(self._losses),
            "avg_loss": avg_loss,
            "max_loss": max_loss,
            "relative_loss_percent": avg_loss * 100,
        }
