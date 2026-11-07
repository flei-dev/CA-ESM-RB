"""Session-level metrics identical to evaluation/rep_6/create_summary_results.py.

QoE model (SQoE-III fit), energy on Galaxy S4 and XCover Pro (measured local
playback energy + network energy + baseline energy during rebuffering), and
energy efficiency = QoE / energy * 1000.  The formulas, constants and the chunk
range used for energy (0 < chunk < 44) are copied from the original script so
new runs and the existing baseline logs are scored by exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_LEN = 44
BASE_POWER = {"galaxy": 1016.0, "xcover": 1800.0}
P_ALPHA, P_BETA = 210.0, 28.0


def qoe(total_vmaf, total_rebuf, rebuf_count, total_vmaf_change, smooth_count):
    return (0.07713539 * total_vmaf - 1.24971639 * total_rebuf - 2.87757412 * rebuf_count
            - 0.04938335 * total_vmaf_change - 1.436473 * smooth_count)


def network_energy(throughput_mbps: float, size_mb: float) -> float:
    return size_mb * (P_ALPHA / throughput_mbps + P_BETA)


class DeviceEnergy:
    """Measured per-chunk local playback energy, indexed [bitrate_kbps][chunk]."""

    def __init__(self, power_dir: Path | str, video: str, device: str):
        self.device = device
        path = Path(power_dir) / f"{device}_{video.upper()}_streaming.csv"
        self.table = pd.read_csv(path, index_col=0)
        self.table.columns = [str(c) for c in self.table.columns]

    def local(self, bitrate_kbps: int, chunk: int) -> float:
        return float(self.table[str(bitrate_kbps)].iloc[chunk])


@dataclass
class ChunkRecord:
    chunk: int            # 0-based chunk index as written in the log
    bitrate: int          # kbps
    buffer: float         # s
    rebuf: float          # s
    size: float           # bytes
    delay: float          # ms
    vmaf: float


def session_metrics(records: list[ChunkRecord], devices: dict[str, DeviceEnergy]) -> dict:
    last_vmaf = 0.0
    rebuf_count = 0
    vmaf, rebuf, smooth, vchange, data, bitrate = [], [], [], [], [], []
    energy = {name: 0.0 for name in devices}
    for r in records:
        bitrate.append(r.bitrate)
        rebuf.append(r.rebuf)
        if r.rebuf > 0 and r.chunk > 0:
            rebuf_count += 1
        data.append(r.size)
        thr = r.size / r.delay * 8.0 * 1000.0 / 1_000_000.0
        vmaf.append(r.vmaf)
        s = 0.0
        if r.chunk > 1:
            s = np.abs(r.vmaf - last_vmaf) // 20.0
            vchange.append(abs(r.vmaf - last_vmaf))
        smooth.append(s)
        last_vmaf = r.vmaf
        if 0 < r.chunk < VIDEO_LEN:
            net = network_energy(thr, r.size / 1_000_000.0)
            for name, dev in devices.items():
                energy[name] += dev.local(r.bitrate, r.chunk) + net + BASE_POWER[name] * r.rebuf
    q = qoe(np.sum(vmaf), np.sum(rebuf), rebuf_count, np.sum(vchange), np.sum(smooth))
    out = {
        "QoE": q,
        "VMAF": float(np.sum(vmaf)),
        "VMAF_mean": float(np.mean(vmaf)),
        "VMAF_p10": float(np.percentile(vmaf, 10)),
        "LowQ_ratio": float(np.mean(np.asarray(vmaf) < 60.0)),
        "Rebuffer_Time": float(np.sum(rebuf)),
        "Rebuffer_Count": rebuf_count,
        "Smoothness_Change": float(np.sum(smooth)),
        "Data": float(np.sum(data)),
        "Bitrate": float(np.sum(bitrate)),
    }
    for name, e in energy.items():
        key = {"galaxy": "G", "xcover": "X"}.get(name, name)
        out[f"Energy_{key}"] = e
        out[f"Efficiency_{key}"] = q / e * 1000.0 if e > 0 else float("nan")
    if "galaxy" in devices and "xcover" in devices:
        out["Energy_Avg"] = (out["Energy_G"] + out["Energy_X"]) / 2.0
        out["Efficiency_Avg"] = (out["Efficiency_G"] + out["Efficiency_X"]) / 2.0
    return out


def read_log(path: Path | str) -> list[ChunkRecord]:
    """Parse a rep_6-style text log (tab separated, header line, blank line at end)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) <= 1:
                break
            if p[0] == "video_chunk":
                continue
            records.append(ChunkRecord(int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3]),
                                       float(p[4]), float(p[5]), float(p[6])))
    return records


def write_log(path: Path | str, records: list[ChunkRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("video_chunk\tbitrate\tbuffer_size\trebuf\tvideo_chunk_size\tdelay\tphone_vmaf\n")
        for r in records:
            f.write(f"{r.chunk}\t{r.bitrate}\t{r.buffer}\t{r.rebuf}\t{r.size}\t{r.delay}\t{r.vmaf}\n")
        f.write("\n")
