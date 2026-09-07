"""
Adaptive Workload-Aware FTL Simulator - Models and Data Structures
AI-SSD Firmware Digital Twin Hackathon Project
Authors: Balaji Akash S, Karthick P, Padmanabhan S, Benit D Binu
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import time


class OperationType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class StreamType(str, Enum):
    LLM_TRAINING = "LLM_TRAINING"      # Sequential bulk streaming, low temporal locality
    KV_INFERENCE = "KV_INFERENCE"      # Random, latency-critical, high temporal locality
    RAG_VECTOR = "RAG_VECTOR"          # Clustered lookups, graph/embedding index traversal
    CHECKPOINT = "CHECKPOINT"          # Large sequential write bursts


class NANDTier(str, Enum):
    TLC = "TLC"                        # Standard triple-level cell (higher density, read disturb risk)
    PSLC = "PSLC"                      # Pseudo-SLC shield pool (3x faster, 10x endurance, immune to read disturb)


class DRAMRegion(str, Enum):
    PINNED_L2P = "PINNED_L2P"          # Region 1: Guaranteed L2P translation entries
    STREAMING_BYPASS = "STREAMING_BYPASS" # Region 2: Direct-to-NAND DMA buffers (0% cache eviction)
    HOT_DATA = "HOT_DATA"              # Region 3: Cached inference KV blocks & vector indices


@dataclass
class IOCommand:
    """Represents an incoming NVMe host command."""
    command_id: int
    op: OperationType
    lba: int
    size_blocks: int                   # in 4KB blocks
    stream_type: StreamType
    timestamp_us: float = 0.0
    fdp_ruh_hint: Optional[int] = None # NVMe FDP hint (0 = training, 1 = KV), optional


@dataclass
class L2PEntry:
    """Represents a Logical-to-Physical translation entry."""
    lba: int
    pba: int
    in_dram: bool = False
    last_accessed_us: float = 0.0
    read_count: int = 0
    tier: NANDTier = NANDTier.TLC
    ruh_handle: int = 0                # 0 = Bulk/Training, 1 = KV/Inference, 2 = pSLC pool


@dataclass
class FlashBlock:
    """Represents a physical NAND flash erase block."""
    block_id: int
    tier: NANDTier
    ruh_handle: int
    total_pages: int = 256
    valid_pages: int = 0
    erase_count: int = 0
    cumulative_reads: int = 0
    is_migrated_to_pslc: bool = False


@dataclass
class HardwareSpecs:
    """Hardware timing specs for modern enterprise NVMe SSD."""
    # Latencies in microseconds (us)
    t_controller_us: float = 0.1
    t_dram_lookup_us: float = 0.1
    t_tlc_read_us: float = 25.0
    t_pslc_read_us: float = 10.0
    t_l2p_nand_fetch_us: float = 35.0  # Time to fetch L2P page from NAND on miss
    t_queue_contention_us: float = 90.0 # Contention penalty under I/O Blender
    
    # Cache and buffer limits
    total_dram_entries: int = 2048     # DRAM slots (each holds an L2P or data entry)
    min_pinned_l2p_fraction: float = 0.50
    bypass_buffer_fraction: float = 0.05
    
    # Thresholds
    seq_score_threshold: float = 0.85
    seq_window_size: int = 64
    seq_stride_delta: int = 8          # Blocks (32KB)
    read_disturb_threshold: int = 50000
    bounded_gc_max_pause_us: float = 15.0


@dataclass
class FTLMetricsSnapshot:
    """Telemetry snapshot for an individual FTL architecture."""
    architecture_name: str
    current_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    avg_latency_us: float = 0.0
    l2p_hit_rate_pct: float = 100.0
    l2p_hits: int = 0
    l2p_misses: int = 0
    total_reads: int = 0
    total_writes: int = 0
    cache_evictions: int = 0
    waf: float = 1.0
    gc_active: bool = False
    gc_pause_us: float = 0.0
    pslc_migrated_blocks: int = 0
    dram_pinned_l2p_pct: float = 50.0
    dram_streaming_bypass_pct: float = 5.0
    dram_hot_data_pct: float = 45.0
    bypass_active: bool = False
    current_seq_score: float = 0.0
    classification_accuracy_pct: float = 93.6


@dataclass
class TelemetryTick:
    """Synchronized dual/tri telemetry tick emitted to dashboard."""
    tick_id: int
    timestamp: float
    workload_phase: str
    standard_ftl: FTLMetricsSnapshot
    adaptive_ftl: FTLMetricsSnapshot
    gpu_stall_reduction_pct: float
    accumulated_gpu_stall_saved_ms: float
    io_blender_active: bool
    heuristic_v1_ftl: Optional[FTLMetricsSnapshot] = None
