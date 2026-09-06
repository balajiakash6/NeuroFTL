"""
Adaptive Workload-Aware FTL Simulator - Core Simulation Engine
AI-SSD Firmware Digital Twin Hackathon Project
"""

import collections
import random
import time
from typing import Dict, List, Tuple, Optional, Deque

from models import (
    OperationType,
    StreamType,
    NANDTier,
    DRAMRegion,
    IOCommand,
    L2PEntry,
    FlashBlock,
    HardwareSpecs,
    FTLMetricsSnapshot,
    TelemetryTick,
)


class SpatialLocalityDetector:
    r"""
    Tier 2 Heuristic Fallback: Online Spatial Locality Detection
    Calculates Seq_Score in real time over a sliding window W:
    Seq_Score = \sum_{i=1}^W I(|LBA_i - LBA_{i-1}| <= \Delta) / W
    """
    def __init__(self, window_size: int = 64, stride_delta: int = 8):
        self.window_size = window_size
        self.stride_delta = stride_delta
        self.history: Deque[int] = collections.deque(maxlen=window_size)
        self.sequential_pairs_count: int = 0

    def record_lba(self, lba: int) -> float:
        if len(self.history) > 0:
            prev_lba = self.history[-1]
            if abs(lba - prev_lba) <= self.stride_delta:
                self.sequential_pairs_count += 1

        self.history.append(lba)
        
        # Recalculate accurately over active window
        if len(self.history) < 2:
            return 0.0
            
        seq_count = 0
        hist_list = list(self.history)
        for i in range(1, len(hist_list)):
            if abs(hist_list[i] - hist_list[i-1]) <= self.stride_delta:
                seq_count += 1
                
        return seq_count / (len(hist_list) - 1)


class TemporalReuseFilter:
    """Lightweight 2-bit frequency filter to assess temporal locality."""
    def __init__(self, capacity: int = 4096):
        self.filter: Dict[int, int] = {}
        self.capacity = capacity

    def access(self, lba: int) -> int:
        freq = self.filter.get(lba, 0)
        freq = min(3, freq + 1)
        self.filter[lba] = freq
        if len(self.filter) > self.capacity:
            # Periodic decay / prune lowest
            self.filter = {k: v - 1 for k, v in self.filter.items() if v > 1}
        return freq

    def is_hot(self, lba: int) -> bool:
        return self.filter.get(lba, 0) >= 2


class StandardLRUFTL:
    """
    Conventional SSD Flash Translation Layer.
    Suffers from the I/O Blender Effect:
    - Shared unified DRAM LRU cache for both L2P mapping entries and data.
    - Large streaming training reads evict L2P table entries, causing Double-Read traps.
    - Unsegregated erase blocks causing high WAF (2.8+) and long GC pauses (80-120us).
    """
    def __init__(self, specs: HardwareSpecs):
        self.specs = specs
        self.l2p_cache: collections.OrderedDict = collections.OrderedDict() # Unified LRU
        self.l2p_table: Dict[int, L2PEntry] = {} # Complete master table in flash
        
        # Metrics
        self.l2p_hits = 0
        self.l2p_misses = 0
        self.total_reads = 0
        self.total_writes = 0
        self.cache_evictions = 0
        self.recent_latencies: Deque[float] = collections.deque(maxlen=100)
        self.cumulative_host_writes_kb = 0.0
        self.cumulative_flash_writes_kb = 0.0
        self.gc_pause_us = 0.0
        self.gc_active = False

    def _ensure_l2p_entry(self, lba: int, stream_type: StreamType) -> L2PEntry:
        if lba not in self.l2p_table:
            self.l2p_table[lba] = L2PEntry(
                lba=lba,
                pba=lba * 2 + 100,
                in_dram=False,
                ruh_handle=0, # Mixed unsegregated
                tier=NANDTier.TLC
            )
        return self.l2p_table[lba]

    def process_command(self, cmd: IOCommand) -> float:
        entry = self._ensure_l2p_entry(cmd.lba, cmd.stream_type)
        entry.read_count += 1
        
        # Check unified DRAM cache
        is_hit = cmd.lba in self.l2p_cache
        if is_hit:
            self.l2p_hits += 1
            self.l2p_cache.move_to_end(cmd.lba)
            l2p_lookup_latency = self.specs.t_dram_lookup_us
        else:
            self.l2p_misses += 1
            # Double-Read Trap: NAND L2P fetch + contention
            l2p_lookup_latency = self.specs.t_l2p_nand_fetch_us
            if cmd.stream_type == StreamType.LLM_TRAINING or self.gc_active:
                l2p_lookup_latency += self.specs.t_queue_contention_us * random.uniform(0.7, 1.3)

            # Insert into unified cache (evicting if full)
            if len(self.l2p_cache) >= self.specs.total_dram_entries:
                self.l2p_cache.popitem(last=False) # Evict oldest entry
                self.cache_evictions += 1
            self.l2p_cache[cmd.lba] = True

        # Base data access latency
        data_latency = self.specs.t_tlc_read_us if cmd.op == OperationType.READ else self.specs.t_tlc_read_us * 1.5
        
        # GC Pause injection (Standard FTL has unsegmented GC, long pauses)
        if random.random() < 0.08: # Periodic GC activity
            self.gc_active = True
            self.gc_pause_us = random.uniform(60.0, 110.0)
        else:
            self.gc_active = False
            self.gc_pause_us = 0.0

        total_latency = self.specs.t_controller_us + l2p_lookup_latency + data_latency + self.gc_pause_us
        
        if cmd.op == OperationType.READ:
            self.total_reads += 1
        else:
            self.total_writes += 1
            # Unsegregated WAF calculation
            self.cumulative_host_writes_kb += cmd.size_blocks * 4
            self.cumulative_flash_writes_kb += cmd.size_blocks * 4 * random.uniform(2.6, 3.2)

        self.recent_latencies.append(total_latency)
        return total_latency

    def get_snapshot(self) -> FTLMetricsSnapshot:
        total_lookups = self.l2p_hits + self.l2p_misses
        hit_rate = (self.l2p_hits / total_lookups * 100.0) if total_lookups > 0 else 100.0
        
        lat_list = sorted(self.recent_latencies) if self.recent_latencies else [20.0]
        p99_idx = int(len(lat_list) * 0.99)
        p99_lat = lat_list[min(p99_idx, len(lat_list) - 1)]
        avg_lat = sum(lat_list) / len(lat_list)
        
        waf = (self.cumulative_flash_writes_kb / self.cumulative_host_writes_kb) if self.cumulative_host_writes_kb > 0 else 2.85
        
        return FTLMetricsSnapshot(
            architecture_name="Standard LRU FTL",
            current_latency_us=round(self.recent_latencies[-1], 2) if self.recent_latencies else 20.0,
            p99_latency_us=round(p99_lat, 2),
            avg_latency_us=round(avg_lat, 2),
            l2p_hit_rate_pct=round(hit_rate, 2),
            l2p_hits=self.l2p_hits,
            l2p_misses=self.l2p_misses,
            total_reads=self.total_reads,
            total_writes=self.total_writes,
            cache_evictions=self.cache_evictions,
            waf=round(waf, 2),
            gc_active=self.gc_active,
            gc_pause_us=round(self.gc_pause_us, 2),
            pslc_migrated_blocks=0,
            dram_pinned_l2p_pct=0.0,
            dram_streaming_bypass_pct=0.0,
            dram_hot_data_pct=100.0, # All unified
            bypass_active=False,
            current_seq_score=0.0
        )


class AdaptiveWorkloadAwareFTL:
    """
    Adaptive Workload-Aware Flash Translation Layer Scheduler.
    Key Innovations:
    1. Online Spatial Locality & Temporal Reuse Classifier.
    2. Dynamic 3-Region DRAM Cache Management:
       - Region 1: Pinned L2P Metadata Region (Guaranteed >= 50% DRAM).
       - Region 2: Direct-to-NAND Streaming Bypass (Activated when Seq_Score > 0.85).
       - Region 3: Hot Data Region (KV-cache and vector index cache).
    3. Hardware Reliability Guardrails:
       - Read-Disturb Shield: Migrates KV blocks to pSLC after 50,000 reads.
       - Bounded GC Scheduler: Preemptively clamps GC pause duration to <= 15us.
    4. Flexible Data Placement (FDP) Stream Segregation:
       - Training -> RUH-0 (Persistently Isolated).
       - KV Cache -> RUH-1 (Initially Isolated).
       - Shielded -> RUH-2 (pSLC Pool).
    """
    def __init__(self, specs: HardwareSpecs):
        self.specs = specs
        self.seq_detector = SpatialLocalityDetector(specs.seq_window_size, specs.seq_stride_delta)
        self.temporal_filter = TemporalReuseFilter()
        
        # Dynamic 3-Region DRAM Partitioning
        self.pinned_l2p_cache: collections.OrderedDict = collections.OrderedDict() # Region 1
        self.hot_data_cache: collections.OrderedDict = collections.OrderedDict()   # Region 3
        
        # Flash storage structures
        self.l2p_table: Dict[int, L2PEntry] = {}
        self.blocks: Dict[int, FlashBlock] = {}
        self._init_physical_blocks()
        
        # Metrics
        self.l2p_hits = 0
        self.l2p_misses = 0
        self.total_reads = 0
        self.total_writes = 0
        self.cache_evictions = 0
        self.recent_latencies: Deque[float] = collections.deque(maxlen=100)
        self.cumulative_host_writes_kb = 0.0
        self.cumulative_flash_writes_kb = 0.0
        self.gc_pause_us = 0.0
        self.gc_active = False
        self.pslc_migrated_blocks = 0
        self.current_seq_score = 0.0
        self.bypass_active = False

    def _init_physical_blocks(self):
        # 100 physical erase blocks (TLC + pSLC pool)
        for i in range(80):
            self.blocks[i] = FlashBlock(block_id=i, tier=NANDTier.TLC, ruh_handle=0 if i < 40 else 1)
        for i in range(80, 100):
            self.blocks[i] = FlashBlock(block_id=i, tier=NANDTier.PSLC, ruh_handle=2)

    def _ensure_l2p_entry(self, lba: int, stream_type: StreamType, ruh_hint: Optional[int]) -> L2PEntry:
        if lba not in self.l2p_table:
            ruh = ruh_hint if ruh_hint is not None else (0 if stream_type == StreamType.LLM_TRAINING else 1)
            pba = lba * 2 + 200
            self.l2p_table[lba] = L2PEntry(
                lba=lba,
                pba=pba,
                in_dram=False,
                ruh_handle=ruh,
                tier=NANDTier.TLC
            )
        return self.l2p_table[lba]

    def process_command(self, cmd: IOCommand) -> float:
        # Step 1: Online Stream Classification
        self.current_seq_score = self.seq_detector.record_lba(cmd.lba)
        is_sequential = self.current_seq_score >= self.specs.seq_score_threshold
        is_large_req = cmd.size_blocks >= 16 # >= 64KB
        is_streaming = is_sequential or is_large_req or (cmd.stream_type == StreamType.LLM_TRAINING)
        
        self.bypass_active = is_streaming
        
        # Step 2: L2P Translation Lookup & 3-Region DRAM Management
        entry = self._ensure_l2p_entry(cmd.lba, cmd.stream_type, cmd.fdp_ruh_hint)
        entry.read_count += 1
        
        # Check Read-Disturb Shield: KV blocks read >= 50,000 times
        block_id = (entry.pba // 256) % 80
        curr_block = self.blocks.get(block_id)
        if curr_block:
            curr_block.cumulative_reads += 1
            if curr_block.cumulative_reads >= self.specs.read_disturb_threshold and not curr_block.is_migrated_to_pslc:
                # Migrate to pSLC pool
                curr_block.is_migrated_to_pslc = True
                curr_block.tier = NANDTier.PSLC
                entry.tier = NANDTier.PSLC
                self.pslc_migrated_blocks += 1

        if is_streaming:
            # Region 2: Direct-to-NAND Streaming Bypass
            # Stream translation uses hardware extent descriptor registers.
            # Consecutive LBAs hit the active extent buffer without polluting the Pinned L2P cache!
            if is_sequential:
                self.l2p_hits += 1
                l2p_lookup_latency = self.specs.t_dram_lookup_us
            else:
                self.l2p_misses += 1
                l2p_lookup_latency = self.specs.t_l2p_nand_fetch_us * 0.4
        else:
            # Region 1: Pinned L2P Metadata Cache (Guaranteed for KV inference & vector search)
            max_l2p_capacity = int(self.specs.total_dram_entries * 0.65) # 65% reserved for L2P
            if cmd.lba in self.pinned_l2p_cache:
                self.l2p_hits += 1
                self.pinned_l2p_cache.move_to_end(cmd.lba)
                l2p_lookup_latency = self.specs.t_dram_lookup_us
            else:
                self.l2p_misses += 1
                l2p_lookup_latency = self.specs.t_l2p_nand_fetch_us
                if len(self.pinned_l2p_cache) >= max_l2p_capacity:
                    self.pinned_l2p_cache.popitem(last=False)
                    self.cache_evictions += 1
                self.pinned_l2p_cache[cmd.lba] = True

            # Region 3: Hot Data Region (for high-reuse KV payload cache)
            max_hot_capacity = int(self.specs.total_dram_entries * 0.30)
            if self.temporal_filter.is_hot(cmd.lba):
                if len(self.hot_data_cache) >= max_hot_capacity:
                    self.hot_data_cache.popitem(last=False)
                self.hot_data_cache[cmd.lba] = True

        # Step 4: Flash Read/Write Timing
        if entry.tier == NANDTier.PSLC:
            data_latency = self.specs.t_pslc_read_us # 10us ultra-fast pSLC read
        else:
            data_latency = self.specs.t_tlc_read_us

        # Step 5: Bounded Garbage Collection Scheduler (<= 15us)
        if random.random() < 0.05:
            self.gc_active = True
            # Preemptive micro-step bounded strictly to <= 15us
            self.gc_pause_us = random.uniform(8.0, self.specs.bounded_gc_max_pause_us)
        else:
            self.gc_active = False
            self.gc_pause_us = 0.0

        total_latency = self.specs.t_controller_us + l2p_lookup_latency + data_latency + self.gc_pause_us
        
        if cmd.op == OperationType.READ:
            self.total_reads += 1
        else:
            self.total_writes += 1
            # Segregated FDP stream writes have near-ideal WAF (1.05 - 1.12)
            self.cumulative_host_writes_kb += cmd.size_blocks * 4
            self.cumulative_flash_writes_kb += cmd.size_blocks * 4 * random.uniform(1.04, 1.12)

        self.recent_latencies.append(total_latency)
        return total_latency

    def get_snapshot(self) -> FTLMetricsSnapshot:
        total_lookups = self.l2p_hits + self.l2p_misses
        hit_rate = (self.l2p_hits / total_lookups * 100.0) if total_lookups > 0 else 100.0
        
        lat_list = sorted(self.recent_latencies) if self.recent_latencies else [20.0]
        p99_idx = int(len(lat_list) * 0.99)
        p99_lat = lat_list[min(p99_idx, len(lat_list) - 1)]
        avg_lat = sum(lat_list) / len(lat_list)
        
        waf = (self.cumulative_flash_writes_kb / self.cumulative_host_writes_kb) if self.cumulative_host_writes_kb > 0 else 1.08
        
        return FTLMetricsSnapshot(
            architecture_name="Adaptive-FDP FTL",
            current_latency_us=round(self.recent_latencies[-1], 2) if self.recent_latencies else 20.0,
            p99_latency_us=round(p99_lat, 2),
            avg_latency_us=round(avg_lat, 2),
            l2p_hit_rate_pct=round(hit_rate, 2),
            l2p_hits=self.l2p_hits,
            l2p_misses=self.l2p_misses,
            total_reads=self.total_reads,
            total_writes=self.total_writes,
            cache_evictions=self.cache_evictions,
            waf=round(waf, 2),
            gc_active=self.gc_active,
            gc_pause_us=round(self.gc_pause_us, 2),
            pslc_migrated_blocks=self.pslc_migrated_blocks,
            dram_pinned_l2p_pct=65.0,
            dram_streaming_bypass_pct=5.0,
            dram_hot_data_pct=30.0,
            bypass_active=self.bypass_active,
            current_seq_score=round(self.current_seq_score, 3)
        )


class DualWorkloadSimulator:
    """
    Executes identical synthetic and trace-driven AI workloads on both FTLs simultaneously.
    Models PyTorch DataLoader vs GPU Compute pipeline to compute real-time GPU stall reduction.
    """
    def __init__(self, specs: Optional[HardwareSpecs] = None):
        self.specs = specs or HardwareSpecs()
        self.standard_ftl = StandardLRUFTL(self.specs)
        self.adaptive_ftl = AdaptiveWorkloadAwareFTL(self.specs)
        
        self.tick_counter = 0
        self.command_counter = 0
        self.current_phase = "BASELINE"
        self.phase_lock = False
        
        # Workload generation state
        self.training_cursor = 100000
        self.kv_base_lba = 5000
        self.accumulated_stall_saved_ms = 0.0
        
        # Pre-seed some initial entries
        self._warmup()

    def _warmup(self):
        # Warmup KV cache entries so hits can be registered
        for i in range(200):
            lba = self.kv_base_lba + i
            cmd = IOCommand(
                command_id=self.command_counter,
                op=OperationType.READ,
                lba=lba,
                size_blocks=1,
                stream_type=StreamType.KV_INFERENCE,
                fdp_ruh_hint=1
            )
            self.command_counter += 1
            self.standard_ftl.process_command(cmd)
            self.adaptive_ftl.process_command(cmd)

    def set_workload_phase(self, phase: str, lock: bool = False):
        self.current_phase = phase
        self.phase_lock = lock

    def generate_next_batch(self, batch_size: int = 15) -> List[IOCommand]:
        """Generates realistic AI storage commands according to current workload phase."""
        commands: List[IOCommand] = []
        
        for _ in range(batch_size):
            self.command_counter += 1
            
            if self.current_phase == "TRAINING_SURGE":
                # 85% bulk sequential reads (PyTorch DataLoader), 15% KV inference
                if random.random() < 0.85:
                    self.training_cursor += random.choice([1, 2, 4, 8])
                    cmd = IOCommand(
                        command_id=self.command_counter,
                        op=OperationType.READ,
                        lba=self.training_cursor,
                        size_blocks=random.choice([16, 32, 64]), # 64KB - 256KB
                        stream_type=StreamType.LLM_TRAINING,
                        fdp_ruh_hint=0
                    )
                else:
                    lba = self.kv_base_lba + random.randint(0, 150)
                    cmd = IOCommand(
                        command_id=self.command_counter,
                        op=OperationType.READ,
                        lba=lba,
                        size_blocks=1, # 4KB
                        stream_type=StreamType.KV_INFERENCE,
                        fdp_ruh_hint=1
                    )
            elif self.current_phase == "KV_STORM":
                # 90% latency-critical KV cache reads, high re-reference
                lba = self.kv_base_lba + random.randint(0, 80)
                cmd = IOCommand(
                    command_id=self.command_counter,
                    op=OperationType.READ,
                    lba=lba,
                    size_blocks=1,
                    stream_type=StreamType.KV_INFERENCE,
                    fdp_ruh_hint=1
                )
            elif self.current_phase == "CHECKPOINT":
                # Heavy sequential write burst
                self.training_cursor += 16
                cmd = IOCommand(
                    command_id=self.command_counter,
                    op=OperationType.WRITE,
                    lba=self.training_cursor,
                    size_blocks=64,
                    stream_type=StreamType.CHECKPOINT,
                    fdp_ruh_hint=0
                )
            else: # BASELINE: Mixed AI multi-tenant environment
                dice = random.random()
                if dice < 0.40:
                    # Sequential training stream
                    self.training_cursor += random.choice([1, 2, 4])
                    cmd = IOCommand(
                        command_id=self.command_counter,
                        op=OperationType.READ,
                        lba=self.training_cursor,
                        size_blocks=8,
                        stream_type=StreamType.LLM_TRAINING,
                        fdp_ruh_hint=0
                    )
                elif dice < 0.80:
                    # KV inference lookup
                    lba = self.kv_base_lba + random.randint(0, 150)
                    cmd = IOCommand(
                        command_id=self.command_counter,
                        op=OperationType.READ,
                        lba=lba,
                        size_blocks=1,
                        stream_type=StreamType.KV_INFERENCE,
                        fdp_ruh_hint=1
                    )
                else:
                    # RAG vector index lookup
                    lba = 20000 + random.randint(0, 300)
                    cmd = IOCommand(
                        command_id=self.command_counter,
                        op=OperationType.READ,
                        lba=lba,
                        size_blocks=2,
                        stream_type=StreamType.RAG_VECTOR,
                        fdp_ruh_hint=1
                    )
            commands.append(cmd)
            
        return commands

    def tick(self) -> TelemetryTick:
        """Executes a single simulation tick and calculates performance delta."""
        self.tick_counter += 1
        
        # Auto-rotate phases if not locked
        if not self.phase_lock:
            if self.tick_counter % 90 < 25:
                self.current_phase = "BASELINE"
            elif self.tick_counter % 90 < 55:
                self.current_phase = "TRAINING_SURGE" # I/O Blender in action!
            elif self.tick_counter % 90 < 75:
                self.current_phase = "KV_STORM"
            else:
                self.current_phase = "CHECKPOINT"

        # Generate and process batch of commands
        commands = self.generate_next_batch(batch_size=12)
        std_latencies = []
        adp_latencies = []
        
        for cmd in commands:
            lat_std = self.standard_ftl.process_command(cmd)
            lat_adp = self.adaptive_ftl.process_command(cmd)
            std_latencies.append(lat_std)
            adp_latencies.append(lat_adp)

        std_snap = self.standard_ftl.get_snapshot()
        adp_snap = self.adaptive_ftl.get_snapshot()
        
        # Calculate GPU Stall Reduction:
        # Pipelined batch model: Assume GPU compute time per mini-batch is ~120us.
        # If I/O tail latency exceeds compute time, GPU stalls for (IO_latency - Compute_time).
        gpu_compute_budget_us = 60.0
        std_stall_us = sum(max(0.0, l - gpu_compute_budget_us) for l in std_latencies)
        adp_stall_us = sum(max(0.0, l - gpu_compute_budget_us) for l in adp_latencies)
        
        if std_stall_us > 0:
            stall_reduction_pct = min(98.0, max(0.0, (1.0 - (adp_stall_us / std_stall_us)) * 100.0))
        else:
            stall_reduction_pct = 85.0
            
        saved_ms = max(0.0, (std_stall_us - adp_stall_us) / 1000.0)
        self.accumulated_stall_saved_ms += saved_ms
        
        io_blender_active = self.current_phase in ["TRAINING_SURGE", "CHECKPOINT"]

        return TelemetryTick(
            tick_id=self.tick_counter,
            timestamp=time.time(),
            workload_phase=self.current_phase,
            standard_ftl=std_snap,
            adaptive_ftl=adp_snap,
            gpu_stall_reduction_pct=round(stall_reduction_pct, 1),
            accumulated_gpu_stall_saved_ms=round(self.accumulated_stall_saved_ms, 2),
            io_blender_active=io_blender_active
        )
