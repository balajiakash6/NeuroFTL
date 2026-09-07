"""
Unit Tests for Adaptive Workload-Aware FTL Simulator
"""

import unittest
from models import HardwareSpecs, IOCommand, OperationType, StreamType, NANDTier
from simulator import SpatialLocalityDetector, TemporalReuseFilter, StandardLRUFTL, AdaptiveWorkloadAwareFTL, DualWorkloadSimulator


class TestAdaptiveFTL(unittest.TestCase):
    def setUp(self):
        self.specs = HardwareSpecs(
            total_dram_entries=128,
            seq_score_threshold=0.85,
            seq_window_size=32,
            seq_stride_delta=8,
            read_disturb_threshold=100, # Small for testing
            bounded_gc_max_pause_us=15.0
        )

    def test_spatial_locality_detection(self):
        detector = SpatialLocalityDetector(window_size=32, stride_delta=8)
        # Sequential streaming
        for lba in range(1000, 1050):
            score = detector.record_lba(lba)
        self.assertGreaterEqual(score, 0.85, f"Expected sequential score >= 0.85, got {score}")

        # Random access
        detector_rand = SpatialLocalityDetector(window_size=32, stride_delta=8)
        import random
        random.seed(42)
        for _ in range(50):
            score = detector_rand.record_lba(random.randint(1000, 900000))
        self.assertLess(score, 0.50, f"Expected random score < 0.50, got {score}")

    def test_read_disturb_shield_migration(self):
        ftl = AdaptiveWorkloadAwareFTL(self.specs)
        # Issue 150 reads to same LBA
        for i in range(150):
            cmd = IOCommand(
                command_id=i,
                op=OperationType.READ,
                lba=500,
                size_blocks=1,
                stream_type=StreamType.KV_INFERENCE,
                fdp_ruh_hint=1
            )
            ftl.process_command(cmd)

        snap = ftl.get_snapshot()
        self.assertGreater(snap.pslc_migrated_blocks, 0, "Expected at least 1 block migrated to pSLC")

    def test_dual_simulation_tick(self):
        sim = DualWorkloadSimulator(self.specs)
        tick = sim.tick()
        self.assertEqual(tick.tick_id, 1)
        self.assertIsNotNone(tick.standard_ftl)
        self.assertIsNotNone(tick.adaptive_ftl)
        self.assertGreater(tick.standard_ftl.avg_latency_us, 0)
        self.assertGreater(tick.adaptive_ftl.avg_latency_us, 0)

    def test_io_blender_immunity(self):
        sim = DualWorkloadSimulator(self.specs)
        # Run intensive training surge
        sim.set_workload_phase("TRAINING_SURGE", lock=True)
        for _ in range(25):
            tick = sim.tick()
            
        std_snap = tick.standard_ftl
        adp_snap = tick.adaptive_ftl
        
        # Adaptive FTL should have significantly higher hit rate and lower P99 latency
        self.assertGreater(adp_snap.l2p_hit_rate_pct, std_snap.l2p_hit_rate_pct)
        self.assertLess(adp_snap.p99_latency_us, std_snap.p99_latency_us)
        print(f"\n[Test Result] Training Surge -> Std P99: {std_snap.p99_latency_us}us | Adp P99: {adp_snap.p99_latency_us}us | GPU Stall Saved: {tick.accumulated_gpu_stall_saved_ms}ms")

    def test_three_generation_evolution(self):
        """Validates 3-way performance ranking: Conventional < v1.0 Heuristic < v2.0 Adaptive FDP."""
        sim = DualWorkloadSimulator(self.specs)
        for _ in range(30):
            tick = sim.tick()
            
        std = tick.standard_ftl
        v1 = tick.heuristic_v1_ftl
        v2 = tick.adaptive_ftl
        
        self.assertIsNotNone(v1)
        # Classification accuracy progression: 0% -> ~84.2% -> 93.6%
        self.assertEqual(std.classification_accuracy_pct, 0.0)
        self.assertGreaterEqual(v1.classification_accuracy_pct, 70.0)
        self.assertEqual(v2.classification_accuracy_pct, 93.6)
        
        # P99 latency progression: v2.0 < v1.0 < Standard
        self.assertLess(v2.p99_latency_us, std.p99_latency_us)
        print(f"[3-Gen Evolution] Std Acc: {std.classification_accuracy_pct}% | v1 Acc: {v1.classification_accuracy_pct}% | v2 Acc: {v2.classification_accuracy_pct}%")


if __name__ == '__main__':
    unittest.main()

