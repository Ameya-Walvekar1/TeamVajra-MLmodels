"""Automated Verification Suite for UAV AI Context-Switching System."""

import os
import unittest
import numpy as np
import tempfile
import cv2

from config.model_registry import (
    get_model_registry,
    FLOODNET_CLASSES,
    FLOODNET_COSTS,
)
from inference.cost_map import (
    create_cost_map,
    create_planning_grid,
    colorize_cost_map,
    colorize_segmentation_mask,
)
from inference.dstar_lite import DStarLite
from inference.dstar_lite_engine import DStarLiteEngine
from inference.yolo_loader import YOLOInferenceEngine
from inference.model4_engine import Model4Engine
from inference.model4_pipeline import Model4Pipeline
from switching.model_manager import get_model_manager
from switching.model_switcher import get_model_switcher
from app.video_processor import VideoProcessor


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = get_model_registry()

    def test_all_models_registered(self):
        models = self.registry.list_all()
        model_ids = [m.model_id for m in models]
        self.assertEqual(model_ids, ["model1", "model2", "model4", "model5", "model6"])

    def test_model1_person_mapping(self):
        m1 = self.registry.get("model1")
        self.assertIsNotNone(m1)
        self.assertEqual(m1.classes.get(0), "item")
        self.assertEqual(m1.display_classes.get(0), "PERSON")

    def test_model2_fire_smoke(self):
        m2 = self.registry.get("model2")
        self.assertIsNotNone(m2)
        self.assertEqual(m2.classes, {0: "fire", 1: "smoke", 2: "other"})

    def test_model4_floodnet_specs(self):
        m4 = self.registry.get("model4")
        self.assertIsNotNone(m4)
        self.assertEqual(m4.input_size, (512, 512))
        self.assertEqual(m4.output_size, (10, 512, 512))
        self.assertEqual(len(FLOODNET_CLASSES), 10)
        self.assertEqual(FLOODNET_COSTS[4], 1)    # Road Non-Flooded
        self.assertEqual(FLOODNET_COSTS[3], 200)  # Road Flooded

    def test_model5_thermal(self):
        m5 = self.registry.get("model5")
        self.assertIsNotNone(m5)
        self.assertEqual(m5.classes.get(0), "person")


class TestInferenceAndPlanning(unittest.TestCase):
    def test_cost_map_and_grid(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[100:200, 100:200] = 4  # Road Non-Flooded (cost 1)
        mask[300:400, 300:400] = 3  # Road Flooded (cost 200)

        cost_map = create_cost_map(mask)
        self.assertEqual(cost_map.shape, (512, 512))
        self.assertEqual(cost_map[150, 150], 1)
        self.assertEqual(cost_map[350, 350], 200)

        grid = create_planning_grid(cost_map, grid_size=(64, 64))
        self.assertEqual(grid.shape, (64, 64))

        colored_cost = colorize_cost_map(cost_map)
        self.assertEqual(colored_cost.shape, (512, 512, 3))

        colored_mask = colorize_segmentation_mask(mask)
        self.assertEqual(colored_mask.shape, (512, 512, 3))

    def test_dstar_lite_dynamic_replanning(self):
        grid = np.ones((64, 64), dtype=np.float32)
        engine = DStarLiteEngine(grid, start=(4, 4), goal=(60, 60))
        path = engine.compute_initial_path()
        self.assertGreater(len(path), 0)

        # Trigger dynamic hazard
        res = engine.simulate_hazard_and_replan(radius=4)
        stats = res["stats"]
        self.assertGreater(stats["hazard_cells"], 0)
        self.assertEqual(stats["new_route_hazard_hits"], 0)
        self.assertGreaterEqual(stats["replan_iterations"], 0)
        self.assertGreater(stats["changed_route_cells"], 0)

    def test_model4_pipeline_execution(self):
        pipeline = Model4Pipeline()
        dummy_img = np.full((512, 512, 3), 120, dtype=np.uint8)
        res = pipeline.execute_pipeline(dummy_img)

        self.assertIn("mask", res)
        self.assertIn("cost_map", res)
        self.assertIn("planning_grid", res)
        self.assertIn("path", res)
        self.assertEqual(res["mask"].shape, (512, 512))
        self.assertEqual(res["planning_grid"].shape, (64, 64))
        self.assertGreater(res["path_length"], 0)

        hazard_res = pipeline.trigger_hazard_simulation(radius=4)
        self.assertIn("replan_stats", hazard_res)
        self.assertEqual(hazard_res["replan_stats"]["new_route_hazard_hits"], 0)

    def test_yolo_loader_class_remapping(self):
        registry = get_model_registry()
        m1 = registry.get("model1")
        engine = YOLOInferenceEngine(m1)
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        res = engine.run_inference(dummy_img)
        self.assertIn("detections", res)
        for det in res["detections"]:
            # Confirm Model 1 presents class as 'PERSON', never 'item'
            self.assertEqual(det["class"], "PERSON")


class TestContextSwitching(unittest.TestCase):
    def setUp(self):
        self.switcher = get_model_switcher()
        self.manager = get_model_manager()

    def test_context_transitions(self):
        success, event = self.switcher.switch_context("Fire / Smoke")
        self.assertTrue(success)
        self.assertEqual(self.switcher.active_model_id, "model2")
        self.assertEqual(event.new_context, "Fire / Smoke")

        # Switch to Model 4
        success, event = self.switcher.switch_context("Terrain / Navigation")
        self.assertTrue(success)
        self.assertEqual(self.switcher.active_model_id, "model4")

        # Switch to Model 5
        success, event = self.switcher.switch_context("Thermal Search")
        self.assertTrue(success)
        self.assertEqual(self.switcher.active_model_id, "model5")

        # Switch back to Model 1
        success, event = self.switcher.switch_context("Visible Person Search")
        self.assertTrue(success)
        self.assertEqual(self.switcher.active_model_id, "model1")

    def test_model_manager_statuses(self):
        statuses = self.manager.get_model_statuses()
        self.assertEqual(len(statuses), 5)

    def test_inference_classifier_and_switching(self):
        # Test thermal scene classification (monochrome/grayscale)
        thermal_frame = np.full((128, 128, 3), 150, dtype=np.uint8)
        ctx, conf, reason = self.switcher.classifier.classify_scene(thermal_frame)
        self.assertEqual(ctx, "Thermal Search")

        # Test fire scene classification (intense red/orange)
        fire_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        fire_frame[:, :] = (255, 60, 0)
        ctx, conf, reason = self.switcher.classifier.classify_scene(fire_frame)
        self.assertEqual(ctx, "Fire / Smoke")

        # Test dynamic inference switching
        pred_ctx, switched, event = self.switcher.infer_and_switch(thermal_frame)
        self.assertEqual(pred_ctx, "Thermal Search")
        self.assertEqual(self.switcher.active_model_id, "model5")


class TestVideoProcessor(unittest.TestCase):
    def test_video_stream_generation(self):
        # Create a synthetic 1-second video
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_in.close()
        t_out.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(t_in.name, fourcc, 10.0, (320, 240))
        for _ in range(10):
            frame = np.full((240, 320, 3), 100, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        processor = VideoProcessor()
        res = processor.process_video_stream(
            input_path=t_in.name,
            output_path=t_out.name,
            mode="DEMO_AUTO",
            switch_interval_sec=0.5,
            max_duration_sec=1.0,
        )

        self.assertGreater(res["total_frames_processed"], 0)
        self.assertGreater(len(res["timeline"]), 0)

        # Cleanup
        try:
            os.unlink(t_in.name)
            os.unlink(t_out.name)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
