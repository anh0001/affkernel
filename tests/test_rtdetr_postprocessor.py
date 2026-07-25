import unittest

import torch
from src.zoo.rtdetr.affordance import DYN_HIDDEN, _dynamic_layer_sizes
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor


class TestRTDETRPostProcessor(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 2
        self.num_queries = 500
        self.num_classes = 10
        self.num_affordance_classes = 5
        self.image_size = (640, 640)  # Height, Width

    def create_mock_outputs(self, use_affordance=False, use_focal_loss=True, num_queries=300):
        """
        Create mock outputs for the post-processor.
        """
        logits = torch.randn(self.batch_size, num_queries, self.num_classes)
        boxes = torch.rand(self.batch_size, num_queries, 4)  # cx, cy, w, h in [0,1]

        outputs = {
            'pred_logits': logits,
            'pred_boxes': boxes
        }

        if use_affordance:
            # New dynamic-kernel representation: a shared affordance-feature
            # map + per-query kernel params (no dense [Q, C, H, W] tensor).
            reduced_dim = 8
            hf = wf = 28
            out_dim = 10  # RTDETRPostProcessor.num_affordance_classes
            w_sizes, b_sizes = _dynamic_layer_sizes(
                reduced_dim + 2, DYN_HIDDEN, out_dim
            )
            n_params = sum(w_sizes) + sum(b_sizes)
            outputs['aff_feat'] = torch.randn(
                self.batch_size, reduced_dim, hf, wf
            )
            outputs['aff_kernel'] = torch.randn(
                self.batch_size, num_queries, n_params
            )
            outputs['aff_meta'] = (reduced_dim, hf, wf)

        return outputs

    def test_initialization(self):
        """
        Test if the post-processor initializes correctly with different configurations.
        """
        # Without affordance
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=False)
        self.assertTrue(pp.use_focal_loss)
        self.assertFalse(pp.use_affordance)
        self.assertEqual(pp.num_top_queries, 300)
        self.assertEqual(pp.num_classes, self.num_classes)

        # With affordance
        pp_aff = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=False, use_affordance=True)
        self.assertFalse(pp_aff.use_focal_loss)
        self.assertTrue(pp_aff.use_affordance)
        self.assertEqual(pp_aff.num_top_queries, 300)
        self.assertEqual(pp_aff.num_classes, self.num_classes)
        self.assertIsNotNone(pp_aff.mask_resizer)

    def test_postprocessor_without_affordance_focal_loss(self):
        """
        Test post-processor without affordance and with focal loss enabled.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=False)
        outputs = self.create_mock_outputs(use_affordance=False, use_focal_loss=True)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        results = pp(outputs, orig_target_sizes)

        # Check the structure of the results
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('labels', res)
            self.assertIn('boxes', res)
            self.assertIn('scores', res)
            self.assertNotIn('affordances', res)
            self.assertEqual(res['labels'].shape[0], pp.num_top_queries)
            self.assertEqual(res['boxes'].shape, (pp.num_top_queries, 4))
            self.assertEqual(res['scores'].shape[0], pp.num_top_queries)

    def test_postprocessor_with_affordance_focal_loss(self):
        """
        Test post-processor with affordance and focal loss enabled.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=True)
        outputs = self.create_mock_outputs(use_affordance=True, use_focal_loss=True)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        # Mock the mask resizer's behavior
        pp.mask_resizer.return_value = torch.randint(0, self.num_affordance_classes, (self.batch_size, pp.num_top_queries, *self.image_size))

        results = pp(outputs, orig_target_sizes)

        # Check the structure of the results
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('labels', res)
            self.assertIn('boxes', res)
            self.assertIn('scores', res)
            self.assertIn('affordances', res)
            self.assertEqual(res['labels'].shape[0], pp.num_top_queries)
            self.assertEqual(res['boxes'].shape, (pp.num_top_queries, 4))
            self.assertEqual(res['scores'].shape[0], pp.num_top_queries)
            self.assertEqual(res['affordances'].shape, (pp.num_top_queries, self.image_size[0], self.image_size[1]))

    def test_postprocessor_without_affordance_softmax(self):
        """
        Test post-processor without affordance and with softmax (focal loss disabled).
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=False, use_affordance=False)
        outputs = self.create_mock_outputs(use_affordance=False, use_focal_loss=False)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        results = pp(outputs, orig_target_sizes)

        # Check the structure of the results
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('labels', res)
            self.assertIn('boxes', res)
            self.assertIn('scores', res)
            self.assertNotIn('affordances', res)
            self.assertEqual(res['labels'].shape[0], pp.num_top_queries)
            self.assertEqual(res['boxes'].shape, (pp.num_top_queries, 4))
            self.assertEqual(res['scores'].shape[0], pp.num_top_queries)

    def test_postprocessor_with_affordance_softmax(self):
        """
        Test post-processor with affordance and softmax (focal loss disabled).
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=False, use_affordance=True)
        outputs = self.create_mock_outputs(use_affordance=True, use_focal_loss=False)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        # Mock the mask resizer's behavior
        pp.mask_resizer.return_value = torch.randint(0, self.num_affordance_classes, (self.batch_size, pp.num_top_queries, *self.image_size))

        results = pp(outputs, orig_target_sizes)

        # Check the structure of the results
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('labels', res)
            self.assertIn('boxes', res)
            self.assertIn('scores', res)
            self.assertIn('affordances', res)
            self.assertEqual(res['labels'].shape[0], pp.num_top_queries)
            self.assertEqual(res['boxes'].shape, (pp.num_top_queries, 4))
            self.assertEqual(res['scores'].shape[0], pp.num_top_queries)
            self.assertEqual(res['affordances'].shape, (pp.num_top_queries, self.image_size[0], self.image_size[1]))

    def test_deploy_mode_without_affordance(self):
        """
        Test the post-processor in deployment mode without affordance.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=False)
        pp.deploy()
        outputs = self.create_mock_outputs(use_affordance=False, use_focal_loss=True)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        labels, boxes, scores = pp(outputs, orig_target_sizes)

        # Check the structure of the outputs
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsInstance(boxes, torch.Tensor)
        self.assertIsInstance(scores, torch.Tensor)
        self.assertEqual(labels.shape, (self.batch_size, pp.num_top_queries))
        self.assertEqual(boxes.shape, (self.batch_size, pp.num_top_queries, 4))
        self.assertEqual(scores.shape, (self.batch_size, pp.num_top_queries))

    def test_deploy_mode_with_affordance(self):
        """
        Test the post-processor in deployment mode with affordance.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=True)
        pp.deploy()
        outputs = self.create_mock_outputs(use_affordance=True, use_focal_loss=True)
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        labels, boxes, scores, affordances = pp(outputs, orig_target_sizes)

        # Check the structure of the outputs
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsInstance(boxes, torch.Tensor)
        self.assertIsInstance(scores, torch.Tensor)
        self.assertIsInstance(affordances, torch.Tensor)
        self.assertEqual(labels.shape, (self.batch_size, pp.num_top_queries))
        self.assertEqual(boxes.shape, (self.batch_size, pp.num_top_queries, 4))
        self.assertEqual(scores.shape, (self.batch_size, pp.num_top_queries))
        self.assertEqual(affordances.shape, (self.batch_size, pp.num_top_queries, self.image_size[0], self.image_size[1]))

    def test_postprocessor_empty_predictions(self):
        """
        Test post-processor behavior when there are no predictions.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=False)
        outputs = {
            'pred_logits': torch.zeros(self.batch_size, self.num_queries, self.num_classes),
            'pred_boxes': torch.zeros(self.batch_size, self.num_queries, 4)
        }
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        results = pp(outputs, orig_target_sizes)

        # Check that all scores are zero and labels are zero (assuming background)
        for res in results:
            self.assertTrue(torch.all(res['scores'] == 0))
            self.assertTrue(torch.all(res['labels'] == 0))  # Assuming label 0 is background

    def test_postprocessor_invalid_boxes(self):
        """
        Test post-processor handling of invalid bounding boxes.
        """
        pp = RTDETRPostProcessor(num_classes=self.num_classes, use_focal_loss=True, use_affordance=False)
        # Create boxes with invalid dimensions (w or h <=0)
        boxes = torch.tensor([
            [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, -0.1, 0.3]],  # Batch 1: one valid, one invalid
            [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.4, 0.4]]   # Batch 2: one invalid, one valid
        ])
        logits = torch.randn(self.batch_size, 2, self.num_classes)  # Adjusted num_queries=2
        outputs = {
            'pred_logits': logits,
            'pred_boxes': boxes
        }
        orig_target_sizes = torch.tensor([[self.image_size[0], self.image_size[1]] for _ in range(self.batch_size)])

        # Adjust num_top_queries in post-processor for this test
        pp.num_top_queries = 2

        with self.assertLogs(level='WARNING'):
            results = pp(outputs, orig_target_sizes)

        # Since our minimal RTDETRPostProcessor does not handle invalid boxes, this test is illustrative.
        # In the actual implementation, you would expect warnings or errors to be logged.
        # Here, we just ensure the code runs without crashing.
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('labels', res)
            self.assertIn('boxes', res)
            self.assertIn('scores', res)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
