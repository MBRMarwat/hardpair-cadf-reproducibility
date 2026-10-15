from __future__ import annotations

import unittest

import torch
from PIL import Image

from cadfnet_v3.data import ColorDecomposition, HeuristicLesionCropper, color_statistics
from cadfnet_v3.fusion import SoftFeasibilityFusion, build_gate_inputs
from cadfnet_v3.losses import ErrorFocusedSpecialistLoss, build_feasibility_targets, correction_rates
from cadfnet_v3.model import CADFNet, FeasibilityPredictor, MLPGate


class CADFComponentsTest(unittest.TestCase):
    def test_color_decomposition_and_statistics_shapes(self) -> None:
        rgb = torch.rand(3, 32, 32).clamp_min(1e-3)
        melanin, hemoglobin = ColorDecomposition()(rgb)
        self.assertEqual(tuple(melanin.shape), (1, 32, 32))
        self.assertEqual(tuple(hemoglobin.shape), (1, 32, 32))
        self.assertEqual(tuple(color_statistics(rgb, melanin, hemoglobin).shape), (12,))

    def test_cropper_returns_valid_image(self) -> None:
        image = Image.new("RGB", (80, 60), (210, 180, 170))
        crop = HeuristicLesionCropper()(image)
        self.assertGreater(crop.width, 0)
        self.assertGreater(crop.height, 0)

    def test_shared_encoder_accepts_both_five_channel_views(self) -> None:
        model = CADFNet(pretrained=False)
        generalist, specialist = model.forward_views(
            torch.randn(2, 5, 64, 64), torch.randn(2, 5, 64, 64)
        )
        self.assertEqual(tuple(generalist.shape), (2, 7))
        self.assertEqual(tuple(specialist.shape), (2, 2))

    def test_feasibility_targets_mean_transfer_helps(self) -> None:
        labels = torch.tensor([4, 4, 2, 5, 4])
        predictions = torch.tensor([2, 5, 2, 0, 6])
        targets = build_feasibility_targets(labels, predictions)
        expected = torch.zeros(5, 5)
        expected[0, 2] = 1.0
        expected[4, 4] = 1.0
        self.assertTrue(torch.equal(targets, expected))

    def test_error_focused_loss_backpropagates(self) -> None:
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        generalist = torch.softmax(torch.randn(4, 7), dim=1)
        loss = ErrorFocusedSpecialistLoss()(logits, targets, generalist)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_gate_and_fusion_are_normalized_and_differentiable(self) -> None:
        logits = torch.randn(6, 7)
        probabilities = torch.softmax(logits, dim=1)
        specialist = torch.softmax(torch.randn(6, 2), dim=1)
        feasibility_model = FeasibilityPredictor()
        feasibility = feasibility_model(torch.randn(6, 12))
        gate = MLPGate(dropout=0.0)
        gate_inputs = build_gate_inputs(probabilities, specialist, feasibility)
        alpha, beta = gate(gate_inputs)
        fused = SoftFeasibilityFusion()(probabilities, specialist, alpha, beta, feasibility)
        self.assertEqual(tuple(gate_inputs.shape), (6, 8))
        self.assertTrue(torch.allclose(fused.sum(dim=1), torch.ones(6), atol=1e-6))
        self.assertTrue((fused >= 0).all())
        loss = -fused[:, 4].clamp_min(1e-8).log().mean()
        loss.backward()
        gate_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in gate.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gate_gradient, 0.0)

    def test_primary_fusion_preserves_pair_mass_when_beta_zero(self) -> None:
        probabilities = torch.softmax(torch.randn(8, 7), dim=1)
        specialist = torch.softmax(torch.randn(8, 2), dim=1)
        feasibility = torch.rand(8, 5)
        alpha = torch.rand(8) * 0.75
        beta = torch.zeros(8)
        fused = SoftFeasibilityFusion()(probabilities, specialist, alpha, beta, feasibility)
        original_mass = probabilities[:, 4] + probabilities[:, 5]
        fused_mass = fused[:, 4] + fused[:, 5]
        self.assertTrue(torch.allclose(original_mass, fused_mass, atol=1e-6))
        self.assertTrue(torch.allclose(probabilities[:, [0, 1, 2, 3, 6]], fused[:, [0, 1, 2, 3, 6]], atol=1e-6))

    def test_correction_rate_uses_explicit_label_mapping(self) -> None:
        labels = torch.tensor([4, 5, 4])
        generalist = torch.tensor(
            [[0.0, 0, 0, 0, 1, 2, 0], [0.0, 0, 0, 0, 2, 1, 0], [3.0, 0, 0, 0, 1, 0, 0]]
        )
        specialist = torch.tensor([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
        rates = correction_rates(specialist, generalist, labels)
        self.assertAlmostEqual(rates["pair_conditional_correction_rate"], 1.0)
        self.assertAlmostEqual(rates["full_error_correction_potential"], 1.0)


if __name__ == "__main__":
    unittest.main()
