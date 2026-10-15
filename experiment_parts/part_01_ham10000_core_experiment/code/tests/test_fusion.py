import unittest

import torch

from hardpair.data import MEL_IDX, NV_IDX
from hardpair.fusion import pair_restricted_fusion


def test_pair_restricted_fusion_invariants():
    torch.manual_seed(7)
    p = torch.softmax(torch.randn(32, 7), dim=1)
    s = torch.softmax(torch.randn(32, 2), dim=1)
    alpha = 0.75 * torch.rand(32, 1)
    q = pair_restricted_fusion(p, s, alpha)
    offpair = [index for index in range(7) if index not in (MEL_IDX, NV_IDX)]
    assert torch.max(torch.abs(q[:, offpair] - p[:, offpair])) < 1e-6
    assert torch.max(torch.abs(q[:, [MEL_IDX, NV_IDX]].sum(1) - p[:, [MEL_IDX, NV_IDX]].sum(1))) < 1e-6
    assert torch.max(torch.abs(q.sum(1) - 1.0)) < 1e-6
    assert torch.all(q >= 0)


def test_offpair_probability_can_be_preserved_while_winner_changes():
    p = torch.tensor([[0.05, 0.05, 0.05, 0.05, 0.20, 0.34, 0.26]])
    s = torch.tensor([[0.99, 0.01]])
    q = pair_restricted_fusion(p, s, torch.tensor([[0.75]]))
    assert torch.equal(q[:, [0, 1, 2, 3, 6]], p[:, [0, 1, 2, 3, 6]])
    assert p.argmax(1).item() == NV_IDX
    assert q.argmax(1).item() == MEL_IDX


class FusionTests(unittest.TestCase):
    def test_invariants(self):
        test_pair_restricted_fusion_invariants()

    def test_preserved_probability_does_not_imply_preserved_winner(self):
        test_offpair_probability_can_be_preserved_while_winner_changes()


if __name__ == "__main__":
    unittest.main()

