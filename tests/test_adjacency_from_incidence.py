import torch

from cpen.models.operators import apply_t1_operator
from cpen.utils.graphs import adjacency_from_batch, adjacency_from_incidence, pairwise_from_incidence


def test_adjacency_from_incidence_shared_hyperedge():
    # One hyperedge connects particles 0 and 1.
    S = torch.zeros(1, 1, 3, dtype=torch.bool)
    S[0, 0, 0] = True
    S[0, 0, 1] = True
    adj = adjacency_from_incidence(S)
    assert adj[0, 0, 1] == 1.0
    assert adj[0, 1, 0] == 1.0
    assert adj[0, 0, 2] == 0.0


def test_pairwise_from_incidence_counts_shared_membership():
    S = torch.zeros(1, 1, 3, dtype=torch.bool)
    S[0, 0, 0] = True
    S[0, 0, 1] = True
    pairwise = pairwise_from_incidence(S)
    assert pairwise[0, 0, 1] == 1.0
    assert pairwise[0, 0, 0] == 1.0


def test_gamma_adjacency_uses_pairwise_over_gamma_11():
    x = torch.ones(1, 3, 2)
    pairwise = torch.tensor([[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
    out = apply_t1_operator(
        x,
        "adjacency",
        normalization="uniform",
        energy_index=0,
        pairwise=pairwise,
        operator_normalization="gamma",
        inv_gamma_11=0.5,
    )
    assert torch.allclose(out[0, 0], torch.tensor([1.5, 1.5]))


def test_adjacency_from_batch_coo():
    batch = {
        "x": torch.zeros(2, 3, 4),
        "incidence_node": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "incidence_edge": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
        "incidence_nnz": torch.tensor([2, 2], dtype=torch.long),
    }
    adj = adjacency_from_batch(batch)
    assert adj.shape == (2, 3, 3)
    assert adj[0, 0, 1] == 1.0
    assert adj[1, 0, 2] == 1.0
