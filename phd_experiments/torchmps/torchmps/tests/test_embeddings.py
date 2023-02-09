# MIT License
#
# Copyright (c) 2021 Jacob Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Tests for embedding functions"""
from functools import partial

import torch
from hypothesis import given, strategies as st

from torchmps import ProbMPS
from torchmps.embeddings import DataDomain, FixedEmbedding, onehot_embed, init_mlp_embed


@given(st.booleans(), st.floats(-100, 100), st.floats(0.0001, 1000))
def test_data_domain(continuous, max_val, length):
    """
    Verify that DataDomain class initializes properly
    """
    min_val = max_val - length
    if not continuous:
        max_val = abs(int(max_val)) + 1
    data_domain = DataDomain(continuous=continuous, max_val=max_val, min_val=min_val)

    assert data_domain.continuous == continuous
    assert data_domain.max_val == max_val
    if continuous:
        assert data_domain.min_val == min_val
    else:
        assert not hasattr(data_domain, "min_val")


@given(st.integers(1, 100))
def test_onehot_embedding(emb_dim):
    """
    Verify that FixedEmbedding works as expected when given one-hot embedding
    """
    data_domain = DataDomain(continuous=False, max_val=emb_dim)
    fixed_embed = FixedEmbedding(partial(onehot_embed, emb_dim=emb_dim), data_domain)
    assert torch.allclose(fixed_embed.lamb_mat, torch.ones(()))

    # Verify that the embedding function can be called and works fine
    rand_inds1 = torch.randint(emb_dim, (10,))
    rand_inds2 = torch.randint(emb_dim, (10, 5))
    rand_vecs1 = fixed_embed(rand_inds1)
    rand_vecs2 = fixed_embed(rand_inds2).reshape(50, emb_dim)
    rand_inds2 = rand_inds2.reshape(50)
    for vec, idx in zip(rand_vecs1, rand_inds1):
        assert torch.all((vec == 1) == (torch.arange(emb_dim) == idx))
    for vec, idx in zip(rand_vecs2, rand_inds2):
        assert torch.all((vec == 1) == (torch.arange(emb_dim) == idx))


@given(st.integers(1, 10), st.integers(1, 3))
def test_mlp_embedding(input_dim, num_layers):
    """
    Verify that MLP embedding function runs and gives properly normalized probs
    """
    bond_dim = 10
    hidden_dim = 10
    embed_fun = init_mlp_embed(input_dim, num_layers=num_layers, hidden_dims=hidden_dim)
    mps = ProbMPS(
        seq_len=1,
        input_dim=input_dim,
        bond_dim=bond_dim,
        complex_params=False,
        embed_fun=embed_fun,
    )
    points = torch.linspace(0, 1, 1000)[:, None]  # Add phony spatial dim
    log_prob_densities = mps(points)
    assert log_prob_densities.shape == (1000,)
    prob_densities = torch.exp(log_prob_densities)
    total_prob = torch.trapz(prob_densities, points[:, 0])
    assert torch.allclose(total_prob, torch.ones(()))
