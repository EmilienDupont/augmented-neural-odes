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

"""Tests for MPS base functions"""
from functools import partial

import torch
import pytest
from hypothesis import given, settings, strategies as st

from torchmps import ProbMPS, ProbUnifMPS
from torchmps.embeddings import DataDomain, onehot_embed
from .utils_for_tests import complete_binary_dataset, allcloseish

# Frequently used Hypothesis strategies
bool_st = st.booleans
seq_len_st = partial(st.integers, 1, 1000)
bond_dim_st = partial(st.integers, 1, 20)
input_dim_st = partial(st.integers, 2, 10)
model_list = ["fixed-len", "uniform"]


# Parameterization over fixed-len and uniform models
def parametrize_models():
    return pytest.mark.parametrize("model", model_list, ids=model_list)


def init_model_and_data(
    model,
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    use_bias,
    vec_input,
    big_batch,
    normal_init=False,
    embed_fun=None,
    continuous=None,
):
    """Initialize probabilistic MPS and the sequence data it will be fed"""
    # Account for case when we want an embedding
    if embed_fun is not None:
        if continuous:
            domain = DataDomain(continuous=True, max_val=1, min_val=0)
        else:
            domain = DataDomain(continuous=False, max_val=input_dim)
    else:
        domain = None

    # Initialization method
    init_method = "normal" if normal_init else "near_eye"

    if model == "fixed-len":
        prob_mps = ProbMPS(
            seq_len,
            input_dim,
            bond_dim,
            complex_params=complex_params,
            use_bias=use_bias,
            init_method=init_method,
            embed_fun=embed_fun,
            domain=domain,
        )
    elif model == "uniform":
        prob_mps = ProbUnifMPS(
            input_dim,
            bond_dim,
            complex_params=complex_params,
            use_bias=use_bias,
            init_method=init_method,
            embed_fun=embed_fun,
            domain=domain,
        )

    batch_dim = 25 if big_batch else 1
    if vec_input:
        fake_data = torch.randn(batch_dim, seq_len, input_dim).abs()
        fake_data /= fake_data.sum(dim=2, keepdim=True)
    else:
        fake_data = torch.randint(input_dim, (batch_dim, seq_len))

    return prob_mps, fake_data


def default_model_data(
    model, complex_params=True, normal_init=True, embed_fun=None, continuous=None
):
    """Default values for model and data"""
    return init_model_and_data(
        model,
        seq_len=14,
        input_dim=4,
        bond_dim=5,
        complex_params=True,
        use_bias=False,
        vec_input=False,
        big_batch=True,
        normal_init=True,
        embed_fun=embed_fun,
        continuous=continuous,
    )


def rescale_embed(input, dim=4):
    """Dummy discrete embedding function that should do nothing"""
    return 3.14 * onehot_embed(input, dim)


@parametrize_models()
@settings(deadline=None)
@given(
    seq_len_st(),
    input_dim_st(),
    bond_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_model_forward(
    model,
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
    normal_init,
):
    """
    Verify that model forward function runs and gives reasonable output
    """
    # Initialize probabilistic MPS and dataset
    prob_mps, fake_data = init_model_and_data(
        model,
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        use_bias,
        vec_input,
        big_batch,
        normal_init=normal_init,
    )
    batch_dim = 25 if big_batch else 1

    # Call the model on the fake data, verify that it looks alright
    log_probs = prob_mps(fake_data, slim_eval=False, parallel_eval=parallel_eval)
    assert log_probs.shape == (batch_dim,)
    assert torch.all(log_probs.isfinite())
    assert log_probs.is_floating_point()
    if not torch.all(log_probs <= 0):
        # For sequences with only one value, probability should be 1
        assert input_dim == 1
        assert torch.allclose(log_probs, 0.0)

    # Check that the old method of evaluation gives identical results
    # Note that the model doesn't support bias matrices with slim_eval
    if not use_bias:
        old_log_probs = prob_mps(fake_data, slim_eval=True)
        assert allcloseish(log_probs, old_log_probs)


@parametrize_models()
@settings(deadline=None, max_examples=1000)
@given(
    input_dim_st(),
    bond_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_valid_binary_probs(
    model,
    seq_len,
    bond_dim,
    complex_params,
    slim_eval,
    parallel_eval,
    use_bias,
    use_embed,
    normal_init,
):
    """
    Verify that for binary distributions, all probabilities sum up to 1
    """
    # Note: model doesn't support bias matrices with slim_eval
    if use_bias and slim_eval:
        return

    # Initialize dataset and model
    all_seqs = complete_binary_dataset(seq_len)
    prob_mps, _ = init_model_and_data(
        model,
        seq_len,
        2,
        bond_dim,
        complex_params,
        use_bias,
        False,
        False,
        normal_init=normal_init,
        embed_fun=(partial(rescale_embed, dim=2) if use_embed else None),
        continuous=False,
    )

    # Get model probabilities and verify they are close to 1 (in log space)
    log_probs = prob_mps(all_seqs, slim_eval=slim_eval, parallel_eval=parallel_eval)
    assert allcloseish(log_probs.logsumexp(dim=0), 0.0)


@parametrize_models()
@settings(deadline=None)
@given(
    seq_len_st(),
    input_dim_st(),
    bond_dim_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
    bool_st(),
)
def test_model_backward(
    model,
    seq_len,
    input_dim,
    bond_dim,
    complex_params,
    slim_eval,
    parallel_eval,
    use_bias,
    vec_input,
    big_batch,
):
    """
    Verify that model backward pass runs and updates model params
    """
    # Note: model doesn't support bias matrices with slim_eval
    if use_bias and slim_eval:
        return

    # Initialize probabilistic MPS, dataset, and optimizer
    prob_mps, fake_data = init_model_and_data(
        model,
        seq_len,
        input_dim,
        bond_dim,
        complex_params,
        use_bias,
        vec_input,
        big_batch,
    )
    optimizer = torch.optim.Adam(prob_mps.parameters())
    old_params = tuple(p.detach().clone() for p in prob_mps.parameters())

    # Call the model on fake data, backpropagate, and take gradient step
    loss = prob_mps.loss(fake_data, slim_eval=slim_eval, parallel_eval=parallel_eval)
    loss.backward()
    optimizer.step()

    # Verify that the new parameters are different from the old ones
    param_diff = False
    new_params = tuple(prob_mps.parameters())
    assert len(old_params) == len(new_params)
    assert all(p.grad is not None for p in new_params)
    for old_p, new_p in zip(old_params, new_params):
        assert old_p.shape == new_p.shape
        assert torch.all(old_p.isfinite())
        assert torch.all(new_p.isfinite())

        # For input dimension of 1, the gradients should be trivial
        if input_dim != 1:
            param_diff = param_diff or not torch.all(old_p == new_p)
        else:
            assert allcloseish(old_p, new_p, tol=1e-3)
            param_diff = True

    assert param_diff


@parametrize_models()
@given(st.booleans(), st.booleans())
def test_trivial_embedding(model, complex_params, normal_init):
    """
    Ensure an embedding that rescales by a constant gives same probabilities
    """
    # Define two models with same parameters, but using embedding for latter
    normal_mps, fake_data = default_model_data(
        model, complex_params=complex_params, normal_init=normal_init
    )
    embed_mps, _ = default_model_data(
        model,
        complex_params=complex_params,
        normal_init=normal_init,
        embed_fun=rescale_embed,
        continuous=False,
    )
    embed_mps.load_state_dict(normal_mps.state_dict())

    normal_output = normal_mps(fake_data)
    embed_output = embed_mps(fake_data)

    assert torch.allclose(normal_output, embed_output)
