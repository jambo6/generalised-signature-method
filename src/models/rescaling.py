"""
rescaling.py
=================================
Contains the important pre- and post- rescaling methods.
"""
import signatory
import torch
import math


def _compute_length(path):
    differences = path[:, 1:] - path[:, :-1]
    # We use the L^\infty norm on reals^d
    out = differences.abs().max(dim=-1).values
    return out.sum(dim=-1)


def _rescale_path(path, depth, by_length=False):
    # Can approximate this pretty well with Stirling's formula if need be... but we don't need to :P
    coeff = math.factorial(depth) ** (1 / depth)

    if by_length:
        length = _compute_length(path)
        coeff = coeff / length
    return coeff * path


def rescale_path(path, depth):
    """Rescales the input path by depth! ** (1 / depth), so that the last signature term should be roughly O(1)."""
    return _rescale_path(path, depth, False)


def rescale_path_by_length(path, depth):
    """Rescales the input path by depth! ** (1 / depth) / Length(path), so that the last signature term should be
    roughly O(1). (In practice this one seems to do a much worse job of rescaling the path then without the length, so
    it's probably not worth using this one.)
    """
    return _rescale_path(path, depth, True)


def _rescale_signature(signature, channels, depth, length, by_length=False):
    sigtensor_channels = signature.size(-1)
    if signatory.signature_channels(channels, depth) != sigtensor_channels:
        raise ValueError("Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
                         "not consistent.".format(sigtensor_channels, channels, depth))

    if by_length:
        length_reciprocal = length.reciprocal()
        for i in range(len(signature.shape) - len(length.shape)):
            length_reciprocal.unsqueeze_(-1)

    end = 0
    term_length = 1
    val = 1
    terms = []
    for d in range(1, depth + 1):
        start = end
        term_length *= channels
        end = start + term_length

        val *= d
        if by_length:
            val *= length_reciprocal

        terms.append(signature[..., start:end] * val)

    return torch.cat(terms, dim=-1)


def rescale_signature(signature, channels, depth):
    """Rescales the output signature by multiplying the depth-d term by d!, so that every term should be about O(1)."""
    return _rescale_signature(signature, channels, depth, None, False)


def rescale_signature_by_length(signature, path, depth):
    """Rescales the output signature by multiplying the depth-d term by d! * Length(path) ** -d, so that every term
    should be about O(1). In practice this seems to do a worse job normalizing things than without the length."""
    return _rescale_signature(signature, path.size(-1), depth, _compute_length(path), True)
