"""Tests for grouped worker splitting in fiona.axisym."""

from __future__ import annotations

import numpy as np

from fiona.axisym import _axisym_make_group_tasks


def _tasks_per_group(tasks, n_groups):
    counts = [0] * n_groups
    for group_id, local_idxs in tasks:
        counts[group_id] += 1
        assert isinstance(local_idxs, np.ndarray)
        assert local_idxs.ndim == 1
        assert len(local_idxs) > 0
    return counts


def _assert_group_partition(tasks, group_id, n_sub):
    chunks = [local_idxs for gid, local_idxs in tasks if gid == group_id]
    merged = np.concatenate(chunks) if chunks else np.array([], dtype=int)
    assert np.array_equal(np.sort(merged), np.arange(n_sub, dtype=int))


def test_axisym_worker_split_even_groups():
    """10 GL groups with 112 workers -> 11 subgroup chunks per group."""
    group_sizes = [56] * 10
    tasks, workers_per_group = _axisym_make_group_tasks(group_sizes, 112)

    assert workers_per_group == 11
    assert len(tasks) == 110
    assert _tasks_per_group(tasks, len(group_sizes)) == [11] * 10

    for gid, n_sub in enumerate(group_sizes):
        _assert_group_partition(tasks, gid, n_sub)


def test_axisym_worker_split_caps_by_group_size():
    """Small groups should never create more chunks than frequencies."""
    group_sizes = [2, 100]
    tasks, workers_per_group = _axisym_make_group_tasks(group_sizes, 112)

    assert workers_per_group == 56
    assert _tasks_per_group(tasks, len(group_sizes)) == [2, 56]

    for gid, n_sub in enumerate(group_sizes):
        _assert_group_partition(tasks, gid, n_sub)


def test_axisym_worker_split_underprovisioned_workers():
    """When workers < groups, each group still gets one chunk."""
    group_sizes = [7, 8, 9]
    tasks, workers_per_group = _axisym_make_group_tasks(group_sizes, 2)

    assert workers_per_group == 1
    assert _tasks_per_group(tasks, len(group_sizes)) == [1, 1, 1]

    for gid, n_sub in enumerate(group_sizes):
        _assert_group_partition(tasks, gid, n_sub)
