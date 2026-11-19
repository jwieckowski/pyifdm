# Copyright (c) 2022-2026 Jakub Więckowski & contributors
"""Unit tests for pyifdm.utils.score - reference values from upstream / literature."""

from __future__ import annotations

import numpy as np
import pytest

from pyifdm.utils.score import (
    chen_score_1,
    chen_score_2,
    kharal_score_1,
    kharal_score_2,
    liu_wang_score,
    supriya_score,
    thakur_score,
    wan_dong_score_1,
    wan_dong_score_2,
    wei_score,
    zhang_xu_score_1,
    zhang_xu_score_2,
)


def test_liu_wang_score():
    x = np.array([0.6, 0.3])
    assert np.round(liu_wang_score(x), 2) == 0.66


def test_wei_score():
    x = np.array([0.6, 0.3])
    assert np.round(wei_score(x), 3) == 0.910


def test_wan_dong_score_1():
    x = np.array([0.6, 0.3])
    assert np.round(wan_dong_score_1(x), 3) == 0.575


def test_wan_dong_score_2():
    x = np.array([0.5422, 0.3416])
    assert np.round(wan_dong_score_2(x), 4) == 0.6003


def test_zhang_xu_score_1():
    x = np.array([0.7209, 0.1551])
    assert np.round(zhang_xu_score_1(x), 4) == 0.7517


def test_chen_score_1():
    x = np.array([0.17125, 0.21033])
    assert np.round(chen_score_1(x), 3) == -0.039


def test_supriya_score():
    x = np.array([0.17125, 0.21033])
    assert np.round(supriya_score(x), 3) == 0.041


def test_kharal_score_1():
    x = np.array([0.17125, 0.21033])
    assert np.round(kharal_score_1(x), 3) == -0.243


def test_kharal_score_2():
    x = np.array([0.17125, 0.21033])
    assert np.round(kharal_score_2(x), 3) == -0.428


def test_chen_score_2():
    x = np.array([0.17125, 0.21033])
    assert np.round(chen_score_2(x), 3) == 0.480


def test_thakur_score():
    x = np.array([0.19028, 0.82374])
    assert np.round(thakur_score(x), 4) == -0.6423


def test_zhang_xu_score_2():
    x = np.array([0.609, 0.301, 0.09])
    assert np.round(zhang_xu_score_2(x), 3) == 0.570


def test_chen_score_1_formula():
    # u - v
    x = np.array([0.7, 0.2])
    assert chen_score_1(x) == pytest.approx(0.5)
