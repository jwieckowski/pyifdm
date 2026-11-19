# Copyright (c) 2022-2026 Jakub Więckowski & contributors
"""
Utility functions for Intuitionistic Fuzzy Sets:
normalization, score functions, distance and similarity measures.
"""

from . import normalization
from . import score
from . import distance
from . import similarity

from .normalization import (
    ecer_normalization,
    max_normalization,
    minmax_normalization,
    supriya_normalization,
    swap_normalization,
)
from .score import (
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
from .distance import (
    euclidean_distance,
    grzegorzewski_distance,
    hamming_distance,
    hausdorf_euclidean_distance,
    luo_distance,
    normalized_euclidean_distance,
    normalized_hamming_distance,
    wang_xin_distance_1,
    wang_xin_distance_2,
    yang_chiclana_distance,
)
from .similarity import (
    chen_similarity,
    fan_zhang_similarity,
    hong_kim_similarity,
    li_similarity,
    li_xu_similarity,
    ye_similarity,
)

__all__ = [
    "normalization",
    "score",
    "distance",
    "similarity",
    # normalization
    "ecer_normalization",
    "max_normalization",
    "minmax_normalization",
    "supriya_normalization",
    "swap_normalization",
    # score
    "chen_score_1",
    "chen_score_2",
    "kharal_score_1",
    "kharal_score_2",
    "liu_wang_score",
    "supriya_score",
    "thakur_score",
    "wan_dong_score_1",
    "wan_dong_score_2",
    "wei_score",
    "zhang_xu_score_1",
    "zhang_xu_score_2",
    # distance
    "euclidean_distance",
    "grzegorzewski_distance",
    "hamming_distance",
    "hausdorf_euclidean_distance",
    "luo_distance",
    "normalized_euclidean_distance",
    "normalized_hamming_distance",
    "wang_xin_distance_1",
    "wang_xin_distance_2",
    "yang_chiclana_distance",
    # similarity
    "chen_similarity",
    "fan_zhang_similarity",
    "hong_kim_similarity",
    "li_similarity",
    "li_xu_similarity",
    "ye_similarity",
]
