"""Gói block-based copy-move (Fridrich/Popescu).

Luồng: ``dct_blocks`` (DCT-II, zigzag, lượng tử) → ``voting`` (sort +
shift voting) → ``pipeline.BlockCopyMove`` (verdict có thể SUSPICIOUS).
Chủ đề môn: DCT, JPEG, feature extraction, morphology trên mask.
"""
