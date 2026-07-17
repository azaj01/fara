"""Model-display coordinate-space sizes shared across agents.

Fara and Qwen35 agents emit click coordinates in a normalized
model-display space rather than viewport pixels. Downstream consumers
(the inference loop's rescale, the prepro coord normalizer, the
failure-analysis viewport scaler) need to know that space's upper bound
to convert back to pixels.

Keeping this as a flat, dependency-free module under ``agents/`` (same
style as ``agents/key_mapping.py``) means any caller — including the
failure-analysis workflow — can import it without pulling in the
heavyweight agent implementations.
"""

# Fara family (FaraQwen3Agent, FaraQwen3NextAgent): emits coords in
# [0, FARA_DISPLAY_SIZE); the agent's inference loop rescales to
# viewport pixels via ``proc_coords`` before clicking.
FARA_DISPLAY_SIZE: int = 1000

# Qwen35Agent in default ``coordinate_type="relative"`` mode: emits
# coords in [0, QWEN35_DISPLAY_SIZE]. Note the off-by-one vs Fara — the
# divisor used in :func:`Qwen35Agent.adjust_coordinates` is 999, not
# 1000.
QWEN35_DISPLAY_SIZE: int = 999
