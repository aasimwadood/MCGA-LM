"""Analytic computational cost (paper Sec. 4.9).

Sec. 4.9 states its own worked figures, so these tests check the module against
the paper rather than against itself.
"""

from __future__ import annotations

import pytest

from mcga_lm.eval import cost as C


def test_decoder_flops_reproduce_the_papers_worked_example() -> None:
    """Sec. 4.9: "~0.32 TFLOP for P = 8e9 and a mean L = 20 tokens"."""
    assert C.decoder_flops(8_000_000_000, 20) / 1e12 == pytest.approx(0.32, abs=1e-6)
    assert C.decoder_flops(8_000_000_000, 20) == pytest.approx(C.PAPER_DECODER_TFLOP * 1e12)


def test_decoder_flops_are_linear_in_tokens_and_parameters() -> None:
    """The ``2PL`` form: doubling either term doubles the cost."""
    base = C.decoder_flops(1_000_000_000, 10)
    assert C.decoder_flops(2_000_000_000, 10) == pytest.approx(2 * base)
    assert C.decoder_flops(1_000_000_000, 20) == pytest.approx(2 * base)


def test_perceiver_self_attention_term_is_independent_of_input_length() -> None:
    """Sec. 4.9: Perceiver cost "is independent of input length N" in the M^2 D term.

    Growing ``N`` by a factor of ten must not grow total cost by anything like a
    factor of ten -- that fixed latent bottleneck is the architectural claim.
    """
    small = C.perceiver_flops(n_inputs=47)
    large = C.perceiver_flops(n_inputs=470)
    assert large > small, "the cross-attention term does scale with N"
    assert large / small < 2.0, "but the dominant self-attention term does not"


def test_perceiver_cross_attention_term_is_linear_in_input_length() -> None:
    a = C.perceiver_flops(n_inputs=100) - C.perceiver_flops(n_inputs=0)
    b = C.perceiver_flops(n_inputs=200) - C.perceiver_flops(n_inputs=0)
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_tft_cost_is_quadratic_in_the_window() -> None:
    """Sec. 4.9: the TFT is ``O(W^2 D_c)``."""
    w32 = C.tft_flops(window=32)
    w64 = C.tft_flops(window=64)
    assert 2.0 < w64 / w32 <= 4.0, "between linear and quadratic, tending quadratic"


def test_gat_cost_is_linear_in_the_active_neighbourhood() -> None:
    """Sec. 4.9: ``O(|E_active| K D_h)`` -- over the active edges only."""
    assert C.gat_flops(n_active_edges=80) == pytest.approx(2 * C.gat_flops(n_active_edges=40))


def test_mc_dropout_is_not_a_twenty_times_multiplier() -> None:
    """Sec. 4.9: N passes over the scoring head, not N regenerations.

    The paper's point is that this keeps uncertainty at ~12% of the budget
    "instead of a 20x multiplier". The head-only cost must therefore be a tiny
    fraction of the avoided alternative.
    """
    head_only = C.mc_dropout_flops()
    naive = C.naive_mc_dropout_flops()
    assert naive / head_only > 1000.0
    assert head_only < C.decoder_flops(), "scoring 20x must cost less than generating once"


def test_peak_memory_is_near_the_papers_figure_and_inside_the_envelope() -> None:
    """Sec. 4.9: "Peak memory ... is 5.6 GB, within the Jetson AGX Orin's 32 GB"."""
    memory = C.peak_memory_bytes()
    total_gb = memory["total"] / 1e9
    assert total_gb == pytest.approx(C.PAPER_PEAK_MEMORY_GB, abs=1.0)
    assert total_gb < C.JETSON_MEMORY_GB


def test_four_bit_quantisation_is_what_fits() -> None:
    """The 4-bit configuration is load-bearing: fp16 would not fit the same way."""
    nf4 = C.peak_memory_bytes(quantisation="nf4")["total"]
    fp16 = C.peak_memory_bytes(quantisation="fp16")["total"]
    assert fp16 > 3 * nf4


def test_unknown_quantisation_is_rejected() -> None:
    with pytest.raises(ValueError):
        C.peak_memory_bytes(quantisation="fp3")


def test_cost_report_shares_sum_to_one() -> None:
    report = C.cost_report()
    assert sum(report.analytic_shares.values()) == pytest.approx(1.0)


def test_cost_report_carries_measured_shares_when_given_timings() -> None:
    measured = {
        "LLM generation": 280.0,
        "MC Dropout (N=20)": 56.0,
        "Perceiver IO": 45.0,
        "TFT": 22.0,
        "GAT retrieval": 12.0,
        "Preprocessing": 34.0,
        "Post-proc. & UI": 8.0,
    }
    report = C.cost_report(measured_ms=measured)
    # Table 6's own numbers: decode is 280/457 = 61.3% of the budget.
    assert report.measured_shares["LLM generation"] == pytest.approx(0.613, abs=0.005)
    assert report.measured_shares["MC Dropout"] == pytest.approx(0.123, abs=0.005)
    assert report.measured_shares["GAT retrieval"] == pytest.approx(0.026, abs=0.005)


def test_paper_shares_sum_to_one() -> None:
    """Sanity check on the transcribed Sec. 4.9 percentages themselves."""
    assert sum(C.PAPER_SHARES.values()) == pytest.approx(1.0, abs=0.005)


def test_table_6_components_sum_to_the_quoted_total() -> None:
    """Table 6: "34 + 45 + 22 + 12 + 280 + 56 + 8" = 457 ms, sensor window excluded."""
    assert 34 + 45 + 22 + 12 + 280 + 56 + 8 == 457
