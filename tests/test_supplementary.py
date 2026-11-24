import numpy as np

from mo433.supplementary import mcnemar, spikes


def test_spikes_finds_a_rise_after_warmup_and_its_recovery():
    losses = [1.0, 0.5, 0.4, 0.8, 0.6, 0.39, 0.3]
    assert spikes(losses, warmup_epochs=2) == [{"epoch": 4, "rise": 1.0, "recovered_in": 2}]
    assert spikes(losses, warmup_epochs=4) == []       # the rise is inside the warm-up


def test_mcnemar_counts_discordant_pairs():
    y = np.array([0, 1, 2, 3])
    r = mcnemar(np.array([0, 1, 0, 0]), np.array([0, 0, 2, 0]), y)
    assert (r["only_c"], r["only_d"], r["p"]) == (1, 1, 1.0)
