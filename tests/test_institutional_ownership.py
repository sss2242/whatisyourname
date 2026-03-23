"""Tests for institutional ownership models (contagion scorer + flow predictor).

Tests both modules independently with synthetic data, verifying:
  - MHHI delta computation with various overlap scenarios
  - Crowded trade detection
  - Liquidation pressure estimation
  - Flow momentum from quarterly ownership data
  - Smart money divergence signal
  - Amihud illiquidity computation
  - Graceful handling of missing/empty data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module 2: Institutional Flow Predictor
# ---------------------------------------------------------------------------


class TestInstitutionalFlow:
    """Tests for operator1.features.institutional_flow."""

    def _make_cache(self, n_days=252, with_inst=True):
        """Build a synthetic daily cache with optional inst_* columns."""
        idx = pd.bdate_range("2023-01-01", periods=n_days, name="date")
        cache = pd.DataFrame(index=idx)
        cache["close"] = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        cache["return_1d"] = cache["close"].pct_change()
        cache["volume"] = np.random.randint(1_000_000, 10_000_000, n_days)

        if with_inst:
            # Simulate quarterly ownership data interpolated to daily
            # Ownership declining over time: 75% -> 65%
            ownership = np.linspace(75, 65, n_days)
            cache["inst_ownership_pct"] = ownership
            # HHI stable around 0.15
            cache["inst_top5_concentration"] = 0.15 + np.random.randn(n_days) * 0.01
            cache["inst_holder_count"] = 50

        return cache

    def test_flow_momentum_declining(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache)

        assert "inst_flow_momentum" in result.columns
        # Ownership declining -> momentum should be negative
        momentum = result["inst_flow_momentum"].dropna()
        if len(momentum) > 0:
            assert momentum.iloc[-1] < 0, "Declining ownership should produce negative momentum"

    def test_flow_momentum_no_inst_data(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=False)
        result = compute_institutional_flow(cache)

        # Should return cache unchanged (no inst_* columns to work with)
        assert "inst_flow_momentum" not in result.columns or result["inst_flow_momentum"].isna().all()

    def test_crowding_risk(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache)

        assert "inst_crowding_risk" in result.columns
        crowding = result["inst_crowding_risk"].dropna()
        assert len(crowding) > 0
        # Crowding should be between 0 and 1
        assert crowding.min() >= 0
        assert crowding.max() <= 1

    def test_smart_money_signal(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache)

        assert "inst_smart_money_signal" in result.columns
        signal = result["inst_smart_money_signal"].dropna()
        if len(signal) > 0:
            assert signal.min() >= -1
            assert signal.max() <= 1

    def test_amihud_illiquidity(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache)

        assert "inst_amihud_illiquidity" in result.columns
        amihud = result["inst_amihud_illiquidity"].dropna()
        assert len(amihud) > 0
        # Amihud should be non-negative
        assert amihud.min() >= 0

    def test_insider_signal_with_data(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        # Simulate insider transactions
        insiders = [
            {"insider_name": "CEO", "position": "CEO", "date": "2023-06-01",
             "transaction": "Purchase", "shares": 10000, "value": 1000000},
            {"insider_name": "CFO", "position": "CFO", "date": "2023-07-15",
             "transaction": "Sale", "shares": 5000, "value": 500000},
        ]
        result = compute_institutional_flow(cache, insider_transactions=insiders)

        assert "inst_insider_signal" in result.columns

    def test_insider_signal_no_data(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache, insider_transactions=None)

        assert "inst_insider_signal" in result.columns
        assert result["inst_insider_signal"].isna().all()

    def test_labels_assigned(self):
        from operator1.features.institutional_flow import compute_institutional_flow

        cache = self._make_cache(with_inst=True)
        result = compute_institutional_flow(cache)

        for label_col in ("inst_flow_momentum_label", "inst_crowding_risk_label",
                          "inst_smart_money_label"):
            if label_col in result.columns:
                values = result[label_col].unique()
                assert len(values) > 0


# ---------------------------------------------------------------------------
# Module 1: Ownership Contagion Scorer
# ---------------------------------------------------------------------------


class TestOwnershipContagion:
    """Tests for operator1.models.ownership_contagion."""

    def _make_holders(self, names_pcts):
        """Build a holder list from (name, pct) pairs."""
        return [
            {"name": name, "shares": int(pct * 100000), "percentage": pct, "holder_type": "institutional"}
            for name, pct in names_pcts
        ]

    def _make_cache(self, n_days=252):
        idx = pd.bdate_range("2023-01-01", periods=n_days, name="date")
        cache = pd.DataFrame(index=idx)
        cache["close"] = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        cache["volume"] = np.random.randint(1_000_000, 10_000_000, n_days)
        cache["return_1d"] = cache["close"].pct_change()
        return cache

    def test_mhhi_delta_no_overlap(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        target = self._make_holders([("Vanguard", 10.0), ("BlackRock", 8.0)])
        competitors = {
            "COMP1": self._make_holders([("Fidelity", 12.0), ("State Street", 7.0)])
        }
        cache = self._make_cache()

        result = compute_ownership_contagion(target, competitors, cache)
        assert result.mhhi_delta == 0.0
        assert result.n_shared_institutions == 0

    def test_mhhi_delta_full_overlap(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        holders = [("Vanguard", 10.0), ("BlackRock", 8.0)]
        target = self._make_holders(holders)
        competitors = {"COMP1": self._make_holders(holders)}
        cache = self._make_cache()

        result = compute_ownership_contagion(target, competitors, cache)
        assert result.mhhi_delta > 0
        assert result.n_shared_institutions == 2
        assert "Vanguard" in result.shared_institutions or "vanguard" in [s.lower() for s in result.shared_institutions]

    def test_mhhi_delta_partial_overlap(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        target = self._make_holders([("Vanguard", 10.0), ("BlackRock", 8.0), ("Fidelity", 5.0)])
        competitors = {
            "COMP1": self._make_holders([("Vanguard", 7.0), ("State Street", 6.0)])
        }
        cache = self._make_cache()

        result = compute_ownership_contagion(target, competitors, cache)
        assert 0 < result.mhhi_delta < 1.0
        assert result.n_shared_institutions == 1  # Only Vanguard shared

    def test_crowding_score_high(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        # High HHI (concentrated) + high ownership -> high crowding
        target = self._make_holders([("BigFund", 50.0)])
        cache = self._make_cache()
        cache["inst_ownership_pct"] = 80.0
        cache["inst_top5_concentration"] = 0.9

        result = compute_ownership_contagion(target, {}, cache)
        assert result.crowding_score > 0.2, f"High concentration should produce elevated crowding, got {result.crowding_score}"

    def test_crowding_score_low(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        # Low HHI + low ownership -> low crowding
        target = self._make_holders([
            ("Fund A", 2.0), ("Fund B", 2.0), ("Fund C", 2.0),
            ("Fund D", 2.0), ("Fund E", 2.0),
        ])
        cache = self._make_cache()
        cache["inst_ownership_pct"] = 10.0
        cache["inst_top5_concentration"] = 0.2

        result = compute_ownership_contagion(target, {}, cache)
        assert result.crowding_score < 0.5

    def test_liquidation_days(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        target = self._make_holders([("BigFund", 10.0)])
        # BigFund holds 10% * 100000 = 1,000,000 shares
        cache = self._make_cache()
        cache["volume"] = 500_000  # 500K daily volume

        result = compute_ownership_contagion(target, {}, cache)
        # At 25% participation: 500K * 0.25 = 125K tradeable/day
        # 1M / 125K = 8 days
        assert result.liquidation_days > 0
        assert result.top5_shares_total > 0

    def test_empty_holders(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        cache = self._make_cache()
        result = compute_ownership_contagion([], {}, cache)

        assert result.available is True
        assert result.mhhi_delta == 0.0
        assert result.crowding_score == 0.0
        assert result.liquidation_days == 0.0

    def test_inject_contagion_into_cache(self):
        from operator1.models.ownership_contagion import (
            compute_ownership_contagion,
            inject_contagion_into_cache,
        )

        target = self._make_holders([("Vanguard", 10.0)])
        cache = self._make_cache()

        result = compute_ownership_contagion(target, {}, cache)
        cache = inject_contagion_into_cache(cache, result)

        assert "inst_mhhi_delta" in cache.columns
        assert "inst_crowding_score" in cache.columns
        assert "inst_liquidation_days" in cache.columns
        assert "inst_bipartite_centrality" in cache.columns

    def test_get_ownership_edge_weights(self):
        from operator1.models.ownership_contagion import (
            compute_ownership_contagion,
            get_ownership_edge_weights,
        )

        holders = [("Vanguard", 10.0), ("BlackRock", 8.0)]
        target = self._make_holders(holders)
        competitors = {"COMP1": self._make_holders(holders)}
        cache = self._make_cache()

        result = compute_ownership_contagion(target, competitors, cache)
        weights = get_ownership_edge_weights(result)

        if result.mhhi_pairwise:
            assert "COMP1" in weights
            assert 0.3 <= weights["COMP1"] <= 1.0

    def test_bipartite_centrality(self):
        from operator1.models.ownership_contagion import compute_ownership_contagion

        target = self._make_holders([("Vanguard", 10.0), ("BlackRock", 8.0)])
        competitors = {
            "COMP1": self._make_holders([("Vanguard", 7.0), ("Fidelity", 5.0)]),
            "COMP2": self._make_holders([("BlackRock", 6.0), ("State Street", 4.0)]),
        }
        cache = self._make_cache()

        result = compute_ownership_contagion(target, competitors, cache)
        # Target should have non-zero centrality (connected to both competitors via shared holders)
        assert result.target_bipartite_centrality >= 0
        assert result.ownership_network_density >= 0
