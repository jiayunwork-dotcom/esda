"""Regression tests for self-weight handling in the by-component reading.

The bug they guard against: a site whose only retained outgoing weight is a
self-loop (so its off-diagonal cardinality is zero) was treated as an island
during conditional randomization, so every simulated value collapsed to zero
while the observed statistic was computed correctly. The same table also
exercised a single global Gi/Gi* decision, which is wrong when components
with and without self-loops are mixed.

All expectations here are derived independently of the implementation by
enumerating the legal conditional permutations of the small components.
"""

import itertools
import warnings

import numpy as np
import pytest
from libpysal.graph import Graph
from libpysal.weights import WSP
from scipy import sparse

from .. import G_Local, Moran_Local

PERMS = 97
SEED = 6421
ALT = "two-sided"


def _make(adj, kind="W"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "W":
            return WSP(sparse.csr_matrix(adj)).to_W()
        return Graph.from_sparse(sparse.csr_matrix(adj))


def _directed_selfloop_graph():
    """0 -> 1 -> 2 and 2 -> 2, all edge weights 1, one component {0, 1, 2}."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    adj[2, 2] = 1.0
    return adj


def _moran_conditional_support(adj, z, i):
    """Statistic values over every legal conditional permutation of site i."""
    n = len(z)
    n1 = n - 1
    total = (z * z).sum()
    others = [k for k in range(n) if k != i]
    values = []
    for order in itertools.permutations(others):
        zp = np.empty(n)
        zp[i] = z[i]
        zp[list(order)] = z[others]
        values.append(n1 * zp[i] * (adj[i] @ zp) / total)
    return np.array(values)


def _moran_total_support(adj, z):
    """Statistic values over every unconditional (all-position) permutation."""
    n = len(z)
    n1 = n - 1
    total = (z * z).sum()
    return np.array(
        [
            n1 * z[list(p)] * (adj @ z[list(p)]) / total
            for p in itertools.permutations(range(n))
        ]
    )


def _moran_j_conditional_support(adj, z, i):
    """Sokal-convention statistic over legal conditional permutations of i."""
    n = len(z)
    m2 = (z * z).sum() / n
    others = [k for k in range(n) if k != i]
    values = []
    for order in itertools.permutations(others):
        zp = np.empty(n)
        zp[i] = z[i]
        zp[list(order)] = z[others]
        values.append(zp[i] * (adj[i] @ zp) / m2)
    return np.array(values)


def _moran_j_total_support(adj, z):
    n = len(z)
    m2 = (z * z).sum() / n
    return np.array(
        [z[list(p)] * (adj @ z[list(p)]) / m2 for p in itertools.permutations(range(n))]
    )


def _g_conditional_support(adj_off, y, i, self_weight, star_component):
    """Gi/Gi* statistic values over legal conditional permutations of i.

    ``adj_off`` must be the *retained* adjacency after the requested star
    handling, with its diagonal removed (self-weights are passed
    separately). ``star_component`` is the component-level mode: when a
    component is Gi* (it has at least one self-loop) every one of its
    sites divides by the whole component sum, otherwise by the sum minus
    the focal value. The focal value contributes through its own
    ``self_weight`` only.
    """
    n = len(y)
    total = y.sum()
    den = total if star_component else total - y[i]
    others = [k for k in range(n) if k != i]
    values = []
    for order in itertools.permutations(others):
        zp = np.empty(n)
        zp[i] = y[i]
        zp[list(order)] = y[others]
        values.append((adj_off[i] @ zp + self_weight * y[i]) / den)
    return np.array(values)


# ----------------------------------------------------------------------
# 1. the reported reproduction: observed and simulated values are non-zero
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_reported_reproduction(kind):
    adj = _directed_selfloop_graph()
    y = np.array([1.0, 3.0, 8.0])
    w = _make(adj, kind)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = G_Local(
            y, w, transform="B", star=True, permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )
        m = Moran_Local(
            y, w, transformation="B", permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )

    # observed statistics
    np.testing.assert_allclose(g.Gs[2], 2 / 3, rtol=1e-12)
    np.testing.assert_allclose(m.Is[2], 1.2307692307692306, rtol=1e-12)

    # station 2's simulated values are the fixed, non-zero conditional value
    # (its only neighbour is itself), never the island placeholder of zero
    np.testing.assert_allclose(g.sim[:, 2], 2 / 3, rtol=1e-12)
    np.testing.assert_allclose(m.sim[:, 2], 1.2307692307692306, rtol=1e-12)
    assert (g.sim[:, 2] != 0).all()
    assert (m.sim[:, 2] != 0).all()


# ----------------------------------------------------------------------
# 2. every simulated value lies in the enumerated conditional support
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_simulations_lie_in_conditional_support(kind):
    adj = _directed_selfloop_graph()
    y = np.array([1.0, 3.0, 8.0])
    w = _make(adj, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = G_Local(
            y, w, transform="B", star=True, permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )
        m = Moran_Local(
            y, w, transformation="B", permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )
    z = m.z
    adj_off = adj.copy()
    np.fill_diagonal(adj_off, 0.0)
    for i in range(3):
        support = {round(float(v), 9) for v in _moran_conditional_support(adj, z, i)}
        drawn = {round(float(v), 9) for v in m.sim[:, i]}
        assert drawn <= support, f"Moran site {i}: {drawn - support}"
        # the resolved self-weight is per site: only site 2 carries the
        # explicit loop, so only it contributes a fixed self term. The
        # component as a whole is Gi* because it has a self-loop, so every
        # site divides by the whole component sum.
        gsupport = {
            round(float(v), 9)
            for v in _g_conditional_support(adj_off, y, i, float(adj[i, i]), True)
        }
        gdrawn = {round(float(v), 9) for v in g.sim[:, i]}
        assert gdrawn <= gsupport, f"G site {i}: {gdrawn - gsupport}"


# ----------------------------------------------------------------------
# 3. Moran analytical moments match the two nulls they belong to
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_moran_moments_match_enumerated_nulls(kind):
    adj = _directed_selfloop_graph()
    y = np.array([1.0, 3.0, 8.0])
    w = _make(adj, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Moran_Local(y, w, transformation="B", permutations=0, by_component=True)
    z = m.z
    total = _moran_j_total_support(adj, z)
    for i in range(3):
        cond = _moran_j_conditional_support(adj, z, i)
        np.testing.assert_allclose(m.EIc[i], cond.mean(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(m.VIc[i], cond.var(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            m.EI[i], total.mean(axis=0)[i], rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            m.VI[i], total.var(axis=0)[i], rtol=1e-9, atol=1e-9
        )


# ----------------------------------------------------------------------
# 4. G_Local analytical moments match the conditional null
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
@pytest.mark.parametrize("star", [None, True, False])
def test_getis_moments_use_component_weights(kind, star):
    """Analytical moments must follow the same retained weights as the
    observed statistic: identical to the whole-graph estimator run on the
    single observed component, and using that component's own denominator."""
    adj = _directed_selfloop_graph()
    y = np.array([1.0, 3.0, 8.0])
    w = _make(adj, kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = G_Local(
            y, w, transform="B", star=star, permutations=0, by_component=True
        )
        whole = G_Local(y, w, transform="B", star=star, permutations=0)
    for name in ("Gs", "EGs", "VGs", "Zs"):
        np.testing.assert_allclose(
            getattr(g, name), getattr(whole, name), rtol=1e-12, atol=1e-12,
            err_msg=name,
        )
    # and the observed statistic equals the conditional-support value that
    # keeps the focal reading and self-weight fixed
    adj_off = adj.copy()
    np.fill_diagonal(adj_off, 0.0)
    if star is False:
        self_weights = np.zeros(3)
        star_component = False
    else:
        self_weights = adj.diagonal().copy()
        star_component = bool(self_weights.any())
    for i in range(3):
        support = _g_conditional_support(
            adj_off, y, i, self_weights[i], star_component
        )
        np.testing.assert_allclose(
            g.Gs[i],
            _observed_g(adj_off, y, i, self_weights[i], star_component),
            rtol=1e-12,
        )
        # the observed statistic is one of the legal conditional draws
        assert min(support) - 1e-9 <= g.Gs[i] <= max(support) + 1e-9


def _observed_g(adj_off, y, i, self_weight, star_component):
    total = y.sum()
    den = total if star_component else total - y[i]
    return (adj_off[i] @ y + self_weight * y[i]) / den


# ----------------------------------------------------------------------
# 5. mixed components: statistics resolved per component
# ----------------------------------------------------------------------
def _mixed_components():
    adj = np.zeros((6, 6))
    # component {0,1,2}: triangle with a self-loop at 2
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        adj[i, j] = adj[j, i] = 1.0
    adj[2, 2] = 1.0
    # component {3,4,5}: triangle without a self-loop
    for i, j in [(3, 4), (4, 5), (3, 5)]:
        adj[i, j] = adj[j, i] = 1.0
    y = np.array([1.0, 2.0, 4.0, 10.0, 12.0, 14.0])
    return adj, y


@pytest.mark.parametrize("kind", ["W", "Graph"])
@pytest.mark.parametrize("star", [None, True, False, 0.5])
def test_mixed_components_resolve_independently(kind, star):
    adj, y = _mixed_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = G_Local(
            y, _make(adj, kind), transform="B", star=star, permutations=0,
            by_component=True,
        )
        a = G_Local(
            y[:3], _make(adj[:3, :3], kind), transform="B", star=star,
            permutations=0, by_component=True,
        )
        b = G_Local(
            y[3:], _make(adj[3:, 3:], kind), transform="B", star=star,
            permutations=0, by_component=True,
        )
    for name in ("Gs", "EGs", "VGs", "Zs"):
        np.testing.assert_array_equal(
            np.r_[getattr(full, name)[:3], getattr(full, name)[3:]],
            np.r_[getattr(a, name), getattr(b, name)],
            err_msg=name,
        )


@pytest.mark.parametrize("kind", ["W", "Graph"])
@pytest.mark.parametrize("transform", ["B", "r"])
def test_mixed_components_moran_extraction(kind, transform):
    adj, y = _mixed_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = Moran_Local(
            y, _make(adj, kind), transformation=transform, permutations=0,
            by_component=True,
        )
        a = Moran_Local(
            y[:3], _make(adj[:3, :3], kind), transformation=transform,
            permutations=0, by_component=True,
        )
        b = Moran_Local(
            y[3:], _make(adj[3:, 3:], kind), transformation=transform,
            permutations=0, by_component=True,
        )
    for name in ("Is", "EI", "VI", "EIc", "VIc", "z"):
        np.testing.assert_allclose(
            np.r_[getattr(full, name)[:3], getattr(full, name)[3:]],
            np.r_[getattr(a, name), getattr(b, name)],
            rtol=1e-10, atol=1e-10, err_msg=name,
        )


# ----------------------------------------------------------------------
# 6. omitted star in component mode behaves like None
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_omitted_star_equals_none(kind):
    adj, y = _mixed_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        omitted = G_Local(
            y, _make(adj, kind), transform="B", permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )
        explicit = G_Local(
            y, _make(adj, kind), transform="B", star=None, permutations=PERMS,
            seed=SEED, alternative=ALT, by_component=True,
        )
    np.testing.assert_array_equal(omitted.Gs, explicit.Gs)
    np.testing.assert_array_equal(omitted.sim, explicit.sim)
    np.testing.assert_array_equal(omitted.p_sim, explicit.p_sim)
    assert omitted.star is True


@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_whole_graph_omitted_star_default_unchanged(kind):
    """Whole-graph calls keep star=False when omitted."""
    adj, y = _mixed_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        omitted = G_Local(y, _make(adj, kind), transform="B", permutations=0)
        explicit = G_Local(
            y, _make(adj, kind), transform="B", star=False, permutations=0
        )
    np.testing.assert_array_equal(omitted.Gs, explicit.Gs)
    assert not omitted.star


@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_array_star_follows_rows(kind):
    """An array-valued ``star`` is indexed like ``y``; reordering the table
    and the array together leaves the analysis unchanged."""
    adj = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2)]:
        adj[i, j] = adj[j, i] = 1.0
    for i, j in [(3, 4), (4, 5)]:
        adj[i, j] = adj[j, i] = 1.0
    y = np.array([1.0, 2.0, 4.0, 10.0, 12.0, 14.0])
    star = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    order = np.array([3, 4, 5, 0, 1, 2])
    y2 = y[order]
    adj2 = adj[np.ix_(order, order)]
    star2 = star[order]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = G_Local(
            y, _make(adj, kind), transform="R", star=star, permutations=0,
            by_component=True,
        )
        moved = G_Local(
            y2, _make(adj2, kind), transform="R", star=star2, permutations=0,
            by_component=True,
        )
    np.testing.assert_allclose(moved.Gs, base.Gs[order], rtol=1e-12)


# ----------------------------------------------------------------------
# 7. single one-way in-edge site with a self-loop stays in its component
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_incoming_only_site_with_selfloop(kind):
    adj = np.zeros((4, 4))
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    adj[1, 2] = 1.0  # 2 has only an incoming edge plus its self-loop
    adj[2, 2] = 1.0
    y = np.array([1.0, 3.0, 8.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = G_Local(
            y, _make(adj, kind), transform="B", star=True, permutations=PERMS,
            seed=SEED, alternative=ALT, by_component=True,
        )
        m = Moran_Local(
            y, _make(adj, kind), transformation="B", permutations=PERMS,
            seed=SEED, alternative=ALT, by_component=True,
        )
    # the isolated site 3 is NaN; sites 0..2 are all finite including 2
    assert np.isnan(g.Gs[3]) and np.isnan(m.Is[3])
    assert np.isfinite(g.Gs[:3]).all() and np.isfinite(m.Is[:3]).all()
    assert np.isfinite(g.sim[:, :3]).all()
    # site 2's simulated values are not the zero island placeholder
    assert (g.sim[:, 2] != 0).any()
    assert (m.sim[:, 2] != 0).any()


# ----------------------------------------------------------------------
# 8. extraction / sandwich / n_jobs / keep_simulations invariance with loops
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
@pytest.mark.parametrize("cls,kw,attr", [
    (Moran_Local, {"transformation": "B"}, "Is"),
    (G_Local, {"transform": "B", "star": None}, "Gs"),
])
def test_loop_extraction_and_invariance(kind, cls, kw, attr):
    adj, y = _mixed_components()
    order = np.array([3, 4, 5, 0, 1, 2])
    y2 = y[order]
    adj2 = adj[np.ix_(order, order)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = cls(
            y, _make(adj, kind), permutations=PERMS, seed=SEED, alternative=ALT,
            by_component=True, **kw,
        )
        moved = cls(
            y2, _make(adj2, kind), permutations=PERMS, seed=SEED, alternative=ALT,
            by_component=True, **kw,
        )
        parallel = cls(
            y, _make(adj, kind), permutations=PERMS, seed=SEED, alternative=ALT,
            by_component=True, n_jobs=2, **kw,
        )
        dropped = cls(
            y, _make(adj, kind), permutations=PERMS, seed=SEED, alternative=ALT,
            by_component=True, keep_simulations=False, **kw,
        )
        alone_a = cls(
            y[:3], _make(adj[:3, :3], kind), permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True, **kw,
        )
    # extraction
    np.testing.assert_array_equal(getattr(base, attr)[:3], getattr(alone_a, attr))
    np.testing.assert_array_equal(base.sim[:, :3], alone_a.sim)
    np.testing.assert_array_equal(base.p_sim[:3], alone_a.p_sim)
    # row permutation
    np.testing.assert_array_equal(getattr(moved, attr), getattr(base, attr)[order])
    np.testing.assert_array_equal(moved.sim, base.sim[:, order])
    np.testing.assert_array_equal(moved.p_sim, base.p_sim[order])
    # parallel
    np.testing.assert_array_equal(parallel.sim, base.sim)
    np.testing.assert_array_equal(parallel.p_sim, base.p_sim)
    # no saved simulations still gives the same p_sim
    np.testing.assert_allclose(dropped.p_sim, base.p_sim, equal_nan=True)


# ----------------------------------------------------------------------
# 9. self-loops never join components, but still reach the statistic
# ----------------------------------------------------------------------
def test_selfloop_does_not_join_components():
    adj = np.zeros((5, 5))
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        adj[i, j] = adj[j, i] = 1.0
    adj[0, 0] = 1.0
    adj[4, 4] = 1.0
    y = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Moran_Local(
            y, _make(adj), transformation="B", permutations=PERMS, seed=SEED,
            alternative=ALT, by_component=True,
        )
    assert np.isnan(m.Is[4])
    assert np.isfinite(m.Is[:4]).all()
    alone = Moran_Local(
        y[:4], _make(adj[:4, :4]), transformation="B", permutations=PERMS,
        seed=SEED, alternative=ALT, by_component=True,
    )
    np.testing.assert_array_equal(m.Is[:4], alone.Is)
    np.testing.assert_array_equal(m.sim[:, :4], alone.sim)


# ----------------------------------------------------------------------
# 10. inputs are never modified when self-loops are present
# ----------------------------------------------------------------------
def test_immutability_with_selfloops():
    adj, y = _mixed_components()
    w = _make(adj)
    before = w.sparse.toarray().copy()
    transform_before = w.transform
    y_before = y.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G_Local(
            y, w, transform="B", star=None, permutations=9, seed=1,
            by_component=True,
        )
        Moran_Local(
            y, w, transformation="B", permutations=9, seed=1, by_component=True
        )
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(w.sparse.toarray(), before)
    assert w.transform == transform_before


# ----------------------------------------------------------------------
# 11. a degenerate reference distribution keeps the undefined z_sim
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["W", "Graph"])
def test_degenerate_reference_keeps_undefined_zsim(kind):
    """A site whose focal value and self-weight exhaust its component has a
    constant simulated distribution; its standardised value and standard
    normal p-value stay undefined instead of being filled with a spurious
    finite number. The permutation p-value itself is still defined."""
    adj = _directed_selfloop_graph()
    y = np.array([1.0, 3.0, 8.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Moran_Local(
            y, _make(adj, kind), transformation="B", permutations=PERMS,
            seed=SEED, alternative=ALT, by_component=True,
        )
        g = G_Local(
            y, _make(adj, kind), transform="B", star=True, permutations=PERMS,
            seed=SEED, alternative=ALT, by_component=True,
        )
    # the simulated column is genuinely constant
    assert np.ptp(m.sim[:, 2]) == 0
    assert np.ptp(g.sim[:, 2]) == 0
    assert np.isnan(m.z_sim[2]) and np.isnan(m.p_z_sim[2])
    assert np.isnan(g.z_sim[2]) and np.isnan(g.p_z_sim[2])
    # the permutation p-value is still a valid pseudo p-value
    assert np.isfinite(m.p_sim[2]) and np.isfinite(g.p_sim[2])
    # non-degenerate sites keep finite standardisation
    assert np.isfinite(m.z_sim[:2]).all()
    assert np.isfinite(g.z_sim[:2]).all()
