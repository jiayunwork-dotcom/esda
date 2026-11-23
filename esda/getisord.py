"""
Getis and Ord G statistic for spatial autocorrelation
"""

__author__ = "Sergio J. Rey <srey@asu.edu>, Myunghwa Hwang <mhwang4@gmail.com> "
__all__ = ["G", "G_Local"]

import warnings

import numpy as np
from libpysal.weights import W
from libpysal.weights.spatial_lag import lag_spatial
from libpysal.weights.util import fill_diagonal
from scipy import stats

from .crand import (
    _prepare_univariate,
    crand_components,
    degenerate_reference_columns,
    mask_unobserved,
    rebuild_sparse_weights,
    transform_observed,
)
from .crand import crand as _crand_plus
from .crand import njit as _njit

PERMUTATIONS = 999


class _StarUnset:
    """Sentinel marking a ``star`` argument the caller did not pass.

    ``G_Local`` documents ``star=False`` as its whole-graph default, but in
    the component reading an omitted ``star`` means "resolve Gi/Gi* from
    each component's retained diagonal" (the same as an explicit ``None``).
    Keeping the sentinel private lets both behaviours coexist without
    changing the whole-graph default.
    """

    def __repr__(self):  # pragma: no cover - cosmetic
        return "None"


_STAR_UNSET = _StarUnset()


class G:
    """
    Global G Autocorrelation Statistic

    Parameters
    ----------
    y             : array (n,1)
                    Attribute values
    w             : W | Graph
                   spatial weights instance as W or Graph aligned with y
    permutations  : int
                    the number of random permutations for calculating pseudo p_values

    Attributes
    ----------
    y : array
        original variable
    w : W
        spatial weights instance as W or Graph aligned with y
    permutation : int
        the number of permutations
    G : float
        the value of statistic
    EG : float
        the expected value of statistic
    VG : float
        the variance of G under normality assumption
    z_norm : float
        standard normal test statistic
    p_norm : float
        p-value under normality assumption (one-sided)
    sim : array
        (if permutations > 0)
        vector of G values for permutated samples
    p_sim : float
        p-value based on permutations (one-sided)
        null: spatial randomness
        alternative: the observed G is extreme it is either
        extremely high or extremely low
    EG_sim : float
        average value of G from permutations
    VG_sim : float
        variance of G from permutations
    seG_sim : float
        standard deviation of G under permutations.
    z_sim : float
        standardized G based on permutations
    p_z_sim : float
        p-value based on standard normal approximation from permutations (one-sided)

    Notes
    -----
    Moments are based on normality assumption.

    For technical details see :cite:`Getis_2010` and :cite:`Ord_2010`.


    Examples
    --------
    >>> import libpysal
    >>> import numpy
    >>> numpy.random.seed(10)

    Preparing a point data set

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points

    >>> w = libpysal.weights.DistanceBand(points, threshold=15)
    >>> w.transform = "B"

    Preparing a variable

    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord G test

    >>> from esda import G
    >>> g = G(y, w)

    Examining the results

    >>> round(g.G, 3)
    np.float64(0.557)

    >>> round(g.p_norm, 3)
    np.float64(0.173)
    """

    def __init__(self, y, w, permutations=PERMUTATIONS):
        y = np.asarray(y).flatten()
        self.n = len(y)
        self.y = y
        if isinstance(w, W):
            w.transform = "B"
        else:
            w = w.transform("B")
            self.summary = w.summary()
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.y2 = y * y
        y = y.reshape(
            len(y), 1
        )  # Ensure that y is an n by 1 vector, otherwise y*y.T == y*y
        self.den_sum = (y * y.T).sum() - (y * y).sum()
        self.G = self.__calc(self.y)
        self.z_norm = (self.G - self.EG) / np.sqrt(self.VG)
        self.p_norm = 1.0 - stats.norm.cdf(np.abs(self.z_norm))

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.y)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.G
            larger = sum(above)
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EG_sim = sum(sim) / permutations
            self.seG_sim = sim.std()
            self.VG_sim = self.seG_sim**2
            self.z_sim = (self.G - self.EG_sim) / self.seG_sim
            self.p_z_sim = 1.0 - stats.norm.cdf(np.abs(self.z_sim))

    def __moments(self):
        y = self.y
        n = self.n
        w = self.w
        n2 = n * n
        s0 = w.s0 if isinstance(w, W) else self.summary.s0
        self.EG = s0 / (n * (n - 1))
        s02 = s0 * s0
        s1 = w.s1 if isinstance(w, W) else self.summary.s1
        s2 = w.s2 if isinstance(w, W) else self.summary.s2
        b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
        b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
        b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
        b4 = s1 - s2 + s02
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        y2 = y * y
        y3 = y * y2
        y4 = y2 * y2
        EG2 = b0 * (sum(y2) ** 2) + b1 * sum(y4) + b2 * (sum(y) ** 2) * sum(y2)
        EG2 += b3 * sum(y) * sum(y3) + b4 * (sum(y) ** 4)
        EG2NUM = EG2
        EG2DEN = ((sum(y) ** 2 - sum(y2)) ** 2) * n * (n - 1) * (n - 2) * (n - 3)
        self.EG2 = EG2NUM / EG2DEN
        self.VG = self.EG2 - self.EG**2

    def __calc(self, y):
        yl = lag_spatial(self.w, y) if isinstance(self.w, W) else self.w.lag(y)
        self.num = y * yl
        return self.num.sum() / self.den_sum

    @property
    def _statistic(self):
        """Standardized accessor for esda statistics"""
        return self.G


class G_Local:
    """
    Generalized Local G Autocorrelation

    Parameters
    ----------
    y : array
        Variable.
    w : W | Graph
        Spatial weights instance as W or Graph aligned with y.
    transform : {'R', 'B'}
        The type of w, either 'B' (binary) or 'R' (row-standardized).
    permutations : int
        The number of random permutations for calculating pseudo p-values.
    star : bool or float
        Whether to include focal observation in sums (default: False).
        If the row-transformed weight is provided, then this is the default
        value to use within the spatial lag. Generally, weights should be
        provided in binary form, and standardization/self-weighting will be
        handled by the function itself. A number sets the self-weight
        explicitly. In the component reading (``by_component=True``) an
        omitted ``star`` means the same as ``None``: the Gi/Gi* variant is
        inferred per component from the retained diagonal, ``False``
        selects Gi, ``True`` selects Gi*, and a number is used as the
        self-weight. The whole-graph reading (``by_component=False``)
        keeps its historical default of ``False``.
    seed : None or int
        Seed to ensure reproducibility of conditional randomizations. Must be
        set here, and not outside of the function, since numba does not
        correctly interpret external seeds nor numpy.random.RandomState
        instances.
    island_weight : float
        Value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.
    alternative : None or str, optional
        The alternative hypothesis for conditional randomization. See
        ``crand.crand()`` for complete description.
    by_component : bool
        (default=False) If True, the analysis is decomposed over the
        weakly connected components of the graph induced by the observed
        (non-NaN) sites. Sites with a missing value are dropped from the
        graph together with all of their edges before the components are
        computed, so a missing observation can split a network into
        separate pieces. Each component normalises its statistic on its
        own observations, the Gi/Gi* (``star``) handling is resolved per
        component on the retained weights, and conditional randomization
        only draws from the remaining sites of the same component. A
        site that is attached to its component only through incoming
        edges (for example a site whose only outgoing weight is a
        self-loop) is still part of the component and is analysed as
        such. Results are reported at the original site positions; sites
        without a valid statistic (missing observations, components with
        fewer than three observed sites, constant components, and cells
        whose denominator is zero) are returned as NaN. The input ``y``
        and ``w`` are never modified, and ``self.w`` then holds the
        transformed weights of the observed subgraph. The default
        (False) keeps the whole-graph analysis.
    component_labels_ : array
        (only set when ``by_component=True``) (n,) component id of every
        site in the observed graph; -1 marks sites dropped because their
        value is NaN. In this mode ``y_sum`` holds, for every site, the
        sum of its component's observed values (NaN for dropped sites),
        and ``sim`` keeps one column per original site, with all-NaN
        columns where no statistic was computed.

    Attributes
    ----------
    y : array
        Original variable.
    w : W | Graph
        Spatial weights instance as W or Graph aligned with y.
    permutations : int
        The number of permutations.
    Gs : array
        Values of the original G statistic in Getis & Ord (1992).
    EGs : float
        Expected value of Gs under the normality assumption. The value is
        scalar, since the expectation is identical across all observations.
    VGs : array
        Variance values of Gs under the normality assumption.
    Zs : array
        Standardized Gs.
    p_norm : array
        P-values under the normality assumption (one-sided). For two-sided
        tests, this value should be multiplied by 2.
    sim : array
        Array of simulated statistic values (if ``permutations > 0``).
    p_sim : array
        P-values based on permutations (one-sided). The null is spatial
        randomness; the alternative is that the observed G is extreme.
    EG_sim : array
        Average value of G from permutations.
    VG_sim : array
        Variance of G from permutations.
    seG_sim : array
        Standard deviation of G under permutations.
    z_sim : array
        Standardized G based on permutations. NaN where the simulated
        reference distribution has no spread (for example a Gi* site whose
        focal value and self-weight exhaust its component), since the
        standardisation is then undefined.
    p_z_sim : array
        P-values based on the standard normal approximation from permutations
        (one-sided). NaN wherever ``z_sim`` is undefined.

    Notes
    -----
    To compute moments of Gs under the normality assumption, PySAL considers
    w to be either binary or row-standardized. For a binary weights object,
    the weight value for self is 1. For a row-standardized weights object,
    the weight value for self is 1 / (the number of its neighbors + 1).

    For technical details see :cite:`Getis_2010` and :cite:`Ord_2010`.

    Examples
    --------
    >>> import libpysal

    Preparing a point data set.

    >>> points = [(10, 10), (20, 10), (40, 10), (15, 20), (30, 20), (30, 30)]

    Creating a weights object from points.

    >>> w = libpysal.weights.DistanceBand(points, threshold=15)

    Preparing a variable.

    >>> import numpy
    >>> y = numpy.array([2, 3, 3.2, 5, 8, 7])

    Applying Getis and Ord local G test using a binary weights object.

    >>> from esda import G_Local
    >>> lg = G_Local(y, w, transform='B', seed=12345, alternative='two-sided')

    Examining the results.

    >>> lg.Zs
    array([-1.0136729 , -0.04361589,  1.31558703, -0.31412676,  1.15373986,
           1.77833941])
    >>> round(lg.p_sim[0], 3)
    np.float32(0.413)

    P-value based on standard normal approximation from permutations.

    >>> round(lg.p_z_sim[0], 3)
    np.float64(0.153)

    Applying Getis and Ord local G* test using a binary weights object.

    >>> lg_star = G_Local(
    ...     y, w, transform='B', star=True, seed=12345, alternative='two-sided',
    ... )

    Examining the results.

    >>> lg_star.Zs
    array([-1.39727626, -0.28917762,  0.65064964, -0.28917762,  1.23452088,
           2.02424331])
    >>> round(lg_star.p_sim[0], 3)
    np.float32(0.413)

    Applying Getis and Ord local G test using a row-standardized weights object.

    >>> lg = G_Local(y, w, transform='R', seed=12345, alternative='two-sided')

    Examining the results.

    >>> lg.Zs
    array([-0.62074534, -0.01780611,  1.31558703, -0.12824171,  0.28843496,
           1.77833941])
    >>> round(lg.p_sim[0], 3)
    np.float32(0.413)

    Applying Getis and Ord local G* test using a row-standardized weights object.

    >>> lg_star = G_Local(
    ...     y, w, transform='R', star=True, seed=12345, alternative='two-sided',
    ... )

    Examining the results.

    >>> lg_star.Zs
    array([-0.62488094, -0.09144599,  0.41150696, -0.09144599,  0.24690418,
           1.28024388])
    >>> round(lg_star.p_sim[0], 3)  # doctest: +SKIP
    np.float32(0.209)
    """

    def __init__(
        self,
        y,
        w,
        transform="R",
        permutations=PERMUTATIONS,
        star=_STAR_UNSET,
        keep_simulations=True,
        n_jobs=-1,
        seed=None,
        island_weight=0,
        alternative=None,
        by_component=False,
    ):
        y = np.asarray(y).flatten()
        if by_component:
            self._init_by_component(
                y,
                w,
                transform,
                permutations,
                star,
                keep_simulations,
                n_jobs,
                seed,
                island_weight,
                alternative,
            )
            return
        # whole-graph reading: an omitted ``star`` keeps its historical
        # Gi default (False); only the by-component reading treats an
        # omitted value as "infer the self-weight per component".
        if star is _STAR_UNSET:
            star = False
        self.n = len(y)
        self.y = y
        w, star = _infer_star_and_structure_w(w, star, transform)
        if isinstance(w, W):
            w.transform = transform
        else:
            w = w.transform(transform)
        self.w_transform = transform
        self.w = w
        self.permutations = permutations
        self.star = star
        self.calc()
        self.p_norm = stats.norm.sf(np.abs(self.Zs))
        if permutations:
            self.p_sim, self.rGs = _crand_plus(
                y,
                w,
                self.Gs,
                permutations,
                keep_simulations,
                n_jobs=n_jobs,
                stat_func=_g_local_star_crand if star else _g_local_crand,
                scaling=y.sum(),
                seed=seed,
                island_weight=island_weight,
                alternative=alternative,
            )
            if keep_simulations:
                self.sim = sim = self.rGs.T
                self.EG_sim = sim.mean(axis=0)
                self.seG_sim = sim.std(axis=0)
                self.VG_sim = self.seG_sim * self.seG_sim
                self.z_sim = (self.Gs - self.EG_sim) / self.seG_sim
                self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))

    def _init_by_component(
        self,
        y,
        w,
        transform,
        permutations,
        star,
        keep_simulations,
        n_jobs,
        seed,
        island_weight,
        alternative,
    ):
        """Component-wise local G inference (see ``by_component``)."""
        n = len(y)
        if w.n != n:
            raise ValueError(
                f"Length of y ({n}) does not match the number of "
                f"observations in w ({w.n})."
            )
        if np.isinf(y).any():
            raise ValueError(
                "y contains infinite values; only finite values or NaN are supported."
            )
        # In the component reading an omitted ``star`` means "infer Gi/Gi*
        # from each component's retained diagonal", exactly like an
        # explicit ``None``.
        if star is _STAR_UNSET:
            star = None
        self.n = n
        self.y = y
        self.permutations = permutations
        self.w_transform = transform

        # outputs at original positions; missing unless computed
        self.Gs = np.full(n, np.nan)
        self.EGs = np.full(n, np.nan)
        self.VGs = np.full(n, np.nan)
        self.Zs = np.full(n, np.nan)
        # component id per original site; -1 marks unobserved sites
        self.component_labels_ = np.full(n, -1, dtype=int)
        # per-site sum of the component's observed values
        self.y_sum = np.full(n, np.nan)

        valid, adj_observed, _, comp_ptr, perm = mask_unobserved(y, w.sparse)
        n_valid = perm.shape[0]
        n_comp = comp_ptr.shape[0] - 1
        valid_pos = np.flatnonzero(valid)
        valid_pos_packed = valid_pos[perm]
        active = np.zeros(0, dtype=bool)
        self.star = star if isinstance(star, (bool, np.bool_)) else False

        if n_valid:
            # the caller's weights object is never modified: the observed
            # subgraph is rebuilt, Gi/Gi* is resolved per component on the
            # retained edges, and only then is the requested transform
            # applied (matching the whole-graph order)
            w_observed = rebuild_sparse_weights(w, adj_observed)
            adjacency = w_observed.sparse[perm][:, perm].tocsr()
            adjacency.eliminate_zeros()
            # an array-valued ``star`` is indexed by observed-site position;
            # reorder it into component-major order to match ``adjacency``
            star_for_components = star
            if (
                star is not None
                and not isinstance(star, (bool, np.bool_))
                and np.asarray(star).ndim == 1
            ):
                star_for_components = np.asarray(star, dtype=np.float64)[perm]
            adjacency, star_any = _resolve_component_star(
                adjacency, comp_ptr, star_for_components, transform
            )
            self.star = star_any
            # write the resolved diagonal back to original-observed order
            # and rebuild the weights object before transforming it
            resolved_diag = adjacency.diagonal().astype(np.float64)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.shape[0])
            resolved_obs = adj_observed.astype(np.float64).tolil(copy=True)
            resolved_obs.setdiag(resolved_diag[inv_perm])
            w_observed = rebuild_sparse_weights(w, resolved_obs.tocsr())
            w_observed = transform_observed(w_observed, transform)
            self.w = w_observed
            adjacency = w_observed.sparse[perm][:, perm].tocsr()
            adjacency.eliminate_zeros()

            y_packed = y[valid][perm]
            sizes = np.diff(comp_ptr)
            labels_packed = np.repeat(np.arange(n_comp), sizes)

            # normalise on each component's own observations
            comp_sum = np.bincount(labels_packed, weights=y_packed, minlength=n_comp)
            comp_min = np.minimum.reduceat(y_packed, comp_ptr[:-1])
            comp_max = np.maximum.reduceat(y_packed, comp_ptr[:-1])
            active_comp = (sizes >= 3) & (comp_min != comp_max)

            # Gi/Gi* is a property of the component, exactly as if each
            # component were handed to the whole-graph estimator on its own:
            # a component with any retained self-loop is Gi* for every one
            # of its sites (sites without a loop simply carry a zero
            # self-weight), otherwise it is Gi. This matters when components
            # with and without self-loops are mixed in one table.
            self_weights = adjacency.diagonal().astype(np.float64)
            loop_site = self_weights != 0
            star_comp = np.zeros(n_comp, dtype=bool)
            if n_comp:
                star_comp = np.maximum.reduceat(
                    loop_site.astype(np.int64), comp_ptr[:-1]
                ).astype(bool)
            star_site = star_comp[labels_packed]

            remove_self = ~star_site
            n_c = sizes[labels_packed]
            N_c = n_c - remove_self
            y_sum = comp_sum[labels_packed]
            den = y_sum - y_packed * remove_self
            Gs_packed = np.full(n_valid, np.nan)
            active = active_comp[labels_packed]
            with np.errstate(all="ignore"):
                Gs_packed[active] = (adjacency @ y_packed)[active] / den[active]
            # cells with a zero denominator have no statistic and no
            # permutation, but their value stays in the sampling pool
            active &= np.isfinite(Gs_packed)

            # analytical moments, with each component's own N and sums
            with np.errstate(all="ignore"):
                ssq = np.bincount(labels_packed, weights=y_packed**2, minlength=n_comp)
                empirical_mean = (y_sum - y_packed * remove_self) / N_c
                mean_of_squares = (ssq[labels_packed] - y_packed**2 * remove_self) / N_c
                empirical_variance = mean_of_squares - empirical_mean**2
                cardinality = np.asarray(adjacency.sum(axis=1)).flatten()
                expected_value = cardinality / N_c
                expected_variance = cardinality * (N_c - cardinality)
                expected_variance /= N_c - 1
                expected_variance *= 1 / N_c**2
                expected_variance *= empirical_variance / (empirical_mean**2)
                Zs_packed = (Gs_packed - expected_value) / np.sqrt(expected_variance)
            # degenerate components report missing moments as well
            degenerate = ~active_comp[labels_packed]
            expected_value[degenerate] = np.nan
            expected_variance[degenerate] = np.nan
            Zs_packed[degenerate] = np.nan

            self.y_sum[valid_pos_packed] = y_sum
            self.Gs[valid_pos_packed] = Gs_packed
            self.EGs[valid_pos_packed] = expected_value
            self.VGs[valid_pos_packed] = expected_variance
            self.Zs[valid_pos_packed] = Zs_packed
            self.component_labels_[valid_pos_packed] = labels_packed
        else:
            # no observed sites at all: keep the caller's weights as passed
            self.w = w

        self.p_norm = stats.norm.sf(np.abs(self.Zs))

        if permutations:
            if n_valid and active.any():
                self_weights = adjacency.diagonal().astype(np.float64)
                adj_off = adjacency.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adj_off.setdiag(0)
                    adj_off.eliminate_zeros()
                adj_off = adj_off.tocsr()
                adj_off.sort_indices()
                other_weights = adj_off.data.astype(np.float64)
                cardinalities = np.asarray((adj_off != 0).sum(axis=1)).flatten()
                # The Gi* kernel reduces to the Gi kernel when the site's
                # fixed self-weight is zero, so one kernel serves a table
                # that mixes both variants. The per-site denominator
                # (``den``) is what distinguishes them: a Gi site divides
                # by the component sum minus its own value, a Gi* site by
                # the whole component sum.
                p_sims, rlocals = crand_components(
                    y_packed,
                    Gs_packed,
                    active,
                    comp_ptr,
                    cardinalities,
                    self_weights,
                    other_weights,
                    den,
                    seed,
                    permutations,
                    keep_simulations,
                    n_jobs,
                    _g_local_star_crand,
                    island_weight=island_weight,
                    alternative=alternative,
                )
            else:
                p_sims = np.full(n_valid, np.nan, dtype=np.float32)
                rlocals = (
                    np.full((n_valid, permutations), np.nan)
                    if keep_simulations
                    else np.empty((1, 1))
                )
            self.p_sim = np.full(n, np.nan)
            if n_valid:
                self.p_sim[valid_pos_packed] = p_sims
            if keep_simulations:
                self.rGs = np.full((n, permutations), np.nan)
                if n_valid:
                    self.rGs[valid_pos_packed] = rlocals
                self.sim = self.rGs.T
                self.EG_sim = self.sim.mean(axis=0)
                self.seG_sim = self.sim.std(axis=0)
                self.VG_sim = self.seG_sim * self.seG_sim
                # See ``Moran_Local``: a degenerate reference distribution
                # leaves the standardisation undefined, so keep the
                # interface's undefined representation rather than a
                # spurious finite score computed from rounding noise.
                degenerate_col = degenerate_reference_columns(self.sim)
                with np.errstate(divide="ignore", invalid="ignore"):
                    self.z_sim = (self.Gs - self.EG_sim) / self.seG_sim
                self.z_sim[degenerate_col] = np.nan
                self.p_z_sim = stats.norm.sf(np.abs(self.z_sim))
            else:
                self.rGs = np.empty((1, 1))

    def __crand(self, keep_simulations):
        warnings.warn(
            "G_Local.__crand is deprecated and will be removed in a future release. "
            "Use the current conditional randomization implementation based on "
            "`esda.crand.crand` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        y = self.y
        if keep_simulations:
            rGs = np.zeros((self.n, self.permutations))
        larger = np.zeros((self.n,))
        n_1 = self.n - 1
        rid = list(range(n_1))
        prange = list(range(self.permutations))
        if isinstance(self.w, W):
            k = self.w.max_neighbors + 1
        else:
            k = self.w.cardinalities.max()
        rids = np.array([np.random.permutation(rid)[0:k] for i in prange])
        ids = np.arange(self.w.n)
        wc = self.__getCardinalities()
        if self.w_transform == "r":  # noqa: SIM108 -- keeping readability
            den = np.array(wc) + self.star
        else:
            den = np.ones(self.w.n)
        for i in range(self.w.n):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            yi_star = y[i] * self.star
            wci = wc[i]
            rGs_i = (y[idsi[rids[:, 0:wci]]]).sum(1) + yi_star
            rGs_i = (np.array(rGs_i) / den[i]) / (self.y_sum - (1 - self.star) * y[i])
            if keep_simulations:
                rGs[i] = rGs_i
            larger[i] = (rGs_i >= self.Gs[i]).sum()
        if keep_simulations:
            self.rGs = rGs
        below = (self.permutations - larger) < larger
        larger[below] = self.permutations - larger[below]
        self.p_sim = (larger + 1) / (self.permutations + 1)

    def __getCardinalities(self):
        if isinstance(self.w, W):
            ido = self.w.id_order
            self.wc = np.array([self.w.cardinalities[ido[i]] for i in range(self.n)])
        else:
            self.wc = self.w.cardinalities.values
        return self.wc

    def calc(self):
        w = self.w
        W = w.sparse

        self.y_sum = self.y.sum()

        y = self.y
        remove_self = not self.star
        N = self.w.n - remove_self

        statistic = (W @ y) / (y.sum() - y * remove_self)

        # ----------------------------------------------------#
        # compute moments necessary for analytical inference  #
        # ----------------------------------------------------#

        empirical_mean = (y.sum() - y * remove_self) / N
        # variance looks complex, yes, but it obtains from E[x^2] - E[x]^2.
        # So, break it down to allow subtraction of the self-neighbor.
        mean_of_squares = ((y**2).sum() - (y**2) * remove_self) / N
        empirical_variance = mean_of_squares - empirical_mean**2

        # Since we have corrected the diagonal, this should work
        cardinality = np.asarray(W.sum(axis=1)).squeeze()
        expected_value = cardinality / N

        expected_variance = cardinality * (N - cardinality)
        expected_variance /= N - 1
        expected_variance *= 1 / N**2
        expected_variance *= empirical_variance / (empirical_mean**2)

        z_scores = (statistic - expected_value) / np.sqrt(expected_variance)

        self.Gs = statistic
        self.EGs = expected_value
        self.VGs = expected_variance
        self.Zs = z_scores

    @property
    def _statistic(self):
        """Standardized accessor for esda statistics"""
        return self.Gs


def _resolve_component_star(adjacency, comp_ptr, star, transform):
    """Resolve the Gi/Gi* self-weight rule independently per component.

    The whole-graph helper :func:`_infer_star_and_structure_w` decides
    ``star`` once for the entire matrix. In the component reading a single
    table may mix components that carry self-loops with components that do
    not, so the decision (and any default self-weight that has to be
    filled in) is per component. The weights are still untransformed here;
    the caller applies the requested transform afterwards, exactly as the
    whole-graph path does, so a binary transform binarises a numeric
    self-weight and a row-standardised transform includes it in the row
    sum.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Retained weights in component-major order, before the requested
        transform.
    comp_ptr : ndarray
        (n_components + 1,) component offsets in ``adjacency``.
    star : None | bool | float | ndarray
        The caller's ``star`` setting. ``None`` asks to infer it from each
        component's retained diagonal; ``False`` selects Gi, ``True``
        selects Gi*, and a number is used as the self-weight, following
        the existing rules (``transform`` decides the binary/row-standard
        default when the component has no diagonal).
    transform : str
        Requested transform, ``'B'`` or ``'R'``.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        A copy with the resolved diagonal. The caller's matrix is not
        modified.
    star : bool
        Whether any component ends up as Gi* (matches the whole-graph
        convention for ``self.star``).
    """
    if transform.lower() not in ("r", "b"):
        raise AssertionError(
            f'Transforms must be binary "b" or row-standardized "r". '
            f"Recieved: {transform}"
        )
    adjacency = adjacency.tocsr(copy=True)
    n_valid = adjacency.shape[0]
    n_comp = comp_ptr.shape[0] - 1
    star_warn = (
        "Gi* requested, but (a) weights are already row-standardized,"
        " (b) no weights are on the diagonal, and"
        " (c) no default value supplied to star. Assuming that the"
        " self-weight is equivalent to the maximum weight in the"
        " row. To use a different default (like, .5), set `star=.5`,"
        " or use libpysal.weights.fill_diagonal() to set the diagonal"
        " values of your weights matrix and use `star=None` in Gi_Local."
    )

    # a numeric / array self-weight applies to every site; booleans and
    # None are resolved from each component's retained diagonal below
    if star is not None and not isinstance(star, (bool, np.bool_)):
        try:
            cand = np.asarray(star, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                f"Type of star ({type(star)}) not understood."
                " Must be an integer, boolean, float, or numpy.ndarray."
            ) from None
        if cand.ndim == 0:
            self_weights = np.full(n_valid, float(cand))
        elif cand.shape == (n_valid,):
            self_weights = cand.astype(np.float64).copy()
        else:
            raise TypeError(
                f"Type of star ({type(star)}) not understood."
                " Must be an integer, boolean, float, or numpy.ndarray."
            )
        adjacency.setdiag(self_weights)
        adjacency.eliminate_zeros()
        return adjacency, bool((self_weights > 0).any())

    diag = adjacency.diagonal().astype(np.float64)
    self_weights = diag.copy()
    star_any = False
    for c in range(n_comp):
        s, e = int(comp_ptr[c]), int(comp_ptr[c + 1])
        nonzero = diag[s:e] != 0
        resolved = bool(nonzero.any()) if star is None else bool(star)
        if not resolved:
            self_weights[s:e] = 0.0
        elif nonzero.any():
            pass  # keep this component's own diagonal
        elif transform.lower() == "b":
            self_weights[s:e] = 1.0
        else:  # 'r': historical "row maximum" default, per component
            warnings.warn(star_warn, UserWarning, stacklevel=3)
            row_max = np.asarray(adjacency[s:e].max(axis=1).todense()).flatten()
            self_weights[s:e] = row_max
        star_any = star_any or resolved

    adjacency.setdiag(self_weights)
    adjacency.eliminate_zeros()
    return adjacency, star_any


def _infer_star_and_structure_w(weights, star, transform):
    assert transform.lower() in ("r", "b"), (
        f'Transforms must be binary "b" or row-standardized "r".Recieved: {transform}'
    )
    adj_matrix = weights.sparse
    diagonal = adj_matrix.diagonal()
    zero_diagonal = (diagonal == 0).all()

    # Gi has a zero diagonal, Gi* has a nonzero diagonal
    star = (not zero_diagonal) if star is None else star

    # Want zero diagonal but do not have it
    if (not zero_diagonal) & (star is False):
        if isinstance(weights, W):
            weights = fill_diagonal(weights, 0)
        else:
            weights = weights.assign_self_weight(0).eliminate_zeros()
    # Want nonzero diagonal and have it
    elif (not zero_diagonal) & (star is True):  # noqa: SIM114 -- keeping readability
        weights = weights
    # Want zero diagonal and have it
    elif zero_diagonal & (star is False):
        weights = weights
    # Want nonzero diagonal and do not have it
    elif zero_diagonal & (star is True):
        # if the input is binary or requested transform is binary,
        # set the diagonal to 1.
        star_warn = (
            "Gi* requested, but (a) weights are already row-standardized,"
            " (b) no weights are on the diagonal, and"
            " (c) no default value supplied to star. Assuming that the"
            " self-weight is equivalent to the maximum weight in the"
            " row. To use a different default (like, .5), set `star=.5`,"
            " or use libpysal.weights.fill_diagonal() to set the diagonal"
            " values of your weights matrix and use `star=None` in Gi_Local."
        )
        if isinstance(weights, W):
            if transform.lower() == "b" or weights.transform.lower() == "b":
                weights = fill_diagonal(weights, 1)

            # if we know the target is row-standardized, use the row max
            # this works successfully for effectively binary but "O"-transformed input
            elif transform.lower() == "r":
                # This warning is presented in the documentation as well
                warnings.warn(star_warn, UserWarning, stacklevel=2)
                weights = fill_diagonal(
                    weights, np.asarray(adj_matrix.max(axis=1).todense()).flatten()
                )
        else:
            if transform.lower() == "b" or weights.transformation.lower() == "b":
                weights = weights.assign_self_weight(1)
            elif transform.lower() == "r":
                warnings.warn(star_warn, UserWarning, stacklevel=2)
                weights = weights.assign_self_weight(
                    np.asarray(adj_matrix.max(axis=1).todense()).flatten()
                )
    else:  # star was something else, so try to fill the weights with it
        try:
            if isinstance(weights, W):
                weights = fill_diagonal(weights, star)
            else:
                weights = weights.assign_self_weight(star)
        except TypeError:
            raise TypeError(
                f"Type of star ({type(star)}) not understood."
                f" Must be an integer, boolean, float, or numpy.ndarray."
            ) from None
    star = (weights.sparse.diagonal() > 0).any()
    if isinstance(weights, W):
        weights.transform = transform
    else:
        weights = weights.transform(transform)

    return weights, star


# --------------------------------------------------------------
# Conditional Randomization Function Implementations
# --------------------------------------------------------------


@_njit(fastmath=True)
def _g_local_crand(i, z, permuted_ids, weights_i, scaling):
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zrand @ other_weights) / (scaling - zi)


@_njit(fastmath=True)
def _g_local_star_crand(i, z, permuted_ids, weights_i, scaling):
    self_weight = weights_i[0]
    other_weights = weights_i[1:]
    zi, zrand = _prepare_univariate(i, z, permuted_ids, other_weights)
    return (zrand @ other_weights + self_weight * zi) / scaling
