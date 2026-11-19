"""
Centralised conditional randomisation engine. Numba accelerated.
"""

import hashlib
import importlib
import os
import warnings

import numpy as np
from libpysal.graph import Graph
from libpysal.weights import WSP, W
from scipy.sparse.csgraph import connected_components

from .significance import _permutation_significance

try:
    from numba import boolean, njit, prange
except (ImportError, ModuleNotFoundError):
    from libpysal.common import jit as njit

    prange = range
    boolean = bool


__all__ = ["crand", "crand_components"]

#######################################################################
#                   Utilities for all functions                       #
#######################################################################


@njit(fastmath=True)
def vec_permutations(max_card: int, n: int, k_replications: int, seed: int):
    """
    Generate `max_card` permuted IDs, sampled from `n` without replacement,
    `k_replications` times
    ...

    Parameters
    ----------
    max_card : int
        Number of permuted IDs to generate per sample
    n : int
        Size of the sample to sample IDs from
    k_replications : int
        Number of samples of permuted IDs to perform
    seed : int
        Seed to ensure reproducibility of conditional randomizations

    Returns
    -------
    result : ndarray
        (k_replications, max_card) array with permuted IDs
    """
    np.random.seed(seed)
    result = np.empty((k_replications, max_card), dtype=np.int64)
    for k in prange(k_replications):
        result[k] = np.random.choice(n - 1, size=max_card, replace=False)
    return result


def crand(
    z,
    w,
    observed,
    permutations,
    keep,
    n_jobs,
    stat_func,
    scaling=None,
    seed=None,
    island_weight=0,
    alternative=None,
):
    """
    Conduct conditional randomization of a given input using the provided
    statistic function. Numba accelerated.
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    w : libpysal.weights.W
        Spatial weights object
    observed : ndarray
        (N,) array with observed values
    permutations : int
        Number of permutations for conditional randomisation
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
     seed : None/int
        Seed to ensure reproducibility of conditional randomizations
    alternative : None | str = None
        The alternative hypothesis for conditional randomization. The current behavior
        is ``None``, which defaults to ``'directed'``. We strongly recommend moving
        to ``'two-sided'``. Valid alternatives include:
            * 'two-sided'
            * 'greater'
            * 'lesser'
            * 'directed' -- Current default behavior
            * 'folded'

    Returns
    -------
    p_sim : ndarray
        (N,) array with pseudo p-values from conditional permutation
    rlocals : ndarray
        If keep=True, (N, permutations) array with simulated values
        of stat_func under the null of spatial randomness; else, empty (1, 1) array
    """
    adj_matrix = w.sparse

    n = len(z)
    if z.ndim == 2:
        if z.shape[1] == 2:
            # assume that matrix is [X Y], and scaling is moran-like
            scaling = (
                (n - 1) / (z[:, 0] * z[:, 0]).sum() if (scaling is None) else scaling
            )
        elif z.shape[1] == 1:
            # assume that matrix is [X], and scaling is moran-like
            scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
        else:
            raise NotImplementedError(
                f"multivariable input is not yet supported in "
                f"conditional randomization. Received `z` of shape {z.shape}"
            )
    elif z.ndim == 1:
        scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
    else:
        raise NotImplementedError(
            f"multivariable input is not yet supported in "
            f"conditional randomization. Received `z` of shape {z.shape}"
        )

    if alternative is None:
        warnings.warn(
            "The alternative hypothesis for conditional randomization"
            " is changing in the next major release of esda. We recommend"
            " setting alternative='two-sided', which will generally"
            " double the p-value returned."
            " To retain the current behavior, set alternative='directed'."
            " We strongly recommend moving to alternative='two-sided'.",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: replace this with 'two-sided' by next major release
        alternative = "directed"
    if alternative not in ("two-sided", "greater", "lesser", "directed", "folded"):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not"
            " one of the supported options: 'two-sided', 'greater', "
            "'lesser', 'directed', 'folded')"
        )

    # paralellise over permutations?
    if seed is None:
        seed = np.random.randint(12345, 12345000)

    # we need to be careful to shuffle only *other* sites, not
    # the self-site. This means we need to
    # extract the self-weight, if any
    self_weights = adj_matrix.diagonal()
    # force the self-site weight to zero
    with warnings.catch_warnings():
        # massive changes to sparsity incur a cost, but it's not
        # large for simply changing the diag
        warnings.simplefilter("ignore")
        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()
    # extract the weights from a now no-self-weighted adj_matrix
    other_weights = adj_matrix.data.astype(z.dtype)  # cast is forced by @ in numba
    # use the non-self weight as the cardinality, since
    # this is the set we have to randomize.
    # if there is a self-neighbor, we need to *not* shuffle the
    # self neighbor, since conditional randomization conditions on site i.
    cardinalities = np.array((adj_matrix != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    permuted_ids = vec_permutations(max_card, n, permutations, seed)

    if n_jobs != 1 and not importlib.util.find_spec("joblib"):
        warnings.warn(
            f"Parallel processing is requested (n_jobs={n_jobs}),"
            f" but joblib cannot be imported. n_jobs will be set"
            f" to 1.",
            stacklevel=2,
        )
        n_jobs = 1

    if n_jobs == 1:
        p_sims, rlocals = compute_chunk(
            0,  # chunk start
            z,  # chunked z, for serial this is the entire data
            z,  # all z, for serial this is also the entire data
            observed,  # observed statistics
            cardinalities,  # cardinalities conforming to chunked z
            self_weights,  # n-length vector containing the self-weights.
            other_weights,  # flat weights buffer
            permuted_ids,  # permuted ids
            scaling,  # scaling applied to all statistics
            keep,  # whether or not to keep the local statistics
            stat_func,
            island_weight,
            alternative=alternative,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs > len(z):
            n_jobs = len(z)
        # Parallel implementation
        p_sims, rlocals = parallel_crand(
            z,
            observed,
            cardinalities,
            self_weights,
            other_weights,
            permuted_ids,
            scaling,
            n_jobs,
            keep,
            stat_func,
            island_weight,
            alternative=alternative,
        )

    return p_sims, rlocals


@njit(parallel=False, fastmath=True)
def compute_chunk(
    chunk_start: int,
    z_chunk: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    keep: bool,
    stat_func,
    island_weight: float,
    alternative: str,
):
    """
    Compute conditional randomisation for a single chunk
    ...

    Parameters
    ----------
    chunk_start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float
        Scaling value to apply to every local statistic
    keep : bool
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Returns
    -------
    larger : ndarray
        (n_chunk,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (n_chunk, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    chunk_n = z_chunk.shape[0]
    n_samples = z.shape[0]
    p_permutations, k_max_card = permuted_ids.shape
    p_sims = np.zeros((chunk_n,), dtype=np.float32)
    rlocals = np.empty((chunk_n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    mask = np.ones((n_samples,), dtype=np.int8) == 1
    wloc = 0

    for i in range(chunk_n):
        cardinality = cardinalities[i]
        if cardinality == 0:  # deal with islands
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            # we need to fix the self-weight to the first position
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            # this chomps the next `cardinality` weights off of `weights`
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        mask[chunk_start + i] = False
        rstats = stat_func(chunk_start + i, z, permuted_ids, weights_i, scaling)
        p_sims[i] = _permutation_significance(
            observed[i], rstats, alternative=alternative
        ).item()
        if keep:
            rlocals[i] = rstats

    return p_sims, rlocals


#######################################################################
#                   Parallel Implementation                           #
#######################################################################


@njit(fastmath=True)
def build_weights_offsets(cardinalities: np.ndarray, n_chunks: int):
    """
    Utility function to construct offsets into the weights
    flat data array found in the W.sparse.data object
    ...

    Parameters
    ----------
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    n_chunks : int
        Number of chunks to split the weights into

    Returns
    -------
    boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk
    """
    boundary_points = np.zeros((n_chunks + 1,), dtype=np.int64)
    n = cardinalities.shape[0]
    chunk_size = np.int64(n / n_chunks) + 1
    start = 0
    for i in range(n_chunks):
        advance = cardinalities[start : start + chunk_size].sum()
        boundary_points[i + 1] = boundary_points[i] + advance
        start += chunk_size
    return boundary_points


@njit(fastmath=True)
def chunk_generator(
    n_jobs: int,
    starts: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    w_boundary_points: np.ndarray,
):
    """
    Construct chunks to iterate over within numba in parallel
    ...

    Parameters
    ----------
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    starts : ndarray
        (n_chunks+1,) array of positional starts for each chunk
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    weights : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    w_boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk

    Yields
    ------
    start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed_chunk : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities_chunk : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    weights_chunk : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    """
    chunk_size = starts[1] - starts[0]
    for i in range(n_jobs):
        start = starts[i]
        z_chunk = z[start : (start + chunk_size)]
        self_weights_chunk = self_weights[start : (start + chunk_size)]
        observed_chunk = observed[start : (start + chunk_size)]
        cardinalities_chunk = cardinalities[start : (start + chunk_size)]
        w_chunk = other_weights[w_boundary_points[i] : w_boundary_points[i + 1]]
        yield (
            start,
            z_chunk,
            z,
            observed_chunk,
            cardinalities_chunk,
            self_weights_chunk,
            w_chunk,
        )


def parallel_crand(
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    n_jobs: int,
    keep: bool,
    stat_func,
    island_weight,
    alternative: str = "directed",
):
    """
    Conduct conditional randomization in parallel using numba
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float64
        Scaling value to apply to every local statistic
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.
    Returns
    -------
    larger : ndarray
        (N,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (N, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    from joblib import Parallel, delayed, parallel_backend

    n = z.shape[0]
    w_boundary_points = build_weights_offsets(cardinalities, n_jobs)
    chunk_size = n // n_jobs + 1
    starts = np.arange(n_jobs + 1) * chunk_size
    # ------------------------------------------------------------------
    # Set up output holders
    rlocals = np.empty((n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    # ------------------------------------------------------------------
    # Joblib parallel loop by chunks

    # construct chunks using a generator
    chunks = chunk_generator(
        n_jobs,
        starts,
        z,
        observed,
        cardinalities,
        self_weights,
        other_weights,
        w_boundary_points,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk)(
                *pars,
                permuted_ids,
                scaling,
                keep,
                stat_func,
                island_weight,
                alternative,
            )
            for pars in chunks
        )

    p_sims, rlocals = zip(*worker_out, strict=True)
    p_sims = np.hstack(p_sims).squeeze()
    rlocals = np.vstack(rlocals).squeeze()
    return p_sims, rlocals


#######################################################################
#                   Local statistical functions                       #
#######################################################################


@njit(fastmath=False)
def _prepare_univariate(i, z, permuted_ids, weights_i):
    mask = np.ones_like(z, dtype=boolean)
    mask[i] = False
    z_no_i = z[mask]
    cardinality = len(weights_i)
    flat_permutation_ids = permuted_ids[:, :cardinality].flatten()
    zrand = z_no_i[flat_permutation_ids].reshape(-1, cardinality)
    return z[i], zrand


@njit(fastmath=False)
def _prepare_bivariate(i, z, permuted_ids, weights_i):
    zx = z[:, 0]
    zy = z[:, 1]

    cardinality = len(weights_i)

    mask = np.ones_like(zx, dtype=boolean)
    mask[i] = False
    zx_no_i = zy[mask]
    zy_no_i = zx[mask]

    flat_permutation_indices = permuted_ids[:, :cardinality].flatten()

    zxrand = zx_no_i[flat_permutation_indices].reshape(-1, cardinality)
    zyrand = zy_no_i[flat_permutation_indices].reshape(-1, cardinality)

    return zx[i], zxrand, zy[i], zyrand


@njit(fastmath=True)
def local(i, z, permuted_ids, weights_i, scaling):
    raise NotImplementedError
    # returns (k_permutations,) array of random statistics for observation i


#######################################################################
#          Connected-component conditional randomisation              #
#######################################################################
#
# The functions below implement the "by component" reading of the local
# statistics: weakly connected components of the graph induced by the
# observed (non-NaN) sites each get their own baseline, and conditional
# randomisation only draws from the remaining sites of the same
# component. The existing ``crand`` whole-graph engine is untouched.
#
# Design notes
# ------------
# * The estimators pack the observed sites component-major (components
#   are contiguous blocks) and hand this driver flat arrays plus the
#   component offsets ``comp_ptr``. Weights stay in sparse (CSR) form
#   throughout; only the flat data buffer is passed around.
# * Every component is seeded from the master seed plus a hash of its
#   own values and weights (in component-relative order). A component's
#   permutation stream therefore does not depend on how many other
#   components exist, on where its rows sit in the table, or on the
#   number of worker processes. Recomputing one component on its own
#   with the same seed reproduces the same draws.
# * Work is scheduled per component. Small components are batched into
#   shared tasks so that many tiny components do not pay one parallel
#   task each; components larger than the target task size are split
#   into site ranges so a few big components can still use all workers.
#   The split never changes results, since seeds are per component.
# * With ``keep=False`` no (n_sites, permutations) matrix is built:
#   each component only materialises its (permutations, max_cardinality)
#   id matrix and the per-site simulated statistics, both transient.


def mask_unobserved(y, adj):
    """
    Restrict a sparse adjacency matrix to the observed (non-NaN) sites and
    find the weakly connected components of the retained graph.

    Parameters
    ----------
    y : ndarray
        (n,) attribute values; NaN marks a missing observation. Sites with
        a missing value are dropped together with all of their edges.
    adj : sparse matrix
        (n, n) adjacency aligned with ``y``. A single non-zero directed
        edge is enough to keep both ends in the same component; a
        self-loop alone does not connect a site to anything.

    Returns
    -------
    valid : ndarray
        (n,) boolean mask of observed sites.
    adj_observed : csr_matrix
        (n_valid, n_valid) adjacency restricted to observed sites, in
        ascending original position order.
    labels : ndarray
        (n_valid,) component id per observed site, numbered 0..C-1 in
        order of first appearance.
    comp_ptr : ndarray
        (C+1,) offsets of the components in component-major order.
    perm : ndarray
        (n_valid,) permutation sorting the observed sites component-major
        (stable, so the relative order inside a component is preserved).
    """
    valid = ~np.isnan(y)
    adj_observed = adj[valid][:, valid].tocsr()
    adj_observed.eliminate_zeros()
    n_valid = adj_observed.shape[0]
    if n_valid == 0:
        return (
            valid,
            adj_observed,
            np.empty(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    n_comp, labels = connected_components(adj_observed, directed=False)
    # renumber to first-appearance order for a stable, compact labelling
    _, labels = np.unique(labels, return_inverse=True)
    perm = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=n_comp)
    comp_ptr = np.zeros(n_comp + 1, dtype=np.int64)
    np.cumsum(counts, out=comp_ptr[1:])
    return valid, adj_observed, labels.astype(np.int64), comp_ptr, perm


def rebuild_sparse_weights(w, adj):
    """
    Rebuild a weights object of the same kind as ``w`` from a sparse
    adjacency matrix, leaving ``w`` itself untouched. The returned object
    carries the original ("O") transformation; estimators apply their own
    transformation to it afterwards, so existing transforms (e.g. row
    standardisation) act on the retained edges only.
    """
    with warnings.catch_warnings():
        # disconnected components and islands are expected here: they are
        # the whole point of the by-component decomposition
        warnings.simplefilter("ignore")
        if isinstance(w, W):
            rebuilt = WSP(adj).to_W(silence_warnings=True)
        elif isinstance(w, Graph):
            rebuilt = Graph.from_sparse(adj)
        else:
            raise TypeError(
                f"Weights of type {type(w)} are not supported. "
                "Provide a libpysal W or Graph."
            )
    return rebuilt


def make_component_seeds(seed, z, comp_ptr, cardinalities, self_weights, other_weights):
    """
    Derive one seed per component from the master seed and a hash of the
    component's own content (values and weights, in component-relative
    order). This makes a component's permutation stream independent of
    the other components, of the row positions it sits at, and of the
    number of worker processes, so long as the relative order of its
    sites is preserved.
    """
    n_comp = comp_ptr.shape[0] - 1
    w_offsets = np.zeros(comp_ptr[-1] + 1, dtype=np.int64)
    if comp_ptr[-1] > 0:
        np.cumsum(cardinalities, out=w_offsets[1:])
    seeds = np.empty(n_comp, dtype=np.int64)
    for c in range(n_comp):
        s, e = comp_ptr[c], comp_ptr[c + 1]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.array([e - s], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(z[s:e]).tobytes())
        digest.update(np.ascontiguousarray(cardinalities[s:e]).tobytes())
        digest.update(np.ascontiguousarray(self_weights[s:e]).tobytes())
        digest.update(
            np.ascontiguousarray(other_weights[w_offsets[s] : w_offsets[e]]).tobytes()
        )
        entropy = int.from_bytes(digest.digest(), "little")
        seeds[c] = np.random.SeedSequence([int(seed), entropy]).generate_state(
            1, dtype=np.uint32
        )[0]
    return seeds


@njit(parallel=False, fastmath=True)
def compute_chunk_components(
    site_offset,
    pool,
    observed,
    active,
    cardinalities,
    self_weights,
    other_weights,
    permuted_ids,
    scaling,
    keep,
    stat_func,
    island_weight,
    alternative,
):
    """
    Compute conditional randomisation for a contiguous range of sites of
    one component. Mirrors ``compute_chunk``, but samples only from the
    component's own sites and skips sites without a statistic.

    Parameters
    ----------
    site_offset : int
        Position of the first site of the range inside ``pool``.
    pool : ndarray
        (n_component,) values of the component the sites belong to. The
        sampling pool for every site; it always includes the values of
        sites that have no statistic of their own.
    observed : ndarray
        (n_chunk,) observed statistics of the sites in the range.
    active : ndarray
        (n_chunk,) boolean mask; False marks sites without a statistic
        (no permutation is run for them, their p-value stays NaN).
    cardinalities : ndarray
        (n_chunk,) number of off-diagonal weights per site.
    self_weights : ndarray
        (n_chunk,) diagonal (self) weights of the sites.
    other_weights : ndarray
        Flat buffer with the off-diagonal weights of the sites in the
        range, in CSR data order.
    permuted_ids : ndarray
        (permutations, max_cardinality) ids into the component's sites
        minus the focal one, shared by all sites of the component.
    scaling : float
        Scaling value of this component, applied to every statistic.
    keep : bool
        If True, store the simulated statistics.
    stat_func : callable
        Statistic implementation with the same signature as in ``crand``;
        it receives component-local positions and the component pool.
    island_weight : float
        Weight for the "fake" neighbor of sites without neighbours.
    alternative : str
        Tail rule for the pseudo p-value.

    Returns
    -------
    p_sims : ndarray
        (n_chunk,) pseudo p-values; NaN where the site has no statistic.
    rlocals : ndarray
        (n_chunk, permutations) simulated statistics if keep, with NaN
        rows where the site has no statistic; else an empty (1, 1) array.
    """
    n_chunk = observed.shape[0]
    n_permutations = permuted_ids.shape[0]
    p_sims = np.empty(n_chunk, dtype=np.float32)
    p_sims[:] = np.nan
    if keep:
        rlocals = np.empty((n_chunk, n_permutations), dtype=np.float64)
        rlocals[:] = np.nan
    else:
        rlocals = np.empty((1, 1), dtype=np.float64)

    wloc = 0
    for i in range(n_chunk):
        cardinality = cardinalities[i]
        if not active[i]:
            wloc += cardinality
            continue
        if cardinality == 0:  # no neighbours inside the component
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            # fix the self-weight to the first position
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        rstats = stat_func(site_offset + i, pool, permuted_ids, weights_i, scaling)
        p_sims[i] = _permutation_significance(
            observed[i], rstats, alternative=alternative
        ).item()
        if keep:
            rlocals[i] = rstats

    return p_sims, rlocals


def _build_component_tasks(comp_ptr, active, n_jobs):
    """
    Group component site-ranges into tasks. Components smaller than the
    target task size are batched together (many small components should
    not pay one parallel task each); larger components are split into
    site ranges of about the target size (a few big components should
    still be able to use all workers). The grouping never affects the
    results, because permutation streams are seeded per component.

    Returns a list of tasks; each task is a list of
    ``(component_id, start, end)`` ranges in component-major positions.
    """
    n_comp = comp_ptr.shape[0] - 1
    n_active = int(active.sum())
    if n_active == 0:
        return []
    n_targets = max(1, 4 * n_jobs) if n_jobs > 1 else 1
    target = max(1, -(-n_active // n_targets))  # ceil division
    tasks = []
    current = []
    current_size = 0
    for c in range(n_comp):
        s, e = int(comp_ptr[c]), int(comp_ptr[c + 1])
        if not active[s:e].any():
            continue
        size = e - s
        if size > target and n_jobs > 1:
            if current:
                tasks.append(current)
                current = []
                current_size = 0
            n_splits = -(-size // target)
            bounds = np.linspace(s, e, n_splits + 1).astype(np.int64)
            for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
                tasks.append([(c, int(b0), int(b1))])
        else:
            current.append((c, s, e))
            current_size += size
            if current_size >= target:
                tasks.append(current)
                current = []
                current_size = 0
    if current:
        tasks.append(current)
    return tasks


def _run_component_task(
    task,
    z,
    observed,
    active,
    cardinalities,
    self_weights,
    other_weights,
    w_offsets,
    comp_ptr,
    max_cards,
    scaling,
    seeds,
    permutations,
    keep,
    stat_func,
    island_weight,
    alternative,
):
    """
    Run one task (a list of component site-ranges). Regenerates each
    component's permuted ids from its seed, so ranges of one component
    processed by different tasks still use identical draws.
    """
    p_parts = []
    r_parts = []
    for c, s, e in task:
        cs, ce = comp_ptr[c], comp_ptr[c + 1]
        n_c = ce - cs
        permuted_ids = vec_permutations(max_cards[c], n_c, permutations, seeds[c])
        p_c, r_c = compute_chunk_components(
            s - cs,  # position of the range's first site inside the pool
            z[cs:ce],
            observed[s:e],
            active[s:e],
            cardinalities[s:e],
            self_weights[s:e],
            other_weights[w_offsets[s] : w_offsets[e]],
            permuted_ids,
            scaling[c],
            keep,
            stat_func,
            island_weight,
            alternative,
        )
        p_parts.append(p_c)
        r_parts.append(r_c)
    return p_parts, r_parts


def crand_components(
    z,
    observed,
    active,
    comp_ptr,
    cardinalities,
    self_weights,
    other_weights,
    scaling,
    seed,
    permutations,
    keep,
    n_jobs,
    stat_func,
    island_weight=0,
    alternative=None,
):
    """
    Conditional randomisation decomposed over connected components.

    Same protocol as ``crand``, but every site is evaluated against the
    remaining sites of its own component only. All arrays are in
    component-major order over the observed sites (see
    ``mask_unobserved``); results line up with that same order.

    Parameters
    ----------
    z : ndarray
        (n_valid,) values used both for the statistics and as the
        sampling pools (e.g. standardised values for local Moran, raw
        values for local G).
    observed : ndarray
        (n_valid,) observed statistics; NaN where a site has none.
    active : ndarray
        (n_valid,) boolean mask of sites that get a permutation. Inactive
        sites keep their value in their component's sampling pool.
    comp_ptr : ndarray
        (n_components+1,) component offsets in the packed arrays.
    cardinalities : ndarray
        (n_valid,) number of off-diagonal weights per site.
    self_weights : ndarray
        (n_valid,) diagonal (self) weights.
    other_weights : ndarray
        Flat buffer of off-diagonal weights in CSR data order.
    scaling : ndarray
        (n_components,) scaling value applied to every statistic of a
        component (e.g. (n_c - 1) / sum(z_c**2) for local Moran, the
        component sum for local G).
    seed : None/int
        Master seed. Each component draws its own seed from it through
        ``make_component_seeds``.
    permutations : int
        Number of permutations for conditional randomisation.
    keep : bool
        If True, return the simulated statistics; otherwise only the
        streaming pseudo p-values are computed and no
        (n_valid, permutations) matrix is built.
    n_jobs : int
        Number of cores to use. If -1, all available cores are used.
    stat_func : callable
        Statistic implementation with the same signature as in ``crand``.
    island_weight : float
        Weight for the "fake" neighbor of sites without neighbours.
    alternative : None | str
        Tail rule for the pseudo p-values; same options as ``crand``.

    Returns
    -------
    p_sims : ndarray
        (n_valid,) pseudo p-values; NaN where a site has no statistic.
    rlocals : ndarray
        If keep, (n_valid, permutations) simulated statistics with NaN
        rows where a site has no statistic; else an empty (1, 1) array.
    """
    n_valid = z.shape[0]
    n_comp = comp_ptr.shape[0] - 1

    if alternative is None:
        warnings.warn(
            "The alternative hypothesis for conditional randomization"
            " is changing in the next major release of esda. We recommend"
            " setting alternative='two-sided', which will generally"
            " double the p-value returned."
            " To retain the current behavior, set alternative='directed'."
            " We strongly recommend moving to alternative='two-sided'.",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: replace this with 'two-sided' by next major release
        alternative = "directed"
    if alternative not in ("two-sided", "greater", "lesser", "directed", "folded"):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not"
            " one of the supported options: 'two-sided', 'greater', "
            "'lesser', 'directed', 'folded')"
        )

    if seed is None:
        seed = np.random.randint(12345, 12345000)

    w_offsets = np.zeros(n_valid + 1, dtype=np.int64)
    if n_valid:
        np.cumsum(cardinalities, out=w_offsets[1:])

    if n_comp:
        max_cards = np.maximum.reduceat(cardinalities, comp_ptr[:-1]).astype(np.int64)
    else:
        max_cards = np.empty(0, dtype=np.int64)
    seeds = make_component_seeds(
        seed, z, comp_ptr, cardinalities, self_weights, other_weights
    )

    if n_jobs != 1 and not importlib.util.find_spec("joblib"):
        warnings.warn(
            f"Parallel processing is requested (n_jobs={n_jobs}),"
            f" but joblib cannot be imported. n_jobs will be set"
            f" to 1.",
            stacklevel=2,
        )
        n_jobs = 1
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    p_sims = np.full(n_valid, np.nan, dtype=np.float32)
    rlocals = np.full((n_valid, permutations), np.nan) if keep else np.empty((1, 1))

    tasks = _build_component_tasks(comp_ptr, active, n_jobs)
    if not tasks:
        return p_sims, rlocals

    task_args = (
        z,
        observed,
        active,
        cardinalities,
        self_weights,
        other_weights,
        w_offsets,
        comp_ptr,
        max_cards,
        scaling,
        seeds,
        permutations,
        keep,
        stat_func,
        island_weight,
        alternative,
    )

    if n_jobs == 1 or len(tasks) == 1:
        worker_out = [_run_component_task(task, *task_args) for task in tasks]
    else:
        n_jobs = min(n_jobs, len(tasks))
        from joblib import Parallel, delayed, parallel_backend

        with parallel_backend("loky", inner_max_num_threads=1):
            worker_out = Parallel(n_jobs=n_jobs)(
                delayed(_run_component_task)(task, *task_args) for task in tasks
            )

    for task, (p_parts, r_parts) in zip(tasks, worker_out, strict=True):
        for (_c, s, e), p_c, r_c in zip(task, p_parts, r_parts, strict=True):
            p_sims[s:e] = p_c
            if keep:
                rlocals[s:e] = r_c

    return p_sims, rlocals
