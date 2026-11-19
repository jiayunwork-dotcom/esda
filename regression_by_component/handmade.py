"""Handmade small graph used by the by-component regression checks.

The graph mimics two river segments joined by a single bridge node::

    0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8        (4 is the bridge)
                    9 - 10                   (degenerate pair)
                    11                       (isolated node)
                    12 - 13 - 14             (constant component)

When the bridge node 4 has a missing reading, the observed graph splits
into the components {0,1,2,3} and {5,6,7,8} plus the degenerate pieces.
"""

import warnings

import numpy as np
from libpysal.graph import Graph
from libpysal.weights import WSP
from scipy import sparse

N = 15

# undirected edges of the full monitoring network
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # bridge node 4
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (12, 13),
    (13, 14),
]

Y_FULL = np.array(
    [10.0, 12.0, 11.0, 13.0, 9.0, 3.0, 4.0, 2.0, 5.0, 7.0, 8.0, 6.0, 20.0, 20.0, 20.0]
)

# positions that must carry missing results once the bridge goes missing:
# the bridge itself (NaN), the size-2 component, the isolated node and the
# constant component
MISSING_AT = [4, 9, 10, 11, 12, 13, 14]
COMP_A = [0, 1, 2, 3]
COMP_B = [5, 6, 7, 8]


def y_with_missing_bridge():
    y = Y_FULL.copy()
    y[4] = np.nan
    return y


def adjacency(directed=False, self_loops=(), n=N):
    """Adjacency of the handmade graph as a csr matrix.

    directed=True keeps every edge one-way (i -> j as listed in EDGES).
    self_loops adds explicit diagonal entries.
    """
    a = np.zeros((n, n))
    for i, j in EDGES:
        a[i, j] = 1.0
        if not directed:
            a[j, i] = 1.0
    for i in self_loops:
        a[i, i] = 1.0
    return sparse.csr_matrix(a)


def make_w(adj):
    """Build a libpysal W from an adjacency matrix (silencing the
    disconnected-components warning that the handmade graph triggers)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WSP(adj.tocsr()).to_W()


def make_graph(adj):
    """Build a libpysal Graph from an adjacency matrix."""
    return Graph.from_sparse(adj.tocsr())


def permute_table(y, adj, order):
    """Re-order the monitoring table: y and adjacency permuted by `order`."""
    order = np.asarray(order)
    y_new = y[order]
    adj_new = adj[order][:, order].tocsr()
    return y_new, adj_new
