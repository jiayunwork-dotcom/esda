"""Generate the reference results of the *old* (whole-graph) entry points.

Run this with an unmodified esda to (re)create ``reference_default_mode.npz``:

    python regression_by_component/make_reference.py

The regression check later recomputes these same calls and requires them to
match, guarding that the default whole-graph behaviour is unchanged.
"""

from pathlib import Path

import numpy as np
from handmade import COMP_A, Y_FULL, adjacency, make_graph, make_w

from esda import G_Local, Moran_Local

HERE = Path(__file__).parent
SEED = 12345
PERMS = 99


def collect(prefix, obj, attrs):
    return {
        f"{prefix}.{name}": np.asarray(getattr(obj, name), dtype=np.float64)
        for name in attrs
    }


def main():
    adj = adjacency()
    w = make_w(adj)
    g = make_graph(adj)
    y = Y_FULL

    out = {}
    moran_attrs = ["z", "Is", "q", "p_sim", "EI", "VI", "EIc", "VIc", "EI_sim", "z_sim"]
    g_attrs = ["Gs", "EGs", "VGs", "Zs", "p_sim", "EG_sim", "z_sim"]

    lm_w = Moran_Local(
        y, w, transformation="r", permutations=PERMS, seed=SEED, alternative="two-sided"
    )
    out.update(collect("moran_w", lm_w, moran_attrs))

    lm_g = Moran_Local(
        y, g, transformation="r", permutations=PERMS, seed=SEED, alternative="two-sided"
    )
    out.update(collect("moran_g", lm_g, moran_attrs))

    lm_par = Moran_Local(
        y,
        make_w(adj),
        transformation="r",
        permutations=PERMS,
        seed=SEED,
        n_jobs=2,
        alternative="two-sided",
    )
    out.update(collect("moran_par", lm_par, ["p_sim"]))

    lg_w = G_Local(
        y, w, transform="R", permutations=PERMS, seed=SEED, alternative="two-sided"
    )
    out.update(collect("g_w", lg_w, g_attrs))

    lg_star = G_Local(
        y, w, transform="R", star=True, permutations=PERMS, seed=SEED,
        alternative="two-sided",
    )
    out.update(collect("g_star_w", lg_star, g_attrs))

    lg_g = G_Local(
        y, g, transform="B", permutations=PERMS, seed=SEED, alternative="two-sided"
    )
    out.update(collect("g_g", lg_g, g_attrs))

    # component-A-only default-mode runs: the by-component statistics must
    # reproduce these numbers exactly (permutation streams differ by design)
    adj_a = adjacency()[COMP_A][:, COMP_A]
    y_a = Y_FULL[COMP_A]
    lm_a = Moran_Local(
        y_a,
        make_w(adj_a),
        transformation="r",
        permutations=PERMS,
        seed=SEED,
        alternative="two-sided",
    )
    out.update(collect("moran_comp_a", lm_a, moran_attrs))
    lg_a = G_Local(
        y_a,
        make_w(adj_a),
        transform="R",
        permutations=PERMS,
        seed=SEED,
        alternative="two-sided",
    )
    out.update(collect("g_comp_a", lg_a, g_attrs))

    path = HERE / "reference_default_mode.npz"
    np.savez(path, **out)
    print(f"wrote {path} with {len(out)} arrays")


if __name__ == "__main__":
    main()
