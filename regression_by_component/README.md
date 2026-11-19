# Regression checks for the by-component mode of `Moran_Local` / `G_Local`

Handmade-graph regression for the optional connected-component reading of the
local statistics (`by_component=True`). Python 3.12.

## Rerun

From the repository root, with the project environment:

```sh
python regression_by_component/check_by_component.py
```

Exit status is 0 only if every check passes. The run takes about a minute
(mostly numba JIT and one parallel dispatch).

## Layout

- `handmade.py` — the handmade graph: two river segments joined by one bridge
  node, plus a degenerate pair, an isolated node and a constant component.
  Setting the bridge reading to NaN splits the two segments into separate
  components.
- `check_by_component.py` — all checks (see below).
- `make_reference.py` — regenerates `reference_default_mode.npz`, the
  reference outputs of the *old* whole-graph entry points
  (`Moran_Local` / `G_Local` with `by_component` left at its default).
  It was produced with an unmodified esda; the check recomputes the same
  calls and requires them to match, guarding the default behaviour.
- `reference_default_mode.npz` — the stored reference results.

To regenerate the reference (only when the old entry points intentionally
change):

```sh
python regression_by_component/make_reference.py
# or
python regression_by_component/check_by_component.py --rewrite-reference
```

## What is checked

1. Old whole-graph entry points reproduce the stored reference exactly
   (W and Graph, serial and `n_jobs=2`, Gi and Gi*).
2. A missing bridge node splits the two river segments; outputs come back at
   the original site positions with NaN at the missing bridge, the size-2
   component, the isolated node and the constant component.
3. Per-component statistics (standardisation, `Is`, quadrants, analytical
   moments; denominators, `Gs`, analytical moments) reproduce the old
   estimator run on the extracted component exactly.
4. Extracting one component and recomputing with the same mode and seed
   reproduces its observed statistics, every simulation round and `p_sim`.
5. Moving the component's rows between other components' rows, changing the
   other component's readings, or changing `n_jobs` leaves its results
   bitwise identical.
6. W and Graph inputs, one-way directed edges, row standardisation on the
   retained edges, Gi/Gi* (`star` False/True/None), self-loops, a single
   zero-denominator cell (whose value still feeds the other draws), the
   all-missing table, both `keep_simulations` modes, and input
   immutability.
