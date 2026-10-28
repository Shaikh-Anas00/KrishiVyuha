"""
Multi-season, rotation-aware portfolio optimizer.

Extends the single-season Markowitz model (portfolio_optimizer.py) to T seasons.
What is grown on a plot in season t changes the *yield* of whatever is grown on
that plot in season t+1 through a crop-interaction matrix B[prev][next]
(N-fixation bonus after legumes, disease/pest carry-over inside a family,
soil-depletion for heavy feeders, monoculture penalty for the same crop twice).

Plot-level (flow) formulation
-----------------------------
Decision variable per season: X_t[i, j] >= 0 = fraction of the farm that grew
crop i last season and grows crop j this season (extra row n = "no history").
    row sums   : sum_j X_t[i, j] = land share that grew i last season
    allocation : w_t[j]          = sum_i X_t[i, j]
    eff. area  : y_t[j]          = sum_i X_t[i, j] * (1 + B[i, j])
    E[profit]  = A * (y.E - w.PREM)       Var = A^2 * y' S y
    worst case = A * (y.WC + w.(PAY - PREM))     (must be >= survival budget, every season)
Maximise  sum_t gamma^(t-1) * [ E[profit_t] - 0.5*lambda*(Var_t/ref_var)*ref_ret ].

Using flows (not an area-averaged multiplier) lets the optimiser decide *where*
to put the legume (on the plots that just grew cotton), which is the actual
rotation decision.

HONESTY NOTE: B is an agronomic prior built from crop-family rules of thumb,
NOT fitted from the panel dataset (it has no field-level crop history).
Magnitudes are conservative and fully exposed so they can be replaced by
locally measured values. Seasons are treated as independent weather draws and
the same climate inputs are used in every season.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from src.portfolio_optimizer import (
    compute_crop_financials, get_pmfby_terms, get_confidence_margin,
    _get_risk_profiles, auto_detect_climate_and_weather,
)

FAMILY = {
    "Rice": "cereal", "Wheat": "cereal", "Maize": "cereal", "Jowar": "cereal", "Bajra": "cereal",
    "Gram": "legume", "Arhar/Tur": "legume", "Urad": "legume", "Moong(Green Gram)": "legume",
    "Soyabean": "legume", "Groundnut": "legume",
    "Mustard": "oilseed", "Sunflower": "oilseed",
    "Cotton": "fibre", "Sugarcane": "sugar",
    "Potato": "veg", "Onion": "veg", "Tomato": "veg",
    "Banana": "perennial", "Coconut": "perennial",
}

# Fractional yield effect of growing <next family> on a plot that grew <prev family>.
FAMILY_EFFECT = {
    ("legume", "cereal"): +0.12, ("legume", "fibre"): +0.10, ("legume", "oilseed"): +0.07,
    ("legume", "sugar"): +0.08, ("legume", "veg"): +0.08, ("legume", "legume"): -0.10,
    ("cereal", "legume"): +0.05, ("cereal", "cereal"): -0.04, ("cereal", "oilseed"): +0.02,
    ("fibre", "legume"): +0.06, ("fibre", "cereal"): +0.02, ("fibre", "fibre"): -0.12,
    ("oilseed", "cereal"): +0.03, ("oilseed", "legume"): +0.02, ("oilseed", "oilseed"): -0.08,
    ("sugar", "legume"): +0.02, ("sugar", "cereal"): -0.03, ("sugar", "fibre"): -0.03,
    ("sugar", "sugar"): -0.10,
    ("veg", "cereal"): +0.02, ("veg", "veg"): -0.10,
}
SAME_CROP_EXTRA = -0.06     # additional monoculture penalty (same species twice)


def build_interaction_matrix(crops: list) -> np.ndarray:
    """B[i, j] = fractional yield effect on crop j of the plot having grown crop i last season."""
    n = len(crops)
    B = np.zeros((n, n))
    for i, a in enumerate(crops):
        fa = FAMILY.get(a, "other")
        for j, b in enumerate(crops):
            fb = FAMILY.get(b, "other")
            if fa == "perennial" or fb == "perennial":
                continue
            v = FAMILY_EFFECT.get((fa, fb), 0.0)
            if a == b:
                v += SAME_CROP_EXTRA
            B[i, j] = v
    return B


class _Ctx:
    pass


def _build_context(state, district, season, crops, area, budget, rainfall, temperature,
                   regime, irrigation, lam, gamma, B):
    cm = get_confidence_margin()
    c = _Ctx()
    c.crops, c.n, c.area, c.budget, c.lam, c.gamma = crops, len(crops), area, budget, lam, gamma
    c.B1 = np.vstack([1.0 + B, np.ones((1, c.n))])        # (n+1, n); last row = no history
    c.fin = {k: compute_crop_financials(k, state, district, season, rainfall, temperature, regime,
                                        area_ha=1.0, confidence_margin=cm, irrigation_type=irrigation)
             for k in crops}
    c.E = np.array([c.fin[k]["expected_revenue"] for k in crops])
    c.WC = np.array([c.fin[k]["worst_case_revenue"] for k in crops])
    c.PAY = np.array([c.fin[k]["pmfby_payout"] for k in crops])
    c.PREM = np.array([c.fin[k]["premium_cost"] for k in crops])
    alpha = np.array([c.fin[k]["agronomic_multiplier"] for k in crops])

    rp = _get_risk_profiles()
    cov = rp.get("covariance_matrices", {}).get(regime) or rp.get("covariance_matrices", {}).get("Overall", {})
    kappa = np.array([float(rp.get("agmarknet_price_risk", {}).get(k, {}).get("revenue_vol_multiplier", 1.0))
                      for k in crops])
    S = np.zeros((c.n, c.n))
    for i, a in enumerate(crops):
        for j, b in enumerate(crops):
            v = cov.get(a, {}).get(b)
            if v is None:
                v = cov.get(b, {}).get(a)
            if v is None:
                v = (1.0 if i == j else 0.25) * 0.16 * c.E[i] * c.E[j]
            S[i, j] = v * kappa[i] * kappa[j] / np.sqrt(max(0.05, alpha[i] * alpha[j]))
    S += np.eye(c.n) * max(100.0, float(np.mean(np.diag(S)))) * 0.01
    c.S = S
    c.ref_var = float(np.mean(np.diag(S))) * area ** 2
    c.ref_ret = max(1000.0, float(np.mean(c.E)) * area)
    return c


def _terms(c, X):
    w = X.sum(axis=0)
    y = (X * c.B1).sum(axis=0)
    ep = float(c.area * (y @ c.E - w @ c.PREM))
    var = float(y @ c.S @ y) * c.area ** 2
    worst = float(c.area * (y @ c.WC + w @ (c.PAY - c.PREM)))
    return w, ep, var, worst


def _season_utility(c, X):
    _, ep, var, _ = _terms(c, X)
    return ep - 0.5 * c.lam * (var / c.ref_var) * c.ref_ret


def _prev_vec(w):
    return np.append(w, 0.0)


def _solve(c, T, hist, starts):
    n, R = c.n, c.n + 1
    size = R * n

    def unpack(x):
        return [x[t * size:(t + 1) * size].reshape(R, n) for t in range(T)]

    def neg_util(x):
        return -sum((c.gamma ** t) * _season_utility(c, X) for t, X in enumerate(unpack(x))) / (c.ref_ret * T)

    def row_res(x, t):
        Xs = unpack(x)
        prev = hist if t == 0 else _prev_vec(Xs[t - 1].sum(axis=0))
        return Xs[t].sum(axis=1) - prev

    def wc_res(x, t):
        return (_terms(c, unpack(x)[t])[3] - c.budget) / max(1000.0, c.budget)

    cons = []
    for t in range(T):
        cons.append({"type": "eq", "fun": (lambda x, t=t: row_res(x, t))})
        cons.append({"type": "ineq", "fun": (lambda x, t=t: wc_res(x, t))})

    best = None
    for x0 in starts:
        r = minimize(neg_util, x0, method="SLSQP", bounds=[(0, 1)] * (T * size), constraints=cons,
                     options={"ftol": 1e-10, "maxiter": 800})
        x = np.clip(r.x, 0, 1)
        ok = all(np.max(np.abs(row_res(x, t))) < 2e-3 and wc_res(x, t) >= -1e-3 for t in range(T))
        sc = (0 if ok else 1, neg_util(x))
        if best is None or sc < best[0]:
            best = (sc, unpack(x), ok)
    return best[1], best[2]


def _start_from_alloc(hist, allocs):
    out, prev = [], hist
    for w in allocs:
        out.append(np.outer(prev, w).ravel())
        prev = _prev_vec(w)
    return np.concatenate(out)


def _static_repeat_plan(c, w, T, hist):
    """Single-season optimum repeated every season on the *same plots* (what a static model implies)."""
    R, n = c.n + 1, c.n
    Xs = [np.outer(hist, w)]
    for _ in range(1, T):
        X = np.zeros((R, n))
        for j in range(n):
            X[j, j] = w[j]
        Xs.append(X)
    return Xs


def _evaluate(c, Xs):
    rows, disc, util = [], 0.0, 0.0
    for t, X in enumerate(Xs):
        w, ep, var, worst = _terms(c, X)
        y = (X * c.B1).sum(axis=0)
        flows = sorted(
            [(c.crops[i] if i < c.n else "(no history)", c.crops[j], float(X[i, j]))
             for i in range(c.n + 1) for j in range(c.n) if X[i, j] > 0.02],
            key=lambda f: -f[2])[:6]
        rows.append({
            "season_index": t + 1,
            "weights": {k: round(float(w[j]), 4) for j, k in enumerate(c.crops)},
            "yield_multiplier": {k: round(float(y[j] / w[j]) if w[j] > 1e-6 else 1.0, 3)
                                 for j, k in enumerate(c.crops)},
            "plot_transitions": [{"from": a, "to": b, "share": round(s, 3)} for a, b, s in flows],
            "expected_profit": round(ep, 2),
            "worst_case_income": round(worst, 2),
            "volatility": round(float(np.sqrt(var)), 2),
        })
        disc += (c.gamma ** t) * ep
        util += (c.gamma ** t) * _season_utility(c, X)
    return rows, disc, util


def optimize_rotation(
    state: str, district: str, season: str, crops: list,
    total_area_ha: float, survival_budget: float,
    n_seasons: int = 3,
    previous_weights: Optional[dict] = None,   # what was grown last season, e.g. {"Cotton": 1.0}
    rainfall: Optional[float] = None, temperature: Optional[float] = None,
    climate_regime: Optional[str] = None,
    irrigation_type: str = "Rainfed", risk_aversion: float = 1.5,
    discount: float = 0.95, seed: int = 0,
) -> dict:
    if rainfall is None or temperature is None or climate_regime is None:
        try:
            d = auto_detect_climate_and_weather(district, season, state)
            rainfall = d["rainfall_mm"] if rainfall is None else rainfall
            temperature = d["temperature_c"] if temperature is None else temperature
            climate_regime = d["climate_regime"] if climate_regime is None else climate_regime
        except Exception:
            rainfall = 600.0 if rainfall is None else rainfall
            temperature = 27.0 if temperature is None else temperature
            climate_regime = climate_regime or "Normal"

    n, T = len(crops), int(n_seasons)
    B = build_interaction_matrix(crops)
    c = _build_context(state, district, season, crops, total_area_ha, survival_budget,
                       rainfall, temperature, climate_regime, irrigation_type,
                       risk_aversion, discount, B)

    hist = np.zeros(n + 1)
    if previous_weights:
        for k, v in previous_weights.items():
            if k in crops:
                hist[crops.index(k)] = float(v)
    if hist.sum() <= 0:
        hist[n] = 1.0                      # no history -> no rotation effect in season 1
    hist = hist / hist.sum()

    rng = np.random.default_rng(seed)
    uniform = np.full(n, 1.0 / n)

    # 1) Static model: season-1 optimum (T=1), then repeated on the same plots
    s1, _ = _solve(c, 1, hist, [_start_from_alloc(hist, [uniform])] +
                   [_start_from_alloc(hist, [rng.dirichlet(np.ones(n))]) for _ in range(3)])
    w_static = s1[0].sum(axis=0)
    static = _static_repeat_plan(c, w_static, T, hist)

    # 2) Greedy: re-optimise each season myopically given last season's plots
    greedy, prev = [], hist
    for t in range(T):
        Xs, _ = _solve(c, 1, prev, [_start_from_alloc(prev, [uniform])] +
                       [_start_from_alloc(prev, [rng.dirichlet(np.ones(n))]) for _ in range(2)])
        greedy.append(Xs[0])
        prev = _prev_vec(Xs[0].sum(axis=0))

    # 3) Rotation-aware joint plan (multi-start, warm-started from greedy/static)
    starts = [np.concatenate([X.ravel() for X in greedy]),
              np.concatenate([X.ravel() for X in static]),
              _start_from_alloc(hist, [uniform] * T)]
    starts += [_start_from_alloc(hist, [rng.dirichlet(np.ones(n)) for _ in range(T)]) for _ in range(4)]
    plan, feasible = _solve(c, T, hist, starts)

    plan_rows, plan_val, plan_u = _evaluate(c, plan)
    greedy_rows, greedy_val, greedy_u = _evaluate(c, greedy)
    static_rows, static_val, static_u = _evaluate(c, static)

    def legume(rows):
        return [round(sum(v for k, v in r["weights"].items() if FAMILY.get(k) == "legume"), 4) for r in rows]

    return {
        "n_seasons": T,
        "feasible": bool(feasible),
        "risk_aversion": risk_aversion,
        "discount": discount,
        "climate": {"regime": climate_regime, "rainfall_mm": rainfall, "temperature_c": temperature},
        "interaction_matrix": {a: {b: round(float(B[i, j]), 3) for j, b in enumerate(crops)}
                               for i, a in enumerate(crops)},
        "crop_family": {k: FAMILY.get(k, "other") for k in crops},
        "rotation_plan": plan_rows,
        "greedy_plan": greedy_rows,
        "static_plan": static_rows,
        "summary": {
            "rotation_discounted_profit": round(plan_val, 2),
            "greedy_discounted_profit": round(greedy_val, 2),
            "static_discounted_profit": round(static_val, 2),
            "gain_vs_static_inr": round(plan_val - static_val, 2),
            "gain_vs_static_pct": round(100 * (plan_val - static_val) / max(1.0, abs(static_val)), 2),
            "gain_vs_greedy_inr": round(plan_val - greedy_val, 2),
            "gain_vs_greedy_pct": round(100 * (plan_val - greedy_val) / max(1.0, abs(greedy_val)), 2),
            "rotation_utility": round(plan_u, 2),
            "greedy_utility": round(greedy_u, 2),
            "static_utility": round(static_u, 2),
            "legume_share_rotation": legume(plan_rows),
            "legume_share_greedy": legume(greedy_rows),
            "legume_share_static": legume(static_rows),
        },
        "assumption_note": ("Interaction matrix is an agronomic prior (crop-family rules), not fitted "
                            "from the dataset; seasons are independent weather draws and the same "
                            "climate inputs are used in every season."),
    }
