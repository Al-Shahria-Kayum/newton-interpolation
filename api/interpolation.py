"""
interpolation.py
----------------
Smart Newton Interpolation System
Supports:
  - Newton Forward Interpolation  (Δ table)
  - Newton Backward Interpolation (∇ table)
  - Auto method selection based on target x proximity
  - Polynomial expression generation when no target x given
"""

import math
from typing import Optional


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_input(x_values: list, y_values: list) -> tuple:
    """
    Full validation. Returns (True, "") or (False, error_message).
    """
    if len(x_values) < 2:
        return False, "At least 2 data points are required."

    if len(x_values) != len(y_values):
        return False, (
            f"Mismatch: {len(x_values)} x-values but {len(y_values)} y-values. "
            "Both lists must have equal length."
        )

    for i, v in enumerate(x_values):
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            return False, f"x[{i}] = '{v}' is not a valid finite number."
    for i, v in enumerate(y_values):
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            return False, f"y[{i}] = '{v}' is not a valid finite number."

    if len(set(x_values)) != len(x_values):
        return False, "Duplicate x-values detected. All x-values must be unique."

    ok, msg = check_equal_interval(x_values)
    if not ok:
        return False, msg

    return True, ""


def check_equal_interval(x_values: list, tol: float = 1e-9) -> tuple:
    """Verifies equally spaced x values."""
    sx = sorted(x_values)
    h  = sx[1] - sx[0]
    if abs(h) < tol:
        return False, "Step size h ≈ 0. x-values must be distinct."
    for i in range(1, len(sx)):
        diff = sx[i] - sx[i - 1]
        if abs(diff - h) > tol:
            return False, (
                f"x-values are NOT equally spaced. "
                f"Expected h = {h}, found gap {diff:.6g} "
                f"between x[{i-1}] = {sx[i-1]} and x[{i}] = {sx[i]}."
            )
    return True, ""


# ─────────────────────────────────────────────────────────────
# AUTO METHOD SELECTION
# ─────────────────────────────────────────────────────────────

def auto_select_method(x_values: list, target_x: float) -> dict:
    """
    Selects Forward or Backward interpolation based on target proximity.
    """
    x0, xn       = x_values[0], x_values[-1]
    dist_start   = abs(target_x - x0)
    dist_end     = abs(target_x - xn)

    if dist_start <= dist_end:
        method = "forward"
        reason = (
            f"Target x = {target_x} is closer to the beginning of the table "
            f"(x₀ = {x0}, distance = {dist_start:.4g}) "
            f"than to the end (xₙ = {xn}, distance = {dist_end:.4g}). "
            "Newton Forward Interpolation selected."
        )
    else:
        method = "backward"
        reason = (
            f"Target x = {target_x} is closer to the end of the table "
            f"(xₙ = {xn}, distance = {dist_end:.4g}) "
            f"than to the beginning (x₀ = {x0}, distance = {dist_start:.4g}). "
            "Newton Backward Interpolation selected."
        )

    warning = ""
    if target_x < x0 or target_x > xn:
        warning = (
            f"⚠ Target x = {target_x} is OUTSIDE the data range [{x0}, {xn}]. "
            "This is extrapolation — accuracy may be reduced."
        )

    return {
        "method":     method,
        "reason":     reason,
        "warning":    warning,
        "dist_start": dist_start,
        "dist_end":   dist_end,
    }


# ─────────────────────────────────────────────────────────────
# DIFFERENCE TABLES
# ─────────────────────────────────────────────────────────────

def build_forward_difference_table(y_values: list) -> list:
    """Builds Newton Forward Difference Table (Δ)."""
    n     = len(y_values)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = float(y_values[i])
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table


def build_backward_difference_table(y_values: list) -> list:
    """Builds Newton Backward Difference Table (∇)."""
    n     = len(y_values)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = float(y_values[i])
    for j in range(1, n):
        for i in range(j, n):
            table[i][j] = table[i][j - 1] - table[i - 1][j - 1]
    return table


# ─────────────────────────────────────────────────────────────
# NEWTON FORWARD INTERPOLATION
# ─────────────────────────────────────────────────────────────

def newton_forward_interpolation(sorted_x, sorted_y, target_x, fwd_table) -> dict:
    """
    y(x) = y₀ + u·Δy₀ + [u(u-1)/2!]·Δ²y₀ + ...
    u = (x - x₀) / h
    """
    n       = len(sorted_x)
    x0      = sorted_x[0]
    h       = sorted_x[1] - sorted_x[0]
    u       = (target_x - x0) / h
    delta   = [fwd_table[0][k] for k in range(n)]

    result, steps, u_prod = 0.0, [], 1.0

    for k in range(n):
        if k == 0:
            coeff, term = 1.0, delta[0]
        else:
            u_prod *= (u - (k - 1))
            coeff   = u_prod / math.factorial(k)
            term    = coeff * delta[k]
        result += term
        steps.append({
            "order":      k,
            "symbol":     _sym_fwd(k),
            "delta_val":  delta[k],
            "coeff":      coeff,
            "term_val":   term,
            "cumulative": result,
        })

    return {
        "result":    result,
        "h": h, "u": u, "x0": x0, "y0": sorted_y[0],
        "delta":     delta,
        "steps":     steps,
        "var_label": "u",
        "var_formula": f"u = (x − x₀) / h = ({target_x} − {x0}) / {h} = {u:.8f}",
    }


# ─────────────────────────────────────────────────────────────
# NEWTON BACKWARD INTERPOLATION
# ─────────────────────────────────────────────────────────────

def newton_backward_interpolation(sorted_x, sorted_y, target_x, bwd_table) -> dict:
    """
    y(x) = yₙ + s·∇yₙ + [s(s+1)/2!]·∇²yₙ + ...
    s = (x - xₙ) / h
    """
    n     = len(sorted_x)
    xn    = sorted_x[-1]
    h     = sorted_x[1] - sorted_x[0]
    s     = (target_x - xn) / h
    nabla = [bwd_table[n - 1][k] for k in range(n)]

    result, steps, s_prod = 0.0, [], 1.0

    for k in range(n):
        if k == 0:
            coeff, term = 1.0, nabla[0]
        else:
            s_prod *= (s + (k - 1))
            coeff   = s_prod / math.factorial(k)
            term    = coeff * nabla[k]
        result += term
        steps.append({
            "order":      k,
            "symbol":     _sym_bwd(k),
            "delta_val":  nabla[k],
            "coeff":      coeff,
            "term_val":   term,
            "cumulative": result,
        })

    return {
        "result":    result,
        "h": h, "s": s, "xn": xn, "yn": sorted_y[-1],
        "nabla":     nabla,
        "steps":     steps,
        "var_label": "s",
        "var_formula": f"s = (x − xₙ) / h = ({target_x} − {xn}) / {h} = {s:.8f}",
    }


# ─────────────────────────────────────────────────────────────
# POLYNOMIAL EXPRESSION GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_polynomial_expression(sorted_x, sorted_y, fwd_table, bwd_table) -> dict:
    """
    Generates Newton polynomial P(x) in both Forward and Backward forms.
    Returns the factored form AND the final expanded numerical polynomial.
    Called when no target x is provided.
    """
    import math as _math
    n     = len(sorted_x)
    x0    = sorted_x[0]
    xn    = sorted_x[-1]
    h     = sorted_x[1] - sorted_x[0]
    delta = [fwd_table[0][k] for k in range(n)]
    nabla = [bwd_table[n - 1][k] for k in range(n)]

    def build_factored(origin_label, diffs, sym, var):
        """Build factored form: P(x) = d0 + d1*(x-x0)/h + ..."""
        terms = []
        for k in range(n):
            dv = diffs[k]
            if abs(dv) < 1e-12:
                continue
            dv_s = _fmt(dv)
            if k == 0:
                terms.append(dv_s)
            else:
                parts = []
                for j in range(k):
                    shift = j * h
                    if sym == "nabla":
                        node = xn - shift
                        parts.append(f"({var} - {_fmt(node)})")
                    else:
                        node = x0 + shift
                        parts.append(f"({var} - {_fmt(node)})")
                denom = _math.factorial(k) * (h ** k)
                num   = " * ".join(parts)
                coeff_str = f"{dv_s} * [{num}]" if abs(dv - 1.0) > 1e-9 else f"[{num}]"
                terms.append(f"{coeff_str} / {_fmt(denom)}")
        return "P(x) = " + "\n      + ".join(terms) if terms else "P(x) = 0"

    def build_final_poly(diffs, sym):
        """
        Build final evaluated polynomial with actual numerical coefficients.
        Expands each binomial coefficient term and collects by power of x.
        Returns a readable string like:
        P(x) = a0 + a1*x + a2*x^2 + ...
        """
        # Collect polynomial coefficients using Newton basis expansion
        # We accumulate the polynomial as a list of coefficients [a0, a1, a2, ...]
        coeffs = [0.0] * n

        if sym == "delta":
            # Forward: basis is (x-x0)(x-x0-h)...(x-x0-(k-1)h) / (k! * h^k)
            for k in range(n):
                dv = diffs[k]
                if abs(dv) < 1e-12:
                    continue
                # Build polynomial (x-x0)(x-x0-h)...(x-x0-(k-1)h)
                # Start with [1], multiply each (x - node)
                basis = [1.0]
                for j in range(k):
                    node = x0 + j * h
                    new_basis = [0.0] * (len(basis) + 1)
                    for i, c in enumerate(basis):
                        new_basis[i + 1] += c      # x * c
                        new_basis[i]     -= node * c  # -node * c
                    basis = new_basis
                denom = _math.factorial(k) * (h ** k)
                scale = dv / denom
                for i, c in enumerate(basis):
                    if i < n:
                        coeffs[i] += scale * c
        else:
            # Backward: basis is (x-xn)(x-xn+h)...(x-xn+(k-1)h)
            for k in range(n):
                dv = diffs[k]
                if abs(dv) < 1e-12:
                    continue
                basis = [1.0]
                for j in range(k):
                    node = xn - j * h
                    new_basis = [0.0] * (len(basis) + 1)
                    for i, c in enumerate(basis):
                        new_basis[i + 1] += c
                        new_basis[i]     -= node * c
                    basis = new_basis
                denom = _math.factorial(k) * (h ** k)
                scale = dv / denom
                for i, c in enumerate(basis):
                    if i < n:
                        coeffs[i] += scale * c

        # Build readable string
        parts = []
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            c_s = _fmt(round(c, 8))
            if i == 0:
                parts.append(c_s)
            elif i == 1:
                parts.append(f"({c_s})x")
            else:
                parts.append(f"({c_s})x^{i}")
        return "P(x) = " + " + ".join(parts) if parts else "P(x) = 0"

    fwd_factored = build_factored(x0, delta, "delta", "x")
    bwd_factored = build_factored(xn, nabla, "nabla", "x")
    fwd_final    = build_final_poly(delta, "delta")
    bwd_final    = build_final_poly(nabla, "nabla")

    return {
        "forward_polynomial":  fwd_factored,
        "backward_polynomial": bwd_factored,
        "forward_final":       fwd_final,
        "backward_final":      bwd_final,
        "delta": delta,
        "nabla": nabla,
        "x0": x0, "xn": xn, "h": h,
        "note": (
            "No target x was provided. "
            "Factored form: original Newton formula with actual values substituted. "
            "Final form: fully expanded polynomial P(x) with numerical coefficients. "
            "Substitute any x value to compute the interpolated result."
        )
    }


# ─────────────────────────────────────────────────────────────
# MASTER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def interpolate(x_values: list, y_values: list, target_x=None) -> dict:
    """
    Master function called by the API.
    Validates → sorts → builds tables → selects method → computes result.
    """
    paired   = sorted(zip(x_values, y_values), key=lambda p: p[0])
    sorted_x = [p[0] for p in paired]
    sorted_y = [p[1] for p in paired]
    n        = len(sorted_x)
    h        = sorted_x[1] - sorted_x[0]

    fwd_table = build_forward_difference_table(sorted_y)
    bwd_table = build_backward_difference_table(sorted_y)

    base = {
        "n":          n,
        "h":          h,
        "sorted_x":   sorted_x,
        "sorted_y":   sorted_y,
        "fwd_table":  _ser(fwd_table, n, "forward"),
        "bwd_table":  _ser(bwd_table, n, "backward"),
        "has_target": target_x is not None,
        "target_x":   target_x,
    }

    if target_x is None:
        poly = generate_polynomial_expression(sorted_x, sorted_y, fwd_table, bwd_table)
        base.update({"mode": "polynomial", "polynomial": poly})
        return base

    sel = auto_select_method(sorted_x, target_x)
    base["selection"] = sel

    if sel["method"] == "forward":
        calc = newton_forward_interpolation(sorted_x, sorted_y, target_x, fwd_table)
        base.update({"mode": "forward", "result": calc["result"], "calc": calc})
    else:
        calc = newton_backward_interpolation(sorted_x, sorted_y, target_x, bwd_table)
        base.update({"mode": "backward", "result": calc["result"], "calc": calc})

    return base


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if v != v:
        return "—"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _sym_fwd(k: int) -> str:
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "y₀" if k == 0 else f"Δ{str(k).translate(sup)}y₀"


def _sym_bwd(k: int) -> str:
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "yₙ" if k == 0 else f"∇{str(k).translate(sup)}yₙ"


def _ser(table, n, direction='forward'):
    """
    Serialize difference table to JSON-safe list.
    Forward  table: None when i + j >= n
    Backward table: None when j > i
    """
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            is_none = (i + j >= n) if direction == 'forward' else (j > i)
            row.append(None if is_none else round(table[i][j], 10))
        out.append(row)
    return out


# ─────────────────────────────────────────────────────────────
# SAMPLE DATASETS
# ─────────────────────────────────────────────────────────────

SAMPLE_DATASETS = {
    "Natural Logarithm": {
        "x": [1.0, 1.25, 1.50, 1.75, 2.00],
        "y": [0.0000, 0.2231, 0.4055, 0.5596, 0.6931],
        "target": 1.35,
        "description": "Approximate ln(1.35) — target near start → Forward"
    },
    "Sine Function": {
        "x": [0, 10, 20, 30, 40, 50],
        "y": [0.0000, 0.1736, 0.3420, 0.5000, 0.6428, 0.7660],
        "target": 45,
        "description": "Approximate sin(45°) — target near end → Backward"
    },
    "Cube Function": {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 8, 27, 64, 125],
        "target": 2.5,
        "description": "Approximate f(2.5) for f(x)=x³ → Forward"
    },
    "Square Root Approx": {
        "x": [1.0, 1.5, 2.0, 2.5, 3.0],
        "y": [1.0000, 1.2247, 1.4142, 1.5811, 1.7321],
        "target": 2.8,
        "description": "Approximate √2.8 — target near end → Backward"
    },
    "Polynomial Only": {
        "x": [0, 1, 2, 3],
        "y": [1, 2, 5, 10],
        "target": None,
        "description": "No target x — generates polynomial P(x) expression"
    },
}
