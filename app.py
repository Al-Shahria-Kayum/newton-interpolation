"""
app.py
------
Flask backend for Newton Interpolation Calculator.
Serves the frontend and exposes a JSON API at /calculate.
"""

from flask import Flask, request, jsonify, render_template
from interpolation import validate_input, interpolate, SAMPLE_DATASETS

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main page."""
    samples = {
        name: {
            "x":           ", ".join(str(v) for v in data["x"]),
            "y":           ", ".join(str(v) for v in data["y"]),
            "target":      data["target"],
            "description": data["description"],
        }
        for name, data in SAMPLE_DATASETS.items()
    }
    return render_template("index.html", samples=samples)


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    POST /calculate
    Body (JSON):
        x_raw    : str  — comma-separated x values
        y_raw    : str  — comma-separated y values
        target_x : str  — target x (empty string = polynomial mode)
    Returns JSON result dict or {"error": message}.
    """
    data = request.get_json(force=True)

    # ── Parse x values ──────────────────────────────
    try:
        x_parts = [p.strip() for p in data.get("x_raw", "").split(",") if p.strip()]
        if not x_parts:
            return jsonify({"error": "x values field is empty."}), 400
        x_vals = [float(p) for p in x_parts]
    except ValueError:
        return jsonify({"error": "x values contain non-numeric input. Use numbers only."}), 400

    # ── Parse y values ──────────────────────────────
    try:
        y_parts = [p.strip() for p in data.get("y_raw", "").split(",") if p.strip()]
        if not y_parts:
            return jsonify({"error": "y values field is empty."}), 400
        y_vals = [float(p) for p in y_parts]
    except ValueError:
        return jsonify({"error": "y values contain non-numeric input. Use numbers only."}), 400

    # ── Parse target x (optional) ───────────────────
    target_raw = data.get("target_x", "").strip()
    target_x   = None
    if target_raw:
        try:
            target_x = float(target_raw)
        except ValueError:
            return jsonify({"error": "Target x must be a valid number."}), 400

    # ── Validate ────────────────────────────────────
    valid, err = validate_input(x_vals, y_vals)
    if not valid:
        return jsonify({"error": err}), 400

    # ── Compute ─────────────────────────────────────
    try:
        result = interpolate(x_vals, y_vals, target_x)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Computation error: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
