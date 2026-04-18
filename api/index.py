"""
api/index.py
------------
Flask app for Vercel deployment.
Vercel requires the app to be in api/index.py
and the Flask instance must be named 'app'.
"""

import sys
import os

# Make sure interpolation.py is importable from same folder
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from interpolation import validate_input, interpolate, SAMPLE_DATASETS

# Tell Flask where templates folder is (inside api/)
app = Flask(__name__, template_folder="templates")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
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
    data = request.get_json(force=True)

    # Parse x
    try:
        x_parts = [p.strip() for p in data.get("x_raw", "").split(",") if p.strip()]
        if not x_parts:
            return jsonify({"error": "x values field is empty."}), 400
        x_vals = [float(p) for p in x_parts]
    except ValueError:
        return jsonify({"error": "x values contain non-numeric input."}), 400

    # Parse y
    try:
        y_parts = [p.strip() for p in data.get("y_raw", "").split(",") if p.strip()]
        if not y_parts:
            return jsonify({"error": "y values field is empty."}), 400
        y_vals = [float(p) for p in y_parts]
    except ValueError:
        return jsonify({"error": "y values contain non-numeric input."}), 400

    # Parse target x (optional)
    target_raw = data.get("target_x", "").strip()
    target_x   = None
    if target_raw:
        try:
            target_x = float(target_raw)
        except ValueError:
            return jsonify({"error": "Target x must be a valid number."}), 400

    # Validate
    valid, err = validate_input(x_vals, y_vals)
    if not valid:
        return jsonify({"error": err}), 400

    # Compute
    try:
        result = interpolate(x_vals, y_vals, target_x)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Computation error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
