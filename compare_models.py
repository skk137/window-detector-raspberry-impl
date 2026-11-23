"""
Σύγκριση μοντέλων σε δεδομένα βαθμονόμησης (τρέχει σε υπολογιστή ή στο Pi - δεν χρειάζεται Sense HAT).

Χρήση:
    python3 compare_models.py                            # επιλογή αρχείου από το calibrations/
    python3 compare_models.py calibrations/window.json   # νέο ή παλιό format JSON
    python3 compare_models.py calibrations/data.csv      # στήλες: distance,delta[,delta_std,n_samples]
    python3 compare_models.py FILE --show                # ανοίγει και το γράφημα σε παράθυρο

Κριτήρια - όλα με τα ίδια βάρη ανά σημείο (βλ. models.point_sigmas):
  - AIC / BIC: ποιότητα fit με ποινή για κάθε επιπλέον παράμετρο
  - σφάλμα απόστασης σε cm, και leave-one-out (σε σημεία που το fit δεν είδε)
  - αντιστρεψιμότητα: για detection το μοντέλο πρέπει να είναι γνησίως φθίνον
Τα πολυώνυμα μπαίνουν μόνο ως αναφορά: δεν έχουν ασύμπτωτη και συχνά δεν είναι μονότονα.

Αποτελέσματα, δίπλα στο αρχείο εισόδου: <name>_comparison.png και <name>_report.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from lmfit import Model

import models


BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_FOLDER = BASE_DIR / "calibrations"

# Σταθερό χρώμα ανά μοντέλο, ίδιο σε όλα τα γραφήματα
SERIES = {
    "dipole": "#2a78d6",
    "power": "#eb6834",
    "exponential": "#1baf7a",
    "poly2": "#eda100",
    "poly3": "#e87ba4",
}
# Χρώματα κειμένου, πλέγματος, αξόνων και φόντου
INK, INK_2, GRID, AXIS, SURFACE, SHADE = (
    "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec")

LABELS = {
    "dipole": "dipole  a/(x+x0)^3 + c",
    "power": "power  a/(x+x0)^n + c",
    "exponential": "exponential  a·exp(-bx) + c",
    "poly2": "poly2 (reference)",
    "poly3": "poly3 (reference)",
}


# ==========================
# Μοντέλα αναφοράς (μόνο για σύγκριση)
# ==========================

def poly2(x, a, b, c):
    return a * x**2 + b * x + c


def poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


REFERENCE = {"poly2": (poly2, 2), "poly3": (poly3, 3)}   # όνομα -> (συνάρτηση, βαθμός)


# ==========================
# Δεδομένα
# ==========================

def load_data(path):
    """Επιστρέφει x, y, y_std (ή None), n_samples (ή None), x_max - ταξινομημένα κατά απόσταση."""
    if path.suffix.lower() == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        cols = table.dtype.names
        x = np.atleast_1d(table["distance"]).astype(float)
        y = np.atleast_1d(table["delta"]).astype(float)
        y_std = np.atleast_1d(table["delta_std"]).astype(float) if "delta_std" in cols else None
        n = np.atleast_1d(table["n_samples"]).astype(float) if "n_samples" in cols else None
        x_max = float(x.max())
    else:
        with open(path) as f:
            data = json.load(f)
        if "points" in data:      # νέο format (window_detector.py)
            pts = data["points"]
            x = np.array([p["distance"] for p in pts], dtype=float)
            y = np.array([p["delta"] for p in pts], dtype=float)
            y_std = np.array([p["delta_std"] for p in pts], dtype=float)
            n = np.array([p["n_samples"] for p in pts], dtype=float)
        else:                     # αρχικό format
            x = np.array(data["distance"], dtype=float)
            y = np.array(data["delta"], dtype=float)
            y_std = n = None
        x_max = float(data.get("max_distance", x.max()))

    order = np.argsort(x)
    pick = (lambda a: a[order] if a is not None else None)
    return x[order], y[order], pick(y_std), pick(n), x_max


def choose_file():
    """Επιλογή αρχείου βαθμονόμησης από λίστα στο τερματικό."""
    files = sorted(CALIBRATION_FOLDER.glob("*.json")) + sorted(CALIBRATION_FOLDER.glob("*.csv"))
    if not files:
        sys.exit(f"No .json or .csv files in {CALIBRATION_FOLDER}")

    print("\nCalibration files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    while True:
        choice = input("Choose file: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("  Invalid choice.")


# ==========================
# Ανάλυση
# ==========================

def grid_inverse(predict, y_values, x_max):
    """
    delta -> απόσταση μέσω πυκνού πλέγματος. Δουλεύει για οποιοδήποτε μοντέλο, αρκεί να είναι
    γνησίως φθίνον στο [0, x_max]· αλλιώς επιστρέφει None (δεν αντιστρέφεται).
    """
    x_grid = np.linspace(0.0, x_max, 4001)
    y_grid = predict(x_grid)
    if not np.all(np.diff(y_grid) < 0):
        return None
    # Εκτός εύρους κόβεται στο 0 / x_max, όπως και στο detection
    return np.interp(y_values, y_grid[::-1], x_grid[::-1])


def summarize(res, x, y, x_max, loo):
    """Συγκεντρώνει τα κριτήρια ενός fit του lmfit (RMSE, χ², AIC, BIC, σφάλματα σε cm)."""
    predict = (lambda xs: res.eval(x=xs))
    x_hat = grid_inverse(predict, y, x_max)
    return {
        "res": res,
        "predict": predict,
        "k": res.nvarys,   # αριθμός παραμέτρων
        "rmse_uT": float(np.sqrt(np.mean((y - res.best_fit)**2))),
        "redchi": float(res.redchi),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "invertible": x_hat is not None,
        "rmse_cm": models.rms(x_hat - x) if x_hat is not None else None,
        "loo": loo if x_hat is not None else None,
        "loo_cm": models.rms(loo) if x_hat is not None and loo is not None else None,
    }


def analyse_physical(name, x, y, sigma, x_max):
    """Fit ενός φυσικού μοντέλου (dipole, power, exponential) και τα κριτήριά του."""
    spec = models.MODELS[name]
    try:
        # Ίδιο fit με αυτό που χρησιμοποιεί το window_detector.py ...
        fit = models.fit_model(name, x, y, sigma)
    except (RuntimeError, ValueError, TypeError) as e:
        return {"error": str(e)}

    # ... και το lmfit ξεκινάει από εκεί για AIC/BIC και αναφορά αβεβαιοτήτων
    _, (low, high) = models.initial_guess(name, x, y)
    model = Model(spec["func"])
    params = model.make_params(**fit["params"])
    for pname, lo, hi in zip(spec["params"], low, high):
        params[pname].set(min=lo, max=hi)
    res = model.fit(y, params, x=x, weights=1.0 / sigma)

    loo = models.loo_distance_errors(name, x, y, sigma, x_max)
    return summarize(res, x, y, x_max, loo)


def analyse_reference(name, x, y, sigma, x_max):
    """Fit ενός πολυωνύμου αναφοράς, με leave-one-out μέσω αριθμητικής αντιστροφής."""
    func, degree = REFERENCE[name]
    if len(x) <= degree + 1:
        return {"error": "not enough points"}

    # Γραμμικό πρόβλημα: το polyfit δίνει την ακριβή λύση, το lmfit την αναφορά
    model = Model(func)
    coeffs = np.polyfit(x, y, degree, w=1.0 / sigma)
    params = model.make_params(**dict(zip(model.param_names, coeffs)))
    res = model.fit(y, params, x=x, weights=1.0 / sigma)

    loo = np.full(len(x), np.nan)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        c = np.polyfit(x[keep], y[keep], degree, w=1.0 / sigma[keep])
        x_hat = grid_inverse(lambda xs: np.polyval(c, xs), y[i:i + 1], x_max)
        if x_hat is not None:
            loo[i] = x_hat[0] - x[i]
    return summarize(res, x, y, x_max, loo)


# ==========================
# Αναφορά (κείμενο)
# ==========================

def fmt(value, spec):
    """Μορφοποίηση αριθμού, ή "-" όταν λείπει η τιμή."""
    return format(value, spec) if value is not None else "-"


def make_report(path, x, y_std, x_max, results, best):
    """Κείμενο αναφοράς: πίνακας κριτηρίων, συμπεράσματα και τα αναλυτικά fit reports του lmfit."""
    lines = [
        f"Model comparison: {path.name}",
        f"{len(x)} points, 0-{x_max:g} cm, weights: "
        f"{'measured std + ' if y_std is not None else ''}"
        f"noise floor {models.NOISE_FLOOR_UT} uT + position error {models.POSITION_SIGMA_CM} cm",
        "",
        f"{'model':<13}{'k':>3}{'RMSE uT':>10}{'chi2_red':>10}{'AIC':>9}{'BIC':>9}"
        f"{'invert.':>9}{'err cm':>8}{'LOO cm':>8}",
    ]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"{name:<13}fit failed: {r['error']}")
            continue
        lines.append(
            f"{name:<13}{r['k']:>3}{r['rmse_uT']:>10.2f}{r['redchi']:>10.2f}"
            f"{r['aic']:>9.1f}{r['bic']:>9.1f}{'yes' if r['invertible'] else 'NO':>9}"
            f"{fmt(r['rmse_cm'], '.2f'):>8}{fmt(r['loo_cm'], '.2f'):>8}")

    lines.append("")
    if best:
        lines.append(f"Best model for detection: {best} "
                     f"(leave-one-out distance error {results[best]['loo_cm']:.2f} cm)")
    else:
        lines.append("No physical model could be fitted.")

    power = results.get("power", {})
    if "res" in power:
        n = power["res"].params["n"]
        err = f" +/- {n.stderr:.2f}" if n.stderr is not None else ""
        lines.append(f"Power-law exponent n = {n.value:.2f}{err}  (pure dipole: n = 3)")

    fitted = [n for n, r in results.items() if "aic" in r]
    if fitted:
        lines.append(f"Lowest AIC: {min(fitted, key=lambda n: results[n]['aic'])}, "
                     f"lowest BIC: {min(fitted, key=lambda n: results[n]['bic'])}")

    for name in REFERENCE:
        r = results.get(name, {})
        if "res" not in r:
            continue
        if not r["invertible"]:
            lines.append(f"{name}: not strictly decreasing -> one delta gives several "
                         "distances, unusable for detection.")
        else:
            lines.append(f"{name}: invertible inside the range, but it has no asymptote "
                         "(diverges outside it) -> reference only.")

    for name in models.MODELS:
        r = results.get(name, {})
        if "res" in r:
            lines += ["", "=" * 60, f"{name}", "=" * 60, r["res"].fit_report()]
    return "\n".join(lines)


# ==========================
# Γράφημα
# ==========================

def plot(png_path, title, x, y, sigma, x_max, results, best, show):
    """Γράφημα 2x2: (a) fits, (b) λογαριθμική κλίμακα, (c) υπόλοιπα, (d) leave-one-out."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")    # χωρίς οθόνη (π.χ. μέσω SSH) σώζει μόνο το PNG
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ax_lin, ax_log, ax_res, ax_loo = axes.flat
    x_fine = np.linspace(0.0, 1.15 * x_max, 400)   # λίγο πέρα από το εύρος: φαίνεται η προέκταση
    physical = [n for n in models.MODELS if "res" in results.get(n, {})]
    fitted = [n for n in results if "res" in results[n]]

    def label(name):
        return LABELS[name] + ("  - selected" if name == best else "")

    # (a) όλα τα μοντέλα, γραμμική κλίμακα, με τη ζώνη προέκτασης σκιασμένη
    for name in fitted:
        style = "--" if name in REFERENCE else "-"
        ax_lin.plot(x_fine, results[name]["predict"](x_fine), style, color=SERIES[name],
                    lw=1.6, label=label(name))
    ax_lin.errorbar(x, y, yerr=sigma, fmt="o", ms=5, color=INK, ecolor=INK_2,
                    elinewidth=1, label="measurements (±σ)", zorder=5)
    span = np.ptp(y)
    ax_lin.set_ylim(min(0.0, y.min()) - 0.08 * span, y.max() + 0.12 * span)
    ax_lin.set_title("(a) Calibration data and fitted models", loc="left")
    ax_lin.set_ylabel("Delta (µT)")
    ax_lin.legend(fontsize=8.5)

    # (b) φυσικά μοντέλα σε λογαριθμική κλίμακα: φαίνεται η ουρά, εκεί που δυσκολεύει το detection
    for name in physical:
        ax_log.plot(x_fine, results[name]["predict"](x_fine), color=SERIES[name],
                    lw=1.6, label=label(name))
    ax_log.errorbar(x, y, yerr=np.minimum(sigma, 0.95 * y), fmt="o", ms=5, color=INK,
                    ecolor=INK_2, elinewidth=1, label="measurements (±σ)", zorder=5)
    ax_log.set_yscale("log", nonpositive="mask")
    ax_log.set_title("(b) Same, log scale (physical models)", loc="left")
    ax_log.set_ylabel("Delta (µT)")
    ax_log.legend(fontsize=8.5)

    # (c) κανονικοποιημένα υπόλοιπα: συστηματικό μοτίβο = λάθος σχήμα μοντέλου
    for name in physical:
        r = results[name]
        ax_res.plot(x, (y - r["res"].best_fit) / sigma, "o-", color=SERIES[name],
                    lw=1.2, ms=5, label=name)
    ax_res.axhline(0, color=AXIS, lw=1)
    ax_res.set_title("(c) Normalized residuals (data − model) / σ", loc="left")
    ax_res.set_ylabel("Residual / σ")
    ax_res.legend(fontsize=8.5)

    # (d) leave-one-out: πόσα cm λάθος σε σημεία που το fit δεν είδε
    for name in physical:
        r = results[name]
        if r["loo"] is not None:
            ax_loo.plot(x, r["loo"], "o-", color=SERIES[name], lw=1.2, ms=5,
                        label=f"{name}  (RMS {fmt(r['loo_cm'], '.2f')} cm)")
    ax_loo.axhline(0, color=AXIS, lw=1)
    ax_loo.set_title("(d) Leave-one-out distance error", loc="left")
    ax_loo.set_ylabel("Predicted − true (cm)")
    ax_loo.legend(fontsize=8.5)

    for ax in axes.flat:
        ax.set_xlabel("Window opening (cm)")
        ax.axvspan(x_max, x_fine[-1], color=SHADE, zorder=0, lw=0)   # εκτός εύρους βαθμονόμησης
        ax.set_xlim(x_fine[0], x_fine[-1])

    fig.suptitle(f"{title}   (shaded: outside calibrated range)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(png_path, dpi=150)
    if show:
        plt.show()


# ==========================
# Κύριο πρόγραμμα
# ==========================

def main():
    """Φόρτωση δεδομένων, ανάλυση όλων των μοντέλων, αναφορά και γράφημα."""
    parser = argparse.ArgumentParser(description="Compare magnet-field models on calibration data.")
    parser.add_argument("file", nargs="?", help="calibration .json or .csv (default: choose)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = parser.parse_args()

    path = Path(args.file) if args.file else choose_file()
    x, y, y_std, n, x_max = load_data(path)
    if len(x) < 5:
        sys.exit(f"Only {len(x)} points - at least 5 are needed (10+ recommended).")

    sigma = models.point_sigmas(x, y, y_std, n)   # ίδια βάρη για όλα τα μοντέλα

    results = {}
    for name in models.MODELS:
        results[name] = analyse_physical(name, x, y, sigma, x_max)
    for name in REFERENCE:
        results[name] = analyse_reference(name, x, y, sigma, x_max)

    # Επιλογή όπως στο window_detector.py: μικρότερο σφάλμα leave-one-out (μόνο φυσικά μοντέλα)
    candidates = [m for m in models.MODELS if results[m].get("loo_cm") is not None]
    best = min(candidates, key=lambda m: results[m]["loo_cm"]) if candidates else None

    report = make_report(path, x, y_std, x_max, results, best)
    report_path = path.with_name(f"{path.stem}_report.txt")
    png_path = path.with_name(f"{path.stem}_comparison.png")
    report_path.write_text(report, encoding="utf-8")

    summary = report.split("\n" + "=" * 60)[0]   # πίνακας + συμπεράσματα, χωρίς τα fit reports
    print(summary)
    print(f"\nSaved: {report_path}\n       {png_path}")

    plot(png_path, path.name, x, y, sigma, x_max, results, best, args.show)


if __name__ == "__main__":
    main()
