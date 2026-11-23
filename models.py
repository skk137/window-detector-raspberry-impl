"""
Μοντέλα: μεταβολή του μαγνητικού πεδίου (delta, µT) ως συνάρτηση του ανοίγματος
του παραθύρου (cm).

Κοινός κώδικας για το window_detector.py (μετρήσεις στο Raspberry Pi) και το
compare_models.py (ανάλυση). Δεν χρειάζεται Sense HAT - τρέχει και σε υπολογιστή.

Περιεχόμενα:
  - τα τρία μοντέλα (δίπολο, νόμος δύναμης, εκθετικό) και η αντιστροφή τους
  - η σταθμισμένη προσαρμογή (fit) με βάρη ανά σημείο
  - η επιλογή μοντέλου με το σφάλμα απόστασης leave-one-out (cm)
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


# Αβεβαιότητες που καθορίζουν τα βάρη του fit (βλ. point_sigmas)
POSITION_SIGMA_CM = 0.2   # ακρίβεια τοποθέτησης του παραθύρου με το χέρι (~2 mm)
NOISE_FLOOR_UT = 0.2      # ελάχιστη αβεβαιότητα ανά σημείο (θόρυβος / ολίσθηση αισθητήρα)


# ==========================
# Μοντέλα
# ==========================

def dipole(x, a, x0, c):
    # Μαγνητικό δίπολο: |B| ~ 1/r^3 με r = x + x0
    # (x0 = απόσταση μαγνήτη-αισθητήρα όταν το παράθυρο είναι κλειστό,
    #  c = σταθερός όρος, η τιμή στην οποία τείνει το delta για πολύ μεγάλο άνοιγμα)
    return a / (x + x0)**3 + c


def power(x, a, x0, n, c):
    # Ίδιο με το dipole, αλλά ο εκθέτης n βγαίνει από τα δεδομένα.
    # n ~ 3 => πειραματική επιβεβαίωση ότι ο μαγνήτης συμπεριφέρεται σαν δίπολο
    return a / (x + x0)**n + c


def exponential(x, a, b, c):
    # Εμπειρικό μοντέλο (η αρχική επιλογή) - κρατιέται για σύγκριση
    return a * np.exp(-b * x) + c


# Όνομα μοντέλου -> συνάρτηση και ονόματα παραμέτρων (με τη σειρά που τις δέχεται)
MODELS = {
    "dipole": {"func": dipole, "params": ("a", "x0", "c")},
    "power": {"func": power, "params": ("a", "x0", "n", "c")},
    "exponential": {"func": exponential, "params": ("a", "b", "c")},
}


def invert(name, params, delta):
    """
    Αντίστροφο μοντέλο: delta (µT) -> απόσταση (cm).

    delta <= c σημαίνει «πιο μακριά απ' όσο μπορεί να ξεχωρίσει το μοντέλο» και
    επιστρέφει inf (ο καλών το κόβει στο μέγιστο -> 100%). Αρνητικό αποτέλεσμα
    σημαίνει πιο κοντά από 0 cm (ο καλών το κόβει στο 0).
    """
    p = params
    excess = np.asarray(delta, dtype=float) - p["c"]   # το μέρος του πεδίου που οφείλεται στον μαγνήτη

    # Λύση της εξίσωσης του μοντέλου ως προς x. Για excess <= 0 βγαίνει nan/inf
    # (διαίρεση με 0, λογάριθμος αρνητικού) - χωρίς προειδοποιήσεις, διορθώνεται παρακάτω.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if name == "dipole":
            x = np.cbrt(p["a"] / excess) - p["x0"]
        elif name == "power":
            x = (p["a"] / excess)**(1.0 / p["n"]) - p["x0"]
        elif name == "exponential":
            x = -np.log(excess / p["a"]) / p["b"]
        else:
            raise ValueError(f"Unknown model: {name}")

    # excess <= 0: το παράθυρο είναι πέρα από όσο ξεχωρίζει το μοντέλο -> inf (δηλ. 100%), όχι 0%
    x = np.where(excess > 0, x, np.inf)
    return float(x) if x.ndim == 0 else x


# ==========================
# Προσαρμογή (fit)
# ==========================

def point_sigmas(x, y, y_std=None, n_samples=None):
    """
    Αβεβαιότητα (µT) κάθε σημείου βαθμονόμησης:
      - θόρυβος του μέσου όρου (std / sqrt(N)) συν ένα μικρό κατώτατο όριο
      - σφάλμα θέσης: αν το παράθυρο τοποθετηθεί 2 mm δίπλα, το delta αλλάζει κατά κλίση * 2 mm

    Κοντά στον μαγνήτη η κλίση είναι πολύ μεγάλη, οπότε τα σημεία εκεί παίρνουν μικρότερο
    βάρος και δεν κυριαρχούν στο fit (το πρόβλημα του απλού RMSE). Η κλίση υπολογίζεται
    από τα ίδια τα δεδομένα, άρα τα βάρη είναι ίδια για όλα τα μοντέλα και τα AIC/BIC
    μένουν συγκρίσιμα μεταξύ τους.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Τυπικό σφάλμα του μέσου όρου κάθε σημείου (0 όταν δεν υπάρχει std, π.χ. σημεία από πίνακα)
    sem = np.zeros_like(y)
    if y_std is not None:
        n = np.asarray(n_samples if n_samples is not None else 1, dtype=float)
        sem = np.asarray(y_std, dtype=float) / np.sqrt(n)

    slope = np.gradient(y, x)   # κλίση d(delta)/dx από διαφορές γειτονικών σημείων
    # Ανεξάρτητες πηγές σφάλματος -> προστίθενται τα τετράγωνα
    return np.sqrt(sem**2 + NOISE_FLOOR_UT**2 + (slope * POSITION_SIGMA_CM)**2)


def initial_guess(name, x, y):
    """Αρχικές τιμές από γραμμικοποίηση των δεδομένων, και όρια των παραμέτρων."""
    span = max(float(np.ptp(y)), 1e-6)
    # Το c λίγο κάτω από το μικρότερο delta, ώστε y - c > 0 (χρειάζεται για λογάριθμο και ρίζα)
    c0 = 0.9 * y.min() if y.min() > 0 else y.min() - 0.05 * span
    excess = y - c0                              # > 0 για όλα τα σημεία
    x0_max = 10 * max(float(x.max()), 1.0) + 10  # άνω όριο του x0, ώστε το fit να μην ξεφεύγει

    if name == "exponential":
        # ln(y - c) = ln(a) - b*x  -> ευθεία
        slope, intercept = np.polyfit(x, np.log(excess), 1)
        b0 = -slope if slope < 0 else 3.0 / max(float(x.max()), 1e-6)
        p0 = [float(np.exp(intercept)), b0, c0]
        bounds = ([1e-12, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        return p0, bounds

    # (y - c)^(-1/3) = (x + x0) / a^(1/3)  -> ευθεία
    slope, intercept = np.polyfit(x, excess**(-1.0 / 3.0), 1)
    if slope > 0 and intercept > 0:
        x00 = intercept / slope
        a0 = slope**-3
    else:
        # Η ευθεία δεν βγήκε λογική (π.χ. θόρυβος): εκτίμηση από το πιο κοντινό σημείο
        x00 = 1.0
        a0 = excess[np.argmin(x)] * (x.min() + x00)**3
    x00 = float(np.clip(x00, 1e-3, x0_max))

    # Όρια: a > 0, x0 > 0 και, στο power, εκθέτης n από 0.5 έως 6
    if name == "dipole":
        return [a0, x00, c0], ([1e-12, 1e-3, -np.inf], [np.inf, x0_max, np.inf])
    return [a0, x00, 3.0, c0], ([1e-12, 1e-3, 0.5, -np.inf], [np.inf, x0_max, 6.0, np.inf])


def fit_model(name, x, y, sigma):
    """
    Σταθμισμένο fit (βάρη 1/sigma) με όρια στις παραμέτρους.
    Επιστρέφει παραμέτρους, τυπικά σφάλματα, RMSE (µT) και ανηγμένο χ².
    """
    spec = MODELS[name]
    p0, bounds = initial_guess(name, x, y)

    with warnings.catch_warnings():
        # "Covariance could not be estimated" -> φαίνεται ως stderr = inf
        warnings.simplefilter("ignore", OptimizeWarning)
        # Με όρια το curve_fit χρησιμοποιεί τη μέθοδο TRF. Το x_scale="jac" βοηθά όταν οι
        # παράμετροι έχουν πολύ διαφορετική τάξη μεγέθους (π.χ. a ~ 1e5, x0 ~ 10).
        popt, pcov = curve_fit(spec["func"], x, y, p0=p0, sigma=sigma, bounds=bounds,
                               maxfev=20000, x_scale="jac")

    resid = y - spec["func"](x, *popt)
    dof = max(len(x) - len(popt), 1)   # βαθμοί ελευθερίας
    with np.errstate(invalid="ignore"):
        stderr = np.sqrt(np.diag(pcov))   # τυπικά σφάλματα από τον πίνακα συνδιακύμανσης

    return {
        "params": dict(zip(spec["params"], map(float, popt))),
        "stderr": dict(zip(spec["params"], map(float, stderr))),
        "rmse_uT": float(np.sqrt(np.mean(resid**2))),
        "chi2_red": float(np.sum((resid / sigma)**2) / dof),
    }


def distance_errors(name, params, x, y, x_max):
    """Σφάλμα απόστασης (cm) όταν το μοντέλο αντιστρέφεται στα σημεία της βαθμονόμησης."""
    x_hat = np.clip(invert(name, params, y), 0.0, x_max)
    return x_hat - x   # εκτίμηση - πραγματική απόσταση


def loo_distance_errors(name, x, y, sigma, x_max):
    """
    Leave-one-out: για κάθε σημείο, fit ΧΩΡΙΣ αυτό και πρόβλεψη της απόστασής του.

    Μετράει πόσο καλά προβλέπει το μοντέλο σημεία που δεν έχει δει - δίκαιη σύγκριση
    ανάμεσα σε μοντέλα με διαφορετικό αριθμό παραμέτρων, στη μονάδα που ενδιαφέρει
    την ανίχνευση (cm).
    """
    errors = np.full(len(x), np.nan)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i   # όλα τα σημεία εκτός από το i
        try:
            fit = fit_model(name, x[keep], y[keep], sigma[keep])
        except (RuntimeError, ValueError, TypeError):
            continue   # το fit απέτυχε χωρίς αυτό το σημείο: μένει NaN
        errors[i] = distance_errors(name, fit["params"], x[i:i + 1], y[i:i + 1], x_max)[0]
    return errors


def rms(values):
    """Ρίζα του μέσου τετραγώνου, αγνοώντας NaN/inf. None αν δεν μένει καμία τιμή."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def fit_all(x, y, sigma, x_max):
    """
    Fit όλων των μοντέλων και επιλογή εκείνου με το μικρότερο σφάλμα απόστασης
    leave-one-out (cm). Επιστρέφει (αποτελέσματα ανά μοντέλο, όνομα επιλεγμένου ή None).
    """
    results = {}
    for name in MODELS:
        try:
            fit = fit_model(name, x, y, sigma)
        except (RuntimeError, ValueError, TypeError) as e:
            results[name] = {"error": str(e)}
            continue

        loo = loo_distance_errors(name, x, y, sigma, x_max)
        fit["rmse_cm"] = rms(distance_errors(name, fit["params"], x, y, x_max))
        fit["loo_rmse_cm"] = rms(loo)
        fit["loo_failed_folds"] = int(np.sum(~np.isfinite(loo)))   # πόσα σημεία δεν προβλέφθηκαν
        results[name] = fit

    candidates = [n for n, r in results.items() if r.get("loo_rmse_cm") is not None]
    selected = min(candidates, key=lambda n: results[n]["loo_rmse_cm"]) if candidates else None
    return results, selected
