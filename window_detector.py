"""
Σύστημα ανίχνευσης ανοίγματος παραθύρου - μαγνητόμετρο του Sense HAT (τρέχει στο Raspberry Pi)

Χρήση:
    python3 window_detector.py                  # μενού στο τερματικό
    python3 window_detector.py --detect NAME    # μόνο ανίχνευση, χωρίς μενού (χωρίς οθόνη)
    python3 window_detector.py --detect latest  # το ίδιο, με την πιο πρόσφατη βαθμονόμηση (systemd)
    streamlit run app.py                        # τα ίδια μέσα από φυλλομετρητή (βλ. app.py)

Βασικά χαρακτηριστικά (σε σχέση με την αρχική έκδοση του κώδικα):
  - delta = |B - B0| με ΔΙΑΝΥΣΜΑΤΑ, όχι διαφορά μέτρων: μετράει μόνο το πεδίο
    του μαγνήτη, ανεξάρτητα από το γήινο πεδίο. Αν μετακινήσεις/περιστρέψεις
    το Pi, αρκεί νέο baseline (επιλογή 4) - όχι νέο calibration.
  - calibration με ωμούς μέσους όρους (χωρίς EMA) μετά από χρόνο σταθεροποίησης,
    και αποθήκευση std ανά σημείο
  - fit 3 μοντέλων (dipole 1/r^3, power 1/r^n, exponential) με βάρη, επιλογή
    με leave-one-out σφάλμα απόστασης σε cm (βλ. models.py)
  - τα δεδομένα σώζονται μετά από κάθε σημείο, πριν το fit
  - σωστή αντιστροφή: delta κάτω από την ασύμπτωτη = 100%, όχι 0%
  - καταγραφή σε CSV (logs/) μαζί με θερμοκρασία, μόνο όσο το παράθυρο είναι ανοιχτό
  - αυτόματο re-baseline όταν το παράθυρο μένει τέρμα ανοιχτό (διόρθωση ολίσθησης)
  - LED matrix σβηστό, Ctrl+C γυρνάει στο μενού

Οι συναρτήσεις χωρίς input()/print (take_baseline, measure_point, run_fit,
apply_baseline, Detector) χρησιμοποιούνται και από το app.py.
"""

import argparse
import csv
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sense_hat import SenseHat

import models


# Φάκελοι δίπλα στο script, ώστε να δουλεύει από όποιον φάκελο κι αν ξεκινήσει
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_FOLDER = BASE_DIR / "calibrations"
LOG_FOLDER = BASE_DIR / "logs"

# Βαθμονόμηση
SAMPLE_INTERVAL = 0.05     # s ανάμεσα σε αναγνώσεις στο calibration (20 Hz)
SETTLE_TIME = 2.0          # s αναγνώσεων που πετιούνται μετά από κάθε μετακίνηση
BASELINE_SAMPLES = 200     # αναγνώσεις για το υπόβαθρο (~10 s + σταθεροποίηση)
POINT_SAMPLES = 50         # αναγνώσεις ανά σημείο βαθμονόμησης (~2.5 s + σταθεροποίηση)
MIN_POINTS = 6             # το power μοντέλο έχει 4 παραμέτρους
NAME_PATTERN = r"[A-Za-z0-9_-]+"   # επιτρεπτά ονόματα βαθμονόμησης (γίνονται ονόματα αρχείων)
POINT_COLUMNS = ("distance", "delta", "delta_std", "n_samples")   # στήλες του CSV κάθε calibration

# Ανίχνευση και καταγραφή
DETECT_INTERVAL = 0.1      # s ανάμεσα σε αναγνώσεις στο detection
DETECT_ALPHA = 0.05        # EMA: σταθερά χρόνου ~ 0.1 / 0.05 = 2 s (μισό τρεμόπαιγμα, ~2-3 s καθυστέρηση)
PRINT_EVERY = 5            # εκτύπωση + καταγραφή κάθε 5 αναγνώσεις (0.5 s)
DISPLAY_STEP = 5           # στρογγυλοποίηση % στην οθόνη (στο CSV μπαίνει η ακριβής τιμή)
LOG_THRESHOLD = 3          # % : στο CSV γράφεται μόνο άνοιγμα πάνω από αυτό (+ η στιγμή που κλείνει)
LOG_EVERY = 60             # s : μία γραμμή στο CSV ανά τόσα δευτερόλεπτα όσο είναι ανοιχτό (στα πειράματα: 15)
AUTO_REBASELINE_AFTER = 300     # s συνεχόμενα στο 100% -> αυτόματο re-baseline (θέλει full_open)
AUTO_REBASELINE_MAX_SHIFT = 5   # µT: μεγαλύτερη αλλαγή δεν είναι ολίσθηση - δεν διορθώνεται


sense = SenseHat()                # μία σύνδεση με τον αισθητήρα για όλο το πρόγραμμα
_sensor_lock = threading.Lock()   # το app.py μπορεί να διαβάζει από πολλά threads


# ==========================
# Ανάγνωση αισθητήρων
# ==========================

def read_vector():
    """Μία ανάγνωση του μαγνητόμετρου: διάνυσμα (bx, by, bz) σε µT."""
    with _sensor_lock:
        mag = sense.get_compass_raw()
    return np.array([mag["x"], mag["y"], mag["z"]], dtype=float)


def read_temperature():
    """Θερμοκρασία (°C) από το Sense HAT, ή None αν αποτύχει η ανάγνωση."""
    try:
        with _sensor_lock:
            # get_temperature(): ο αισθητήρας θερμοκρασίας/υγρασίας του Sense HAT,
            # που ζεσταίνεται μαζί με την πλακέτα όπως και το μαγνητόμετρο
            return round(float(sense.get_temperature()), 2)
    except OSError:   # η θερμοκρασία δεν είναι κρίσιμη - δεν σταματάμε τη μέτρηση
        return None


def collect(samples):
    """Διαβάζει και πετάει αναγνώσεις για SETTLE_TIME, μετά επιστρέφει `samples` ωμά διανύσματα."""
    # Διαβάζουμε (όχι απλό sleep) ώστε να αδειάσει και το smoothing της βιβλιοθήκης
    for _ in range(int(SETTLE_TIME / SAMPLE_INTERVAL)):
        read_vector()
        time.sleep(SAMPLE_INTERVAL)

    vectors = []
    for _ in range(samples):
        vectors.append(read_vector())
        time.sleep(SAMPLE_INTERVAL)
    return np.array(vectors)   # πίνακας samples x 3


def now_iso():
    """Τρέχουσα ώρα σε μορφή ISO (για τα αρχεία)."""
    return datetime.now().isoformat(timespec="seconds")


def take_baseline(samples=BASELINE_SAMPLES):
    """
    Πεδίο υποβάθρου με τον μαγνήτη απομακρυσμένο
    (χωρίς ερωτήσεις - το χρησιμοποιεί και το app.py).
    """
    vectors = collect(samples)
    b0 = vectors.mean(axis=0)   # B0: μέσος όρος ανά άξονα (διάνυσμα, όχι μέτρο)
    return {
        "vector": b0.tolist(),
        "std": vectors.std(axis=0, ddof=1).tolist(),
        "magnitude": float(np.linalg.norm(b0)),
        "temperature": read_temperature(),   # για την προειδοποίηση θερμοκρασίας στο detection
        "time": now_iso(),
    }


def measure_baseline(samples=BASELINE_SAMPLES):
    """Έκδοση τερματικού: ζητά να απομακρυνθεί ο μαγνήτης και μετράει το υπόβαθρο."""
    print("\nRemove the magnet from the sensor (the same way every time).")
    input("Press ENTER when ready...")
    print(f"Measuring baseline ({samples} samples)...")

    baseline = take_baseline(samples)
    x, y, z = baseline["vector"]
    print(f"Baseline: |B0| = {baseline['magnitude']:.2f} uT (x={x:.2f}, y={y:.2f}, z={z:.2f})")
    return baseline


# ==========================
# Είσοδος από τον χρήστη (τερματικό)
# ==========================

def ask_number(prompt, cast, minimum):
    """Ζητά αριθμό μέχρι να δοθεί έγκυρος και >= minimum (δέχεται και κόμμα για υποδιαστολή)."""
    while True:
        text = input(prompt).strip().replace(",", ".")
        try:
            value = cast(text)
        except ValueError:
            print("  Not a valid number, try again.")
            continue
        if value < minimum:
            print(f"  Must be at least {minimum}.")
            continue
        return value


def ask_profile_name():
    """Ζητά όνομα βαθμονόμησης· ρωτάει πριν αντικαταστήσει μια υπάρχουσα."""
    while True:
        name = input("Calibration name (example: vertical_window): ").strip()
        if not valid_name(name):
            print("  Use only latin letters, numbers, _ and - (no spaces).")
            continue
        if calibration_path(name).exists():
            answer = input(f"  '{name}' already exists. Overwrite? [y/N]: ")
            if answer.strip().lower() != "y":
                continue
        return name


# ==========================
# Αρχεία βαθμονόμησης
# ==========================

def valid_name(name):
    """Μόνο λατινικά γράμματα, αριθμοί, _ και - (το όνομα γίνεται όνομα αρχείου)."""
    return re.fullmatch(NAME_PATTERN, name) is not None


def calibration_path(name):
    """Το JSON με όλη τη βαθμονόμηση (σημεία, fits, υπόβαθρο, ιστορικό)."""
    return CALIBRATION_FOLDER / f"{name}.json"


def points_path(name):
    """Το CSV με μόνο τα σημεία της βαθμονόμησης."""
    return CALIBRATION_FOLDER / f"{name}.csv"


def save_calibration(cal):
    """Αποθηκεύει τη βαθμονόμηση σε JSON και τα σημεία της σε CSV."""
    CALIBRATION_FOLDER.mkdir(exist_ok=True)
    with open(calibration_path(cal["name"]), "w") as f:
        json.dump(cal, f, indent=4)

    # Οι μετρήσεις και σε CSV δίπλα στο JSON: ανοίγει σε Excel, ξαναφορτώνεται στο app
    with open(points_path(cal["name"]), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(POINT_COLUMNS)
        for p in cal["points"]:
            writer.writerow([p[c] for c in POINT_COLUMNS])


def load_calibration(name):
    """Φορτώνει μια βαθμονόμηση από το JSON της."""
    with open(calibration_path(name)) as f:
        return json.load(f)


def list_calibrations():
    """
    Επιστρέφει (χρησιμοποιήσιμα, παραλειπόμενα) ονόματα.
    Παραλείπονται τα αρχεία παλιού format ή χωρίς επιτυχημένο fit.
    """
    usable, skipped = [], []
    for path in sorted(CALIBRATION_FOLDER.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped.append(path.stem)
            continue
        # version 2 = διανυσματικό υπόβαθρο· τα αρχεία του αρχικού script δεν είναι συμβατά
        if data.get("version") == 2 and data.get("selected_model"):
            usable.append(path.stem)
        else:
            skipped.append(path.stem)
    return usable, skipped


def latest_calibration():
    """Η πιο πρόσφατη χρησιμοποιήσιμη βαθμονόμηση (κατά ημερομηνία δημιουργίας), ή None."""
    names, _ = list_calibrations()
    if not names:
        return None
    # Από το "created" μέσα στο JSON, όχι από την ημερομηνία του αρχείου,
    # που αλλάζει σε κάθε (αυτόματο) re-baseline
    return max(names, key=lambda n: load_calibration(n)["created"])


def choose_calibration():
    """Επιλογή βαθμονόμησης από λίστα στο τερματικό· None = επιστροφή στο μενού."""
    names, skipped = list_calibrations()
    if skipped:
        print(f"\n(Skipped - old format or no fit: {', '.join(skipped)})")
    if not names:
        print("\nNo usable calibrations - run option 2 first.")
        return None

    print("\nAvailable calibrations:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input("Choose calibration (0 = back): ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("  Invalid choice.")


# ==========================
# Βαθμονόμηση
# ==========================

def new_calibration(name, max_distance, baseline):
    """Νέα, κενή βαθμονόμηση: υπόβαθρο, ρυθμίσεις μέτρησης και χώρος για σημεία και fits."""
    return {
        "version": 2,
        "name": name,
        "created": now_iso(),
        "max_distance": max_distance,
        "baseline": baseline,
        "baseline_history": [],
        "settings": {   # οι ρυθμίσεις με τις οποίες έγινε, για να ξέρουμε αργότερα πώς μετρήθηκε
            "sample_interval": SAMPLE_INTERVAL,
            "settle_time": SETTLE_TIME,
            "samples_per_point": POINT_SAMPLES,
            "position_sigma_cm": models.POSITION_SIGMA_CM,
            "noise_floor_uT": models.NOISE_FLOOR_UT,
        },
        "points": [],
        "fits": {},
        "selected_model": None,
    }


def calibration_distances(max_distance, points):
    """Ισαπέχουσες αποστάσεις από 0 έως max_distance (cm)."""
    step = max_distance / (points - 1)
    return [round(i * step, 2) for i in range(points)]


def measure_point(distance, b0):
    """
    Παράθυρο ακίνητο στα `distance` cm -> ένα σημείο βαθμονόμησης
    (χωρίς ερωτήσεις - το χρησιμοποιεί και το app.py).
    """
    vectors = collect(POINT_SAMPLES)
    deltas = np.linalg.norm(vectors - b0, axis=1)   # delta = |B - B0| για κάθε ανάγνωση
    return {
        "distance": distance,
        "delta": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)),
        "n_samples": len(deltas),
        "vector_mean": vectors.mean(axis=0).tolist(),   # χρειάζεται στο full_open_from_last_point
    }


def measure_full_open(cal):
    """
    Παράθυρο τέρμα ανοιχτό (στο μηχανικό του όριο), με τον μαγνήτη στη θέση του: αποθηκεύει
    το πεδίο του μαγνήτη εκεί, ώστε αργότερα να διορθώνεται το υπόβαθρο χωρίς να βγαίνει ο μαγνήτης.
    Επιστρέφει κείμενο προειδοποίησης αν η μέτρηση φαίνεται λάθος, αλλιώς None.
    """
    vectors = collect(2 * POINT_SAMPLES)
    field = vectors.mean(axis=0) - np.array(cal["baseline"]["vector"])   # πεδίο μαγνήτη = B - B0
    cal["full_open"] = {"field": field.tolist(), "delta": float(np.linalg.norm(field)),
                        "time": now_iso()}

    # Πιο μακριά από το τελευταίο σημείο -> το πεδίο πρέπει να είναι μικρότερο
    if cal["points"] and cal["full_open"]["delta"] > cal["points"][-1]["delta"]:
        return (f"Fully open gave {cal['full_open']['delta']:.2f} uT, more than the last point "
                f"({cal['points'][-1]['delta']:.2f} uT) - is the window really at its stop?")
    return None


def full_open_from_last_point(cal):
    """
    Το όριο του παραθύρου είναι το τελευταίο σημείο βαθμονόμησης:
    το ξαναχρησιμοποιεί, χωρίς νέα μέτρηση.
    """
    field = np.array(cal["points"][-1]["vector_mean"]) - np.array(cal["baseline"]["vector"])
    cal["full_open"] = {"field": field.tolist(), "delta": float(np.linalg.norm(field)),
                        "time": now_iso(), "from_last_point": True}


def run_fit(cal):
    """Fit όλων των μοντέλων στα σημεία· αποθηκεύει στο cal τα fits και το επιλεγμένο μοντέλο."""
    pts = cal["points"]
    x = np.array([p["distance"] for p in pts])
    y = np.array([p["delta"] for p in pts])
    sigma = models.point_sigmas(x, y,
                                [p["delta_std"] for p in pts],
                                [p["n_samples"] for p in pts])
    cal["fits"], cal["selected_model"] = models.fit_all(x, y, sigma, cal["max_distance"])


def calibration_mode():
    """Βαθμονόμηση από το τερματικό: υπόβαθρο, σημεία, «τέρμα ανοιχτό» και fit."""
    name = ask_profile_name()
    max_distance = ask_number("Maximum window opening (cm, about 35-40 at most - the smaller the "
                              "more accurate, depends on the magnet; beyond it = 100% open): ",
                              float, 0.5)
    points = ask_number(f"Number of calibration points (about one every 5 cm, min {MIN_POINTS}): ",
                        int, MIN_POINTS)

    baseline = measure_baseline()
    print("\nPut the magnet back in its place.")

    cal = new_calibration(name, max_distance, baseline)
    b0 = np.array(baseline["vector"])

    print("\nStarting calibration (Ctrl+C aborts, points so far stay saved)\n")

    for distance in calibration_distances(max_distance, points):
        input(f"\nMove window to {distance} cm (press ENTER)")
        print("Measuring...")

        point = measure_point(distance, b0)
        cal["points"].append(point)
        save_calibration(cal)   # σώζεται μετά από κάθε σημείο

        print(f"Saved: {distance} cm  Delta={point['delta']:.2f} uT "
              f"(std {point['delta_std']:.2f})")

    # Μέτρηση "τέρμα ανοιχτό" για το αυτόματο re-baseline
    answer = ""
    while answer not in ("y", "n", "s"):
        answer = input(f"\nDoes the window open further than {max_distance:g} cm? "
                       "[y = yes, n = no, s = skip automatic re-baseline]: ").strip().lower()
    if answer == "y":
        input("Open the window fully, against its stop (press ENTER)")
        print("Measuring...")
        warning = measure_full_open(cal)
        if warning:
            print(f"  WARNING: {warning}")
    elif answer == "n":
        full_open_from_last_point(cal)
        print(f"Using the last point ({max_distance:g} cm) as fully open.")

    print("\nRunning curve fitting (3 models + leave-one-out check)...")
    run_fit(cal)
    save_calibration(cal)
    print_fit_summary(cal)
    print(f"\nCalibration saved: {calibration_path(name)}")


def print_fit_summary(cal):
    """Πίνακας με τα αποτελέσματα του fit κάθε μοντέλου στο τερματικό."""
    fits, selected = cal["fits"], cal["selected_model"]

    print(f"\n{'model':<13}{'RMSE uT':>9}{'err cm':>9}{'LOO cm':>9}")
    for name, fit in fits.items():
        if "error" in fit:
            print(f"{name:<13}fit failed: {fit['error']}")
            continue
        err, loo = fit["rmse_cm"], fit["loo_rmse_cm"]
        err_text = f"{err:.2f}" if err is not None else "-"
        loo_text = f"{loo:.2f}" if loo is not None else "-"
        print(f"{name:<13}{fit['rmse_uT']:>9.2f}{err_text:>9}{loo_text:>9}")

    power = fits.get("power", {})
    if "params" in power:
        print(f"\nPower-law exponent n = {power['params']['n']:.2f} "
              f"+/- {power['stderr']['n']:.2f}  (pure dipole: n = 3)")

    if selected:
        print(f"Selected model: {selected} (smallest leave-one-out distance error)")
    else:
        print("No model could be fitted - the data is saved, check it with compare_models.py")


# ==========================
# Ανίχνευση
# ==========================

def estimate_distance(model, params, delta, max_distance):
    """delta -> απόσταση στο [0, max]. delta <= c (πολύ μακριά) δίνει max, δηλαδή 100%."""
    return float(np.clip(models.invert(model, params, delta), 0.0, max_distance))


def temperature_warning(cal):
    """Κείμενο προειδοποίησης αν η θερμοκρασία άλλαξε πάνω από 5 °C από το baseline, αλλιώς None."""
    t_cal = cal["baseline"].get("temperature")
    t_now = read_temperature()
    if t_cal is not None and t_now is not None and abs(t_now - t_cal) > 5:
        return (f"Temperature is {t_now:.1f} C, the baseline was taken at {t_cal:.1f} C"
                " - consider a re-baseline.")
    return None


class Detector:
    """
    Μία συνεδρία ανίχνευσης: EMA του διανύσματος του πεδίου -> άνοιγμα παραθύρου, και
    καταγραφή σε CSV όσο το παράθυρο είναι ανοιχτό. Τη χρησιμοποιούν το τερματικό και το app.py.

    Κύκλος: update() κάθε DETECT_INTERVAL s, και κάθε PRINT_EVERY αναγνώσεις
    estimate() -> log() -> auto_rebaseline().
    """

    def __init__(self, name, log=True, log_every=LOG_EVERY):
        # Βαθμονόμηση: επιλεγμένο μοντέλο, παράμετροι, μέγιστο άνοιγμα, υπόβαθρο B0
        self.name = name
        self.cal = load_calibration(name)
        self.model = self.cal["selected_model"]
        self.params = self.cal["fits"][self.model]["params"]
        self.max_distance = self.cal["max_distance"]
        self.b0 = np.array(self.cal["baseline"]["vector"])

        self.smooth = read_vector()     # το EMA ξεκινάει από πραγματική μέτρηση, όχι από 0
        self.start = time.monotonic()
        self.was_open = False
        self.log_every = log_every
        self._last_row = None           # πότε γράφτηκε η τελευταία γραμμή στο CSV

        # Αυτόματο re-baseline: μόνο αν το calibration έχει μέτρηση "τέρμα ανοιχτό"
        full_open = self.cal.get("full_open")
        self.full_open_field = np.array(full_open["field"]) if full_open else None
        self.auto_message = None
        self._full_since = None         # από πότε το παράθυρο είναι συνεχώς στο 100%
        self._full_sum = np.zeros(3)    # άθροισμα των διανυσμάτων στο 100% (για μέσο όρο)
        self._full_count = 0

        # Αρχείο καταγραφής: ένα νέο CSV για κάθε συνεδρία, με όνομα την ώρα έναρξης
        self.log_path = None
        self._file = self._writer = None
        if log:
            LOG_FOLDER.mkdir(exist_ok=True)
            self.log_path = LOG_FOLDER / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            self._file = open(self.log_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["time", "elapsed_s", "delta_uT", "distance_cm", "opening_pct",
                                   "bx_uT", "by_uT", "bz_uT", "temperature_c"])

    def update(self):
        """Μία ανάγνωση του μαγνητόμετρου στο EMA (κλήση κάθε DETECT_INTERVAL s)."""
        # EMA στο διάνυσμα, όχι στο μέτρο: ο θόρυβος κάθε άξονα μειώνεται πριν αφαιρεθεί το B0
        self.smooth = (1 - DETECT_ALPHA) * self.smooth + DETECT_ALPHA * read_vector()

    def estimate(self):
        """Από το εξομαλυμένο διάνυσμα: delta (µT), απόσταση (cm), άνοιγμα (%) και αν είναι ανοιχτό."""
        delta = float(np.linalg.norm(self.smooth - self.b0))   # delta = |B - B0|
        distance = estimate_distance(self.model, self.params, delta, self.max_distance)
        percentage = 100.0 * distance / self.max_distance
        return {
            "delta": delta,
            "distance": distance,
            "percentage": percentage,
            "shown": round(percentage / DISPLAY_STEP) * DISPLAY_STEP,
            "is_open": percentage >= LOG_THRESHOLD,
        }

    def log(self, est):
        """
        Γραμμή στο CSV: τη στιγμή που ανοίγει το παράθυρο, κάθε `log_every` s όσο είναι
        ανοιχτό, και τη στιγμή που κλείνει.
        """
        now = time.monotonic()
        changed = est["is_open"] != self.was_open   # μόλις άνοιξε ή μόλις έκλεισε
        due = est["is_open"] and (self._last_row is None or now - self._last_row >= self.log_every)
        if self._writer and (changed or due):
            self._last_row = now
            self._writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                round(now - self.start, 2),
                round(est["delta"], 3), round(est["distance"], 2), round(est["percentage"], 1),
                *np.round(self.smooth, 3), read_temperature(),
            ])
            self._file.flush()   # κάθε γραμμή γράφεται αμέσως - δεν χάνεται αν κοπεί το ρεύμα
        self.was_open = est["is_open"]

    def auto_rebaseline(self, est):
        """
        Παράθυρο στο 100% για AUTO_REBASELINE_AFTER s -> θεωρείται ότι είναι στο όριό του.
        Νέο υπόβαθρο = μέσο πεδίο σε αυτό το διάστημα μείον το πεδίο του μαγνήτη στο όριο
        (από τη βαθμονόμηση). Διορθώνει την αργή ολίσθηση χωρίς να βγαίνει ο μαγνήτης.
        Επιστρέφει μήνυμα όταν ενεργεί, αλλιώς None.
        """
        if self.full_open_field is None or est["percentage"] < 100.0:
            self._full_since = None   # δεν είναι (πια) στο 100%: ο χρόνος μηδενίζεται
            return None

        now = time.monotonic()
        if self._full_since is None:
            self._full_since, self._full_sum, self._full_count = now, np.zeros(3), 0
        self._full_sum += self.smooth
        self._full_count += 1
        if now - self._full_since < AUTO_REBASELINE_AFTER:
            return None

        self._full_since = None    # το επόμενο ξαναμετράει 5 λεπτά από την αρχή
        new_b0 = self._full_sum / self._full_count - self.full_open_field   # B0 = B - πεδίο μαγνήτη
        shift = float(np.linalg.norm(new_b0 - self.b0))
        clock = datetime.now().strftime("%H:%M")

        # Μεγάλη αλλαγή = κάτι άλλο (π.χ. μετακίνηση, μέταλλο κοντά), όχι ολίσθηση: δεν διορθώνεται
        if shift > AUTO_REBASELINE_MAX_SHIFT:
            self.auto_message = (f"{clock}: automatic re-baseline skipped - change of "
                                 f"{shift:.1f} uT is too large for drift. Do a manual Re-baseline.")
            return self.auto_message

        self.b0 = new_b0
        apply_baseline(self.cal, {
            "vector": new_b0.tolist(),
            "std": None,
            "magnitude": float(np.linalg.norm(new_b0)),
            "temperature": read_temperature(),
            "time": now_iso(),
            "source": "auto (fully open)",
        })
        self.auto_message = f"{clock}: automatic re-baseline (corrected {shift:.2f} uT)"
        return self.auto_message

    def auto_countdown(self):
        """Δευτερόλεπτα που μένουν μέχρι το αυτόματο re-baseline όσο είναι στο 100%, αλλιώς None."""
        if self._full_since is None:
            return None
        return max(0.0, AUTO_REBASELINE_AFTER - (time.monotonic() - self._full_since))

    def close(self):
        """Κλείνει το αρχείο καταγραφής."""
        if self._file:
            self._file.close()
            self._file = self._writer = None


def detection_mode(name=None):
    """Ανίχνευση στο τερματικό (ή χωρίς μενού με --detect NAME), μέχρι Ctrl+C."""
    if name is None:
        name = choose_calibration()
        if name is None:
            return

    det = Detector(name)
    print(f"\nCalibration '{name}': model={det.model}, max={det.max_distance} cm, "
          f"baseline from {det.cal['baseline']['time']}")
    warning = temperature_warning(det.cal)
    if warning:
        print(f"  WARNING: {warning} (option 4)")
    if det.full_open_field is None:
        print("  Automatic re-baseline: off (no fully-open measurement in this calibration)")
    else:
        print(f"  Automatic re-baseline: on (after {AUTO_REBASELINE_AFTER // 60} min at 100%)")
    print(f"Logging to {det.log_path}")
    print("\nDetection started (Ctrl+C to stop)\n")

    try:
        while True:
            for _ in range(PRINT_EVERY):
                time.sleep(DETECT_INTERVAL)
                det.update()
            est = det.estimate()
            det.log(est)
            print(f"Delta:{est['delta']:8.2f} uT   Distance:{est['distance']:6.1f} cm   "
                  f"Opening:{est['shown']:4.0f}%{'   REC' if est['is_open'] else ''}")
            message = det.auto_rebaseline(est)
            if message:
                print(f"  {message}")
    finally:
        det.close()   # το CSV κλείνει σωστά και με Ctrl+C


# ==========================
# Ζωντανή προβολή & re-baseline
# ==========================

def live_mode():
    """Ζωντανή προβολή της μεταβολής του πεδίου ανά άξονα (για έλεγχο της τοποθέτησης)."""
    b0 = np.array(measure_baseline(samples=60)["vector"])
    print("\nPut the magnet back. Live view (Ctrl+C to stop)\n")

    while True:
        d = read_vector() - b0
        print(f"Delta={np.linalg.norm(d):8.2f} uT   "
              f"dx={d[0]:8.2f}  dy={d[1]:8.2f}  dz={d[2]:8.2f}")
        time.sleep(0.2)


def apply_baseline(cal, baseline):
    """
    Αντικαθιστά το υπόβαθρο μιας βαθμονόμησης (το παλιό πάει στο ιστορικό).
    Επιστρέφει τη μετατόπιση σε µT.
    """
    shift = float(np.linalg.norm(np.array(baseline["vector"]) - np.array(cal["baseline"]["vector"])))
    cal.setdefault("baseline_history", []).append(cal["baseline"])
    del cal["baseline_history"][:-100]   # κρατάει τα τελευταία 100 (τα αυτόματα γίνονται συχνά)
    cal["baseline"] = baseline
    save_calibration(cal)
    return shift


def rebaseline_mode():
    """Νέο υπόβαθρο για υπάρχουσα βαθμονόμηση, π.χ. μετά από μετακίνηση/περιστροφή του Pi."""
    name = choose_calibration()
    if name is None:
        return

    shift = apply_baseline(load_calibration(name), measure_baseline())
    print(f"\nBaseline of '{name}' updated (moved by {shift:.2f} uT). "
          "Put the magnet back in its place.")


# ==========================
# Κύριο μενού
# ==========================

MENU = """
============================

Window Detection System

1. Live Magnetometer
2. Calibration + Curve Fitting
3. Detection Mode
4. Re-baseline (after moving the Pi)
0. Exit

============================
"""


def run_menu():
    """Κύριο μενού του τερματικού· Ctrl+C μέσα σε μια λειτουργία γυρνάει εδώ."""
    actions = {"1": live_mode, "2": calibration_mode, "3": detection_mode, "4": rebaseline_mode}

    while True:
        print(MENU)
        choice = input("Select: ").strip()
        if choice == "0":
            break
        action = actions.get(choice)
        if action is None:
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\n(stopped - back to menu)")


def main():
    """Σημείο εκκίνησης: μενού, ή απευθείας ανίχνευση με --detect NAME."""
    parser = argparse.ArgumentParser(
        description="Window opening detection with the Sense HAT magnetometer.")
    parser.add_argument("--detect", metavar="NAME",
                        help="start detection directly with calibration NAME, or 'latest' for "
                             "the newest calibration (no menu)")
    args = parser.parse_args()

    name = args.detect
    if name == "latest":
        name = latest_calibration()
        if name is None:
            parser.error(f"no usable calibration in {CALIBRATION_FOLDER}")
    elif name and name not in list_calibrations()[0]:
        parser.error(f"no usable calibration named '{name}' in {CALIBRATION_FOLDER}")

    sense.clear()   # LED matrix σβηστό: τα ρεύματά του επηρεάζουν το μαγνητόμετρο
    try:
        if name:
            detection_mode(name)
        else:
            run_menu()
    except KeyboardInterrupt:
        print()
    finally:
        sense.clear()


if __name__ == "__main__":
    main()
