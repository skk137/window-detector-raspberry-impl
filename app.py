"""
Σύστημα ανίχνευσης ανοίγματος παραθύρου - εφαρμογή web (Streamlit). Τρέχει στο Raspberry Pi.

    streamlit run app.py

Μετά άνοιξε το http://localhost:8501 στο Pi (π.χ. μέσω Pi Connect), ή το
http://raspberrypi.local:8501 από υπολογιστή/κινητό στο ίδιο δίκτυο.

Χρησιμοποιεί τον ίδιο κώδικα και τους ίδιους φακέλους calibrations/ και logs/ με το
window_detector.py. Μην τρέχουν και τα δύο ταυτόχρονα - μοιράζονται τον αισθητήρα.
Χρειάζεται Streamlit 1.37 ή νεότερο (st.fragment, st.navigation).

Σελίδες: Home, Live, Calibration, Re-baseline, Logs. Το Streamlit ξανατρέχει τη
συνάρτηση της σελίδας σε κάθε αλληλεπίδραση· ό,τι πρέπει να κρατηθεί ανάμεσα στις
εκτελέσεις (Detector, βήματα του οδηγού βαθμονόμησης) μπαίνει στο st.session_state.
"""

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import window_detector as wd


UPDATE_EVERY = wd.DETECT_INTERVAL * wd.PRINT_EVERY   # 0.5 s, όπως στο terminal
HISTORY_SECONDS = 60                                 # πόσα δευτερόλεπτα δείχνει το γράφημα
CHART_COLOR = "#2a78d6"

# Κενός πίνακας-πρότυπο για βαθμονόμηση από πίνακα (6 σημεία = το ελάχιστο)
TEMPLATE = pd.DataFrame({"distance": [0.0, 3.0, 6.0, 9.0, 12.0, 15.0],
                         "delta": [float("nan")] * 6})
COLUMN_LABELS = {"distance": "distance (cm)", "delta": "delta (µT)",
                 "delta_std": "delta_std (µT)", "n_samples": "n_samples"}

# Κείμενα βοήθειας που εμφανίζονται σε περισσότερα από ένα σημεία
MAX_OPENING_HINT = ("set the maximum window opening to about 35-40 cm at most - the smaller, "
                    "the more accurate. How far it can reach depends on the magnet: a bigger / "
                    "stronger magnet reaches further. Anything beyond the maximum you set is "
                    "shown as 100% open.")
POINTS_HINT = ("use about one point every 5 cm - e.g. 9 points for 0-40 cm - "
               f"and at least {wd.MIN_POINTS}.")
OLD_DATA_WARNING = ("The delta values must come from this system (vector field change), with "
                    "the Pi and the magnet mounted the same way as now. Measurements from the "
                    "original script were calculated differently and would give wrong openings.")


st.set_page_config(page_title="Window Detection", page_icon="🪟")


@st.cache_resource
def init_sensor():
    """Μία φορά ανά εκκίνηση του server (όχι σε κάθε επανεκτέλεση της σελίδας)."""
    wd.sense.clear()   # LED matrix σβηστό: τα ρεύματά του επηρεάζουν το μαγνητόμετρο
    return True


def stop_live():
    """Σταματά μια ανίχνευση που τρέχει (κλείνει το αρχείο καταγραφής της)."""
    # Καλείται πριν από κάθε μέτρηση βαθμονόμησης/υποβάθρου, ώστε να μη διαβάζουν δύο μαζί
    det = st.session_state.get("detector")
    if det is not None:
        det.close()
        st.session_state.detector = None


def sensor_status():
    """Κείμενο κατάστασης του αισθητήρα για την αρχική σελίδα."""
    try:
        field = np.linalg.norm(wd.read_vector())
    except OSError:
        return "not responding"
    return f"OK ({field:.0f} µT)"


def round2(value):
    """Στρογγυλοποίηση σε 2 δεκαδικά, χωρίς να σκάει σε None."""
    return round(value, 2) if value is not None else None


def model_label(name):
    """'name · model: power' - εμφανίζεται όπου επιλέγεται βαθμονόμηση."""
    return f"{name}  ·  model: {wd.load_calibration(name)['selected_model']}"


def no_calibration_message():
    """Μήνυμα όταν δεν υπάρχει καμία χρησιμοποιήσιμη βαθμονόμηση (και γιατί)."""
    _, skipped = wd.list_calibrations()
    message = "No usable calibration yet - make one in **Calibration**."
    if skipped:
        message += (f" Found {len(skipped)} older file(s) ({', '.join(skipped)}): made with the "
                    "original script, or not finished - they can't be used.")
    st.info(message)


def name_error(name, overwrite):
    """Κείμενο σφάλματος για άκυρο ή ήδη χρησιμοποιημένο όνομα βαθμονόμησης, αλλιώς None."""
    if not wd.valid_name(name):
        return "Use only latin letters, numbers, _ and - (no spaces)."
    if wd.calibration_path(name).exists() and not overwrite:
        return f"'{name}' already exists - choose another name or tick Overwrite."
    return None


def points_table(cal):
    """Τα σημεία μιας βαθμονόμησης ως πίνακας (ίδιες στήλες με το CSV της)."""
    return pd.DataFrame([{c: p[c] for c in wd.POINT_COLUMNS} for p in cal["points"]],
                        columns=list(wd.POINT_COLUMNS))


def normalize_table(df):
    """Κρατάει τις γνωστές στήλες (ονόματα με μικρά)· οι distance και delta είναι υποχρεωτικές."""
    df = df.rename(columns=lambda c: str(c).strip().lower())
    missing = {"distance", "delta"} - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s): {', '.join(sorted(missing))}")
    return df[[c for c in wd.POINT_COLUMNS if c in df.columns]]


def points_from_table(df):
    """Πίνακας -> σημεία βαθμονόμησης. ValueError όσο ο πίνακας δεν είναι ακόμα έτοιμος."""
    # Κενά ή μη αριθμητικά κελιά -> NaN, και οι ελλιπείς γραμμές αγνοούνται
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["distance", "delta"])
    df = df.sort_values("distance")

    if (df["distance"] < 0).any():
        raise ValueError("negative distance")
    if df["distance"].duplicated().any():
        raise ValueError("the same distance appears more than once")
    if len(df) < wd.MIN_POINTS:
        raise ValueError(f"{len(df)} complete rows - at least {wd.MIN_POINTS} rows with "
                         "distance and delta are needed")

    def optional(row, col, default):
        return row[col] if col in df.columns and pd.notna(row[col]) else default

    # Χωρίς std/N: μένει μόνο ο θόρυβος αισθητήρα + σφάλμα θέσης στα βάρη (βλ. models.point_sigmas)
    return [{
        "distance": float(row["distance"]),
        "delta": float(row["delta"]),
        "delta_std": float(optional(row, "delta_std", 0.0)),
        "n_samples": int(optional(row, "n_samples", 1)),
    } for _, row in df.iterrows()]


# ==========================
# Αρχική σελίδα (Home)
# ==========================

def home_page():
    """Τι κάνει η εφαρμογή, κατάσταση του συστήματος και σύνδεσμοι προς τις σελίδες."""
    st.title("Window Detection System")
    st.write("Measures how far a window is open, from the field of a magnet on the window, "
             "using the Sense HAT magnetometer on the Raspberry Pi.")

    usable, _ = wd.list_calibrations()
    col1, col2, col3 = st.columns(3)
    col1.metric("Calibrations", len(usable))
    col2.metric("Sensor", sensor_status())
    col3.metric("Live", "running" if st.session_state.get("detector") else "stopped")

    st.subheader("What you can do")

    st.page_link(live, label="Live - see the window opening in real time", icon=":material/sensors:")
    st.caption("Choose a calibration and see the opening in %, the distance in cm and the field "
               "change in µT, with a chart of the last minute. While the window is open, the "
               "measurements are saved to a CSV file in the logs folder.")

    st.page_link(logs, label="Logs - look at saved measurements", icon=":material/history:")
    st.caption("Pick a log file and see when and how much the window was open, as a chart and "
               "a table, or download it.")

    st.page_link(calibration, label="Calibration - teach the system a new window",
                 icon=":material/straighten:")
    st.caption("Measure step by step with the sensor (about 2 minutes for 10 points), or type / "
               "upload the measurements as a table. Every calibration is also saved as a CSV "
               "file, and shows which model was selected.")

    st.page_link(rebaseline, label="Re-baseline - after moving the Raspberry Pi",
                 icon=":material/restart_alt:")
    st.caption("If you move or rotate the Pi, re-measure the background field (about 12 seconds) "
               "instead of doing a new calibration.")

    if not usable:
        st.info("No calibration yet - start with **Calibration**.")


# ==========================
# Ζωντανή ανίχνευση (Live)
# ==========================

def live_page():
    """Επιλογή βαθμονόμησης και Start· όσο τρέχει, η ζωντανή προβολή (live_view)."""
    st.title("Live")

    usable, _ = wd.list_calibrations()
    if not usable:
        no_calibration_message()
        return

    # Χωρίς ενεργή ανίχνευση: φόρμα έναρξης
    det = st.session_state.get("detector")
    if det is None:
        name = st.selectbox("Calibration file to use", usable, format_func=model_label)
        log = st.toggle("Save measurements to CSV (only while the window is open)", value=True,
                        help=f"One row every {wd.LOG_EVERY} s, plus the moments the window "
                             "opens and closes.")
        if st.button("Start", type="primary"):
            st.session_state.detector = wd.Detector(name, log=log)
            st.session_state.history = []
            st.rerun()
        return

    # Ενεργή ανίχνευση: στοιχεία της βαθμονόμησης, προειδοποιήσεις και ζωντανή προβολή
    col1, col2 = st.columns([4, 1])
    col1.caption(f"Calibration **{det.name}** · model: {det.model} · "
                 f"max opening: {det.max_distance:g} cm")
    if col2.button("Stop"):
        stop_live()
        st.rerun()

    warning = wd.temperature_warning(det.cal)
    if warning:
        st.warning(warning)
    if det.full_open_field is None:
        st.caption("Automatic re-baseline: off - add a fully-open measurement in Re-baseline.")
    else:
        st.caption(f"Automatic re-baseline: on - after {wd.AUTO_REBASELINE_AFTER // 60} min at 100%.")

    live_view()


@st.fragment(run_every=UPDATE_EVERY)
def live_view():
    """
    Ξανατρέχει μόνο αυτό το κομμάτι κάθε UPDATE_EVERY s (όχι όλη η σελίδα):
    αναγνώσεις, εκτίμηση, καταγραφή, αυτόματο re-baseline και ενημέρωση της οθόνης.
    """
    det = st.session_state.get("detector")
    if det is None:
        return

    for _ in range(wd.PRINT_EVERY):        # ίδιος ρυθμός ανάγνωσης με το terminal
        time.sleep(wd.DETECT_INTERVAL)
        det.update()
    est = det.estimate()
    det.log(est)
    det.auto_rebaseline(est)

    # Ιστορικό για το γράφημα: μόνο το τελευταίο λεπτό
    history = st.session_state.history
    history.append({"seconds": round(time.monotonic() - det.start, 1),
                    "opening": round(est["percentage"], 1)})
    del history[:-int(HISTORY_SECONDS / UPDATE_EVERY)]

    st.metric("Opening", f"{est['shown']:.0f} %")
    st.progress(min(max(est["percentage"] / 100, 0.0), 1.0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Distance", f"{est['distance']:.1f} cm")
    col2.metric("Field change", f"{est['delta']:.2f} µT")
    col3.metric("Recording", "REC" if det.log_path and est["is_open"] else "—")

    # Μπάρα αντίστροφης μέτρησης όσο το παράθυρο είναι στο 100%
    remaining = det.auto_countdown()
    if remaining is not None:
        minutes, seconds = divmod(int(remaining), 60)
        st.progress(1 - remaining / wd.AUTO_REBASELINE_AFTER,
                    text=f"Fully open - automatic re-baseline in {minutes}:{seconds:02d}")

    chart = (
        alt.Chart(pd.DataFrame(history))
        .mark_line(color=CHART_COLOR, strokeWidth=2)
        .encode(
            x=alt.X("seconds:Q", title="Time (s)"),
            y=alt.Y("opening:Q", title="Opening (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["seconds", "opening"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart)

    if det.log_path:
        st.caption(f"Log file: logs/{det.log_path.name}")
    if det.auto_message:
        st.caption(det.auto_message)


# ==========================
# Βαθμονόμηση (Calibration)
# ==========================

def calibration_page():
    """
    Δύο τρόποι: οδηγός με τον αισθητήρα (βήματα baseline -> points -> full_open -> done,
    η κατάσταση στο session_state.wizard) ή εισαγωγή σημείων από πίνακα/CSV.
    """
    st.title("Calibration")

    wiz = st.session_state.get("wizard")
    if wiz is None:
        mode = st.radio("How do you want to create it?",
                        ["Measure with the sensor", "Type or upload a table"], horizontal=True)
        if mode == "Measure with the sensor":
            calibration_setup()
        else:
            calibration_table()
        saved_calibrations()
        st.divider()
        st.caption(f":bulb: **Tips:** {MAX_OPENING_HINT[0].upper() + MAX_OPENING_HINT[1:]}  \n"
                   f"For the points, {POINTS_HINT}")
    elif wiz["stage"] == "baseline":
        calibration_baseline(wiz)
    elif wiz["stage"] == "points":
        calibration_points(wiz)
    elif wiz["stage"] == "full_open":
        calibration_full_open(wiz)
    else:
        calibration_done(wiz)


def calibration_setup():
    """Φόρμα του οδηγού: όνομα, μέγιστο άνοιγμα και αριθμός σημείων."""
    st.write("Teach the system a new window. Do it at the place where you will measure, "
             "with the Pi and the magnet already mounted.")

    with st.form("calibration_setup"):
        name = st.text_input("Name", placeholder="e.g. kitchen_window",
                             help="Latin letters, numbers, _ and - only")
        max_distance = st.number_input("Maximum window opening (cm)", min_value=0.5,
                                       value=15.0, step=0.5,
                                       help=MAX_OPENING_HINT[0].upper() + MAX_OPENING_HINT[1:])
        points = st.number_input("Number of points", min_value=wd.MIN_POINTS, value=9, step=1,
                                 help=POINTS_HINT[0].upper() + POINTS_HINT[1:])
        overwrite = st.checkbox("Overwrite if a calibration with this name exists")
        submitted = st.form_submit_button("Next", type="primary")

    if not submitted:
        return

    name = name.strip()
    error = name_error(name, overwrite)
    if error:
        st.error(error)
        return

    # Ξεκινά ο οδηγός: πρώτο βήμα η μέτρηση του υποβάθρου
    st.session_state.wizard = {
        "stage": "baseline",
        "name": name,
        "max_distance": float(max_distance),
        "distances": wd.calibration_distances(float(max_distance), int(points)),
    }
    st.rerun()


def calibration_table():
    """Βαθμονόμηση από πίνακα: πρότυπο, σημεία υπάρχουσας βαθμονόμησης ή δικό σου CSV."""
    st.write("Fill in the table here, or start from a template / an existing calibration / "
             "your own CSV, then create the calibration.")

    usable, _ = wd.list_calibrations()
    sources = ["Empty template (6 points)"] + [f"Points of {n}" for n in usable]

    col1, col2 = st.columns(2)
    source = col1.selectbox("Start from", sources)
    uploaded = col2.file_uploader("...or upload a CSV (replaces the table)", type="csv")
    st.download_button("Download the empty template (CSV)", TEMPLATE.to_csv(index=False),
                       file_name="calibration_template.csv", mime="text/csv")

    # Πηγή του πίνακα: ανεβασμένο CSV, κενό πρότυπο ή σημεία υπάρχουσας βαθμονόμησης
    source_cal = None
    if uploaded is not None:
        try:
            table = normalize_table(pd.read_csv(uploaded))
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            st.error(f"Can't use this file: {e}")
            return
        table_id = f"upload_{uploaded.name}_{uploaded.size}"
    elif source == sources[0]:
        table, table_id = TEMPLATE, "template"
    else:
        source_cal = wd.load_calibration(usable[sources.index(source) - 1])
        table, table_id = points_table(source_cal), f"cal_{source_cal['name']}"

    # Νέο key ανά πηγή, ώστε ο πίνακας να ξαναγεμίζει όταν αλλάζει η πηγή
    edited = st.data_editor(
        table, num_rows="dynamic", hide_index=True, key=f"editor_{table_id}",
        column_config={c: st.column_config.NumberColumn(label)
                       for c, label in COLUMN_LABELS.items() if c in table.columns},
    )
    st.download_button("Download this table (CSV)", edited.to_csv(index=False),
                       file_name="calibration_points.csv", mime="text/csv")
    st.caption("Columns: distance (cm) and delta (µT); delta_std and n_samples are optional. "
               "Add or delete rows at the bottom of the table.")

    try:
        points = points_from_table(edited)
    except ValueError as e:
        st.info(f"Table not ready yet: {e}.")
        return

    st.warning(OLD_DATA_WARNING)

    with st.form("calibration_table"):
        name = st.text_input("Name", value=source_cal["name"] if source_cal else "",
                             placeholder="e.g. kitchen_window",
                             help="Latin letters, numbers, _ and - only")
        max_distance = st.number_input(
            "Maximum window opening (cm)", min_value=0.5, step=0.5,
            value=source_cal["max_distance"] if source_cal else max(0.5, points[-1]["distance"]),
            help=MAX_OPENING_HINT[0].upper() + MAX_OPENING_HINT[1:])
        overwrite = st.checkbox("Overwrite if a calibration with this name exists")
        keep_background = False
        if source_cal:
            keep_background = st.checkbox(
                f"Keep the background field stored in '{source_cal['name']}' (no new measurement)",
                value=True)
        st.write("Otherwise the background field is measured now: **remove the magnet from the "
                 "sensor**, then press the button (about 12 seconds).")
        submitted = st.form_submit_button("Create calibration", type="primary")

    if not submitted:
        return

    name = name.strip()
    error = name_error(name, overwrite)
    if error:
        st.error(error)
        return

    # Υπόβαθρο: από την αρχική βαθμονόμηση ή νέα μέτρηση τώρα
    if keep_background:
        baseline = source_cal["baseline"]
    else:
        stop_live()
        with st.spinner("Measuring the background field..."):
            baseline = wd.take_baseline()

    cal = wd.new_calibration(name, float(max_distance), baseline)
    cal["points"] = points
    cal["source"] = "table" if uploaded is None else f"table from {uploaded.name}"
    if keep_background and source_cal.get("full_open"):
        cal["full_open"] = source_cal["full_open"]   # ίδια τοποθέτηση -> ίδιο πεδίο στο στοπ
    with st.spinner("Fitting the models..."):
        wd.run_fit(cal)
    wd.save_calibration(cal)

    st.session_state.wizard = {"stage": "done", "cal": cal}
    st.rerun()


def saved_calibrations():
    """Λίστα των αποθηκευμένων βαθμονομήσεων με μοντέλο, σφάλμα και λήψη του CSV τους."""
    usable, _ = wd.list_calibrations()
    if not usable:
        return

    st.divider()
    st.subheader("Saved calibrations")
    for name in usable:
        cal = wd.load_calibration(name)
        model = cal["selected_model"]
        loo = round2(cal["fits"][model]["loo_rmse_cm"])

        col1, col2, col3 = st.columns([3, 3, 2])
        col1.markdown(f"**{name}**")
        col1.caption(f"{len(cal['points'])} points · 0-{cal['max_distance']:g} cm · "
                     f"{cal['created'][:10]}")
        col2.markdown(f"model: **{model}**")
        col2.caption(f"leave-one-out error: {loo if loo is not None else '-'} cm · "
                     f"auto re-baseline: {'on' if cal.get('full_open') else 'off'}")
        col3.download_button("CSV", points_table(cal).to_csv(index=False),
                             file_name=f"{name}.csv", mime="text/csv", key=f"csv_{name}")


def calibration_baseline(wiz):
    """Βήμα 1 του οδηγού: μέτρηση του υποβάθρου με τον μαγνήτη απομακρυσμένο."""
    st.subheader("Step 1 of 2 - Background field")
    st.write("Remove the magnet from the sensor - take it off, or move it far away - "
             "**the same way every time**. Then press Measure (about 12 seconds).")

    col1, col2 = st.columns(2)
    if col1.button("Measure background", type="primary"):
        stop_live()
        with st.spinner("Measuring the background field..."):
            baseline = wd.take_baseline()
        wiz["cal"] = wd.new_calibration(wiz["name"], wiz["max_distance"], baseline)
        wiz["stage"] = "points"
        st.rerun()
    if col2.button("Cancel"):
        st.session_state.wizard = None
        st.rerun()


def calibration_points(wiz):
    """Βήμα 2 του οδηγού: ένα σημείο σε κάθε εκτέλεση, μέχρι να μετρηθούν όλες οι αποστάσεις."""
    cal = wiz["cal"]
    distances = wiz["distances"]
    i = len(cal["points"])   # επόμενο σημείο = όσα έχουν ήδη μετρηθεί

    st.subheader(f"Step 2 of 2 - Point {i + 1} of {len(distances)}")
    st.progress(i / len(distances))
    if i == 0:
        st.info("Put the magnet back in its place.")
    st.markdown(f"### Move the window to **{distances[i]:g} cm**")
    st.caption("Then press Measure (about 5 seconds) and keep the window still.")

    col1, col2 = st.columns(2)
    # key ανά σημείο: ένα διπλό κλικ δεν μετράει κατά λάθος και το επόμενο σημείο
    if col1.button("Measure", type="primary", key=f"measure_{i}"):
        stop_live()
        with st.spinner(f"Measuring at {distances[i]:g} cm..."):
            cal["points"].append(wd.measure_point(distances[i], np.array(cal["baseline"]["vector"])))
            wd.save_calibration(cal)   # σώζεται μετά από κάθε σημείο
        if len(cal["points"]) == len(distances):
            wiz["stage"] = "full_open"
        st.rerun()
    if col2.button("Cancel (points so far stay saved)"):
        st.session_state.wizard = None
        st.rerun()

    if cal["points"]:
        st.dataframe(points_table(cal).rename(columns=COLUMN_LABELS), hide_index=True)


def calibration_full_open(wiz):
    """Τελευταίο βήμα: μέτρηση «τέρμα ανοιχτό» για το αυτόματο re-baseline, και fit."""
    cal = wiz["cal"]
    max_text = f"{cal['max_distance']:g} cm"

    st.subheader("Last step - Fully open")
    st.caption("With this, Live corrects slow drift by itself: when the window stays at 100% for "
               f"{wd.AUTO_REBASELINE_AFTER // 60} minutes it re-baselines automatically, "
               "without removing the magnet.")

    # Αν το παράθυρο δεν ανοίγει πέρα από το μέγιστο, το τελευταίο σημείο είναι το όριό του
    opens_further = st.radio(f"Does the window open further than {max_text}?",
                             ["Yes", f"No - {max_text} is its stop"], horizontal=True) == "Yes"
    if opens_further:
        st.write("Open the window **fully, against its stop**, and press Measure (about 7 seconds).")
    else:
        st.write(f"The last point ({max_text}) is used as fully open - no new measurement.")

    col1, col2 = st.columns(2)
    if opens_further:
        confirm = col1.button("Measure fully open", type="primary", key="measure_full_open")
    else:
        confirm = col1.button(f"Window max is {max_text}", type="primary", key="use_last_point")
    skip = col2.button("Skip (no automatic re-baseline)")
    if not (confirm or skip):
        return

    if confirm and opens_further:
        stop_live()
        with st.spinner("Measuring fully open..."):
            wiz["full_open_warning"] = wd.measure_full_open(cal)
    elif confirm:
        wd.full_open_from_last_point(cal)
    with st.spinner("Fitting the models..."):
        wd.run_fit(cal)
        wd.save_calibration(cal)
    wiz["stage"] = "done"
    st.rerun()


def calibration_done(wiz):
    """Αποτελέσματα: σύγκριση μοντέλων, επιλεγμένο μοντέλο και λήψη των μετρήσεων."""
    cal = wiz["cal"]
    selected = cal["selected_model"]

    if selected:
        st.success(f"Calibration **{cal['name']}** saved. Selected model: **{selected}**.")
    else:
        st.error("No model could be fitted. The measurements are saved - "
                 "check them with compare_models.py.")

    rows = []
    for name, fit in cal["fits"].items():
        if "error" in fit:
            rows.append({"model": name, "note": "fit failed"})
            continue
        rows.append({
            "model": name + ("  (selected)" if name == selected else ""),
            "RMSE (µT)": round2(fit["rmse_uT"]),
            "error (cm)": round2(fit["rmse_cm"]),
            "leave-one-out (cm)": round2(fit["loo_rmse_cm"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Detection uses the model with the smallest leave-one-out error: how many cm "
               "off it is on points it did not see while fitting.")

    power = cal["fits"].get("power", {})
    if "params" in power:
        st.caption(f"Power-law exponent n = {power['params']['n']:.2f} "
                   f"± {power['stderr']['n']:.2f} (pure dipole: n = 3)")

    if wiz.get("full_open_warning"):
        st.warning(wiz["full_open_warning"])
    if cal.get("full_open"):
        st.info(f"Automatic re-baseline: **on** (after {wd.AUTO_REBASELINE_AFTER // 60} min at 100%).")
    else:
        st.info("Automatic re-baseline: **off**. You can add it later in Re-baseline.")

    st.download_button("Download measurements (CSV)", points_table(cal).to_csv(index=False),
                       file_name=f"{cal['name']}.csv", mime="text/csv")
    st.caption(f"Also saved next to the calibration: calibrations/{cal['name']}.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.page_link(live, label="Go to Live", icon=":material/sensors:")
    if col2.button("New calibration"):
        st.session_state.wizard = None
        st.rerun()


# ==========================
# Νέο υπόβαθρο (Re-baseline)
# ==========================

def rebaseline_page():
    """Νέο υπόβαθρο για υπάρχουσα βαθμονόμηση, και μέτρηση «τέρμα ανοιχτό» αν λείπει."""
    st.title("Re-baseline")
    st.write("After moving or rotating the Raspberry Pi, the background (Earth's) field that "
             "the sensor sees changes. Re-measure it here (about 12 seconds) instead of doing "
             "a new calibration - as long as the magnet and the Pi are mounted the same way.")

    usable, _ = wd.list_calibrations()
    if not usable:
        no_calibration_message()
        return

    name = st.selectbox("Calibration", usable, format_func=model_label)
    st.write("Remove the magnet from the sensor, **the same way as during calibration**, "
             "then press the button.")

    if st.button("Measure new background", type="primary"):
        stop_live()
        with st.spinner("Measuring the background field..."):
            shift = wd.apply_baseline(wd.load_calibration(name), wd.take_baseline())
        st.success(f"Background of **{name}** updated (changed by {shift:.2f} µT). "
                   "Put the magnet back in its place.")

    st.divider()
    st.subheader("Fully-open measurement (for automatic re-baseline)")
    cal = wd.load_calibration(name)
    status = "on" if cal.get("full_open") else "off"
    st.write(f"Automatic re-baseline for **{name}**: **{status}**. With it, Live re-baselines by "
             f"itself when the window stays at 100% for {wd.AUTO_REBASELINE_AFTER // 60} minutes "
             "- no need to remove the magnet.")
    st.caption("Do it **right after** a Re-baseline (button above): put the magnet back, open "
               "the window fully against its stop, then press the button (about 7 seconds).")

    # Το πεδίο του μαγνήτη στο όριο μετριέται ως προς το τρέχον υπόβαθρο - γι' αυτό αμέσως μετά
    if st.button("Measure fully open"):
        stop_live()
        with st.spinner("Measuring fully open..."):
            warning = wd.measure_full_open(cal)
            wd.save_calibration(cal)
        if warning:
            st.warning(warning)
        st.success(f"Saved - automatic re-baseline is on for **{name}**.")


# ==========================
# Αρχεία καταγραφής (Logs)
# ==========================

def logs_page():
    """Προβολή ενός αρχείου καταγραφής: σύνοψη, γράφημα ανοίγματος και πίνακας."""
    st.title("Logs")
    st.write("The measurements saved by Live - one file each time you press Start.")

    # Πιο πρόσφατο αρχείο πρώτο
    files = sorted(wd.LOG_FOLDER.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("No logs yet - start Live with 'Save measurements to CSV' on.")
        return

    path = st.selectbox("Log file", files,
                        format_func=lambda p: f"{p.name}  ·  {p.stat().st_size / 1024:.0f} KB")
    st.download_button("Download this log (CSV)", path.read_bytes(), file_name=path.name,
                       mime="text/csv")

    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        st.error(f"Can't read this file: {e}")
        return
    if df.empty:
        st.info("No rows - the window was not opened while this log was recording.")
        return

    df["time"] = pd.to_datetime(df["time"])
    closed = df["opening_pct"] < wd.LOG_THRESHOLD
    # Κάθε άνοιγμα ξεχωριστή γραμμή στο γράφημα, ώστε να μην ενώνονται τα κενά όσο ήταν κλειστό
    df["opening_no"] = closed.shift(fill_value=False).cumsum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Openings", df.loc[~closed, "opening_no"].nunique())
    col2.metric("Max opening", f"{df['opening_pct'].max():.0f} %")
    col3.metric("Rows", len(df))
    st.caption(f"From {df['time'].iloc[0]:%Y-%m-%d %H:%M:%S} to {df['time'].iloc[-1]:%Y-%m-%d %H:%M:%S}")

    chart = (
        alt.Chart(df)
        .mark_line(color=CHART_COLOR, strokeWidth=2, point=True)
        .encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("opening_pct:Q", title="Opening (%)", scale=alt.Scale(domain=[0, 100])),
            detail="opening_no:N",
            tooltip=[alt.Tooltip("time:T", format="%H:%M:%S"), "opening_pct", "distance_cm",
                     "delta_uT", "temperature_c"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart)
    st.dataframe(df.drop(columns="opening_no"), hide_index=True)


# ==========================
# Πλοήγηση
# ==========================

init_sensor()

# Οι σελίδες ορίζονται εδώ και χρησιμοποιούνται και στα st.page_link παραπάνω
home = st.Page(home_page, title="Home", icon=":material/home:", default=True)
live = st.Page(live_page, title="Live", icon=":material/sensors:", url_path="live")
calibration = st.Page(calibration_page, title="Calibration", icon=":material/straighten:",
                      url_path="calibration")
rebaseline = st.Page(rebaseline_page, title="Re-baseline", icon=":material/restart_alt:",
                     url_path="rebaseline")
logs = st.Page(logs_page, title="Logs", icon=":material/history:", url_path="logs")

st.navigation([home, live, calibration, rebaseline, logs]).run()
