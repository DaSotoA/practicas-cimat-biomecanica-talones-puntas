#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluacion final del Plan de Trabajo (Tabla 2: Indicadores cuantitativos de exito)
==================================================================================
Una sola herramienta (PC o Jetson) con un subcomando por cada indicador:

  clasificadores  OE4 / A2.2 / A2.3  Compara RandomForest, SVM-RBF, XGBoost y MLP con validacion
                                      cruzada independiente del sujeto (Leave-One-Subject-Out),
                                      F1 macro, MCC, peso en KB y latencia. Guarda exercise_clf.joblib,
                                      matriz de confusion e importancia por permutacion (PNG).
  camara          OE1 / A1.3          Compara las vistas frontal / obl30 / obl45 / lateral (visibilidad
                                      de cadera-rodilla-tobillo y error angular en postura neutra).
  hardware        OE1 / A1.1 / A3.3   FPS sostenidos, latencia, CPU, RAM, temperatura, GPU.
  conteo          OE5 / A3.4          MAE, % de series con error <= 1 rep, ICC y Bland-Altman del
                                      conteo automatico vs. conteo manual (2 observadores).
  fases           OE4 / A2.3          F1 de segmentacion de fases con tolerancia +-0.2 s.
  fp16            A3.1                Degradacion del F1 al pasar de FP32 a FP16 (<= 2 puntos).
  tabla2          Cierre              Junta todo en resultados/tabla2_indicadores.(md|csv).
  plantillas      A1.3 / A3.4         Crea hojas CSV en blanco (conteo manual, fases manuales).

Todos los resultados se escriben en ./resultados (cambia con --out-dir). Ver GUIA_PASO_A_PASO.md.
"""
import argparse
import csv
import glob
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

TARGETS = {
    "mae_conteo":   ("Error absoluto medio de conteo por serie", "<=", 0.5),
    "pct_err_le1":  ("Series contadas con error <= 1 repeticion", ">=", 90.0),
    "f1_macro":     ("F1 macro de clasificacion del ejercicio", ">=", 0.90),
    "icc":          ("Concordancia del rango de movimiento / ciclo (CCI)", ">=", 0.75),
    "fps":          ("Tasa de procesamiento sostenida (cuadros/s)", ">=", 8.0),
    "fps_floor":    ("Piso de tasa de procesamiento (cuadros/s, p5)", ">=", 5.0),
    "peso_kb":      ("Huella del clasificador desplegado (KB)", "<=", 300.0),
    "delta_f1_fp16": ("Degradacion por cuantizacion / iluminacion (puntos de F1)", "<=", 2.0),
}
CLASS_ORDER = ["TALONES", "PUNTAS", "REST", "TRANSITION", "STS", "ABD_I", "ABD_D"]
FEAT_NAMES = [
    "foot_pitch_deg",
    "tibiotarsal_deg",
    "heel_toe_dy_norm",
    "foot_omega_dps",
    "knee_flexion",
    "trunk_lean",
]
WINDOW = 24


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d)
    return d


def save_metrics(out_dir, name, data):
    ensure_dir(out_dir)
    p = os.path.join(out_dir, "metrics_%s.json" % name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("[OK] metricas guardadas en %s" % p)


def load_metrics(out_dir, name):
    p = os.path.join(out_dir, "metrics_%s.json" % name)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else ([p] if os.path.isfile(p) else []))
    if not out:
        sys.exit("No se encontraron archivos para: %s" % ", ".join(patterns))
    return out


def fnum(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_csv_dicts(path):
    with io.open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("sep=")]
    if not lines:
        return []
    delim = ";" if lines[0].count(";") > lines[0].count(",") else ","
    return list(csv.DictReader(lines, delimiter=delim))


def icc_2_1(data):
    """ICC(2,1): efectos aleatorios de dos vias, acuerdo absoluto. data = matriz n x k."""
    x = np.asarray(data, float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = x.mean()
    ssr = k * ((x.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((x.mean(axis=0) - gm) ** 2).sum()
    sst = ((x - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den == 0:
        return 1.0 if sst == 0 else float("nan")
    return float((msr - mse) / den)


def bland_altman_plot(a, b, title, unit, path):
    plt = get_plt()
    a, b = np.asarray(a, float), np.asarray(b, float)
    m, d = (a + b) / 2.0, a - b
    bias, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)
    ax.scatter(m, d, s=36, alpha=0.8, color="#0b6fb8", edgecolor="white", linewidth=0.6)
    ax.axhline(bias, color="#c0392b", lw=1.6, label="Sesgo %.2f %s" % (bias, unit))
    ax.axhline(bias + 1.96 * sd, color="#7f8c8d", ls="--", lw=1.2,
               label="+1.96 DE = %.2f" % (bias + 1.96 * sd))
    ax.axhline(bias - 1.96 * sd, color="#7f8c8d", ls="--", lw=1.2,
               label="-1.96 DE = %.2f" % (bias - 1.96 * sd))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Media (sistema, manual) [%s]" % unit)
    ax.set_ylabel("Sistema - manual [%s]" % unit)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return bias, sd


def verdict(key, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "PENDIENTE"
    _, op, tgt = TARGETS[key]
    ok = value <= tgt if op == "<=" else value >= tgt
    return "CUMPLE" if ok else "NO CUMPLE"


# --------------------------------------------------------------------------- #
# 1) CLASIFICADORES (OE4, A2.2, A2.3)
# --------------------------------------------------------------------------- #
def load_features(paths, trim_boundary=4):
    """Carga las ventanas etiquetadas de David, resta la linea base de pie plano (REST) de cada
    sesion (Etapa E2: referencia al pivote de apoyo) y descarta las primeras ventanas tras un
    cambio de tecla en vivo."""
    X, y, groups = [], [], []
    n_feat = len(FEAT_NAMES)
    for path in paths:
        with io.open(path, newline="", encoding="utf-8-sig") as f:
            r = csv.reader(f)
            header = next(r)
            fi = [i for i, h in enumerate(header) if re.search(r"_t\d\d$", h)]
            idx = dict((h, i) for i, h in enumerate(header))
            body = [row for row in r if len(row) >= len(header)]
            if not body:
                continue
            rest_mats = [
                np.array([float(row[i]) for i in fi], np.float32).reshape(WINDOW, n_feat)
                for row in body if row[idx["label"]] == "REST"
            ]
            if rest_mats:
                base = np.median(np.vstack(rest_mats), axis=0)
                if n_feat > 3:
                    base[3] = 0.0   # velocidad angular en reposo = 0
            else:
                base = np.zeros(n_feat, np.float32)

            runs = []
            for row in body:
                lbl = row[idx["label"]]
                if not runs or runs[-1][0] != lbl:
                    runs.append([lbl, []])
                runs[-1][1].append(row)
            for lbl, rs in runs:
                kept = rs[trim_boundary:-2] if (trim_boundary > 0 and len(rs) > trim_boundary + 5) else rs
                for row in kept:
                    subj = row[idx["subject"]].strip() if "subject" in idx else ""
                    sess = "%s:%s" % (os.path.basename(path), row[idx["session"]] if "session" in idx else "0")
                    mat = np.array([float(row[i]) for i in fi], np.float32).reshape(WINDOW, n_feat) - base
                    X.append(mat.reshape(-1))
                    y.append(row[idx["label"]])
                    groups.append(subj if subj else sess)
    if not X:
        sys.exit("No hay filas de caracteristicas en %s" % ", ".join(paths))
    return np.array(X, np.float32), np.array(y), np.array(groups)


def dump_size_kb(model, path=None):
    import joblib
    import tempfile
    tmp = path or tempfile.mktemp(suffix=".joblib")
    joblib.dump({"model": model, "window": WINDOW, "features": len(FEAT_NAMES)}, tmp,
                compress=3, protocol=2)
    kb = os.path.getsize(tmp) / 1024.0
    if path is None:
        os.remove(tmp)
    return kb


class XGBWrap(object):
    """XGBoost con etiquetas de texto (para predict/predict_proba/classes_)."""

    def __init__(self, seed=0):
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        self.seed = seed
        self.le = LabelEncoder()
        self.m = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.15,
                               subsample=0.9, colsample_bytree=0.8, n_jobs=1,
                               random_state=seed, verbosity=0)

    def fit(self, X, y):
        self.m.fit(X, self.le.fit_transform(y))
        self.classes_ = self.le.classes_
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.m.predict(X))

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def build_models(max_kb, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {}
    models["RandomForest"] = lambda: RandomForestClassifier(
        n_estimators=40, max_depth=10, min_samples_leaf=4, class_weight="balanced",
        n_jobs=1, random_state=seed)
    models["SVM-RBF"] = lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=5.0, gamma="scale", probability=True,
                              class_weight="balanced", random_state=seed))
    models["MLP"] = lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                                        early_stopping=True, random_state=seed))
    try:
        import xgboost  # noqa: F401
        models["XGBoost"] = lambda: XGBWrap(seed=seed)
    except Exception:
        print("[AVISO] xgboost no instalado: se omite (pip install xgboost).")
    return models


def fit_compact(name, factory, X, y, max_kb):
    """Ajusta y, solo para RandomForest, reduce el tamano hasta cumplir max_kb."""
    if name != "RandomForest":
        m = factory().fit(X, y)
        return m, dump_size_kb(m), {}
    from sklearn.ensemble import RandomForestClassifier
    for n, d, leaf in [(60, 12, 3), (40, 10, 4), (30, 9, 5), (25, 8, 6), (20, 8, 8),
                       (15, 7, 10), (12, 6, 12), (10, 6, 15)]:
        m = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_leaf=leaf,
                                   class_weight="balanced", n_jobs=1, random_state=0).fit(X, y)
        kb = dump_size_kb(m)
        if kb <= max_kb:
            return m, kb, {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    return m, kb, {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}


def make_splits(X, y, g):
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    ng = len(np.unique(g))
    if ng < 2:
        sys.exit("Se necesitan >= 2 sujetos/sesiones distintos para validacion por sujeto.")
    cv = LeaveOneGroupOut() if ng >= 3 else GroupKFold(n_splits=ng)
    return list(cv.split(X, y, g)), ng


def cmd_clasificadores(a):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 matthews_corrcoef)
    import joblib
    plt = get_plt()
    ensure_dir(a.out_dir)
    X, y, g = load_features(expand(a.csv))
    labels = [c for c in CLASS_ORDER if c in set(y)] + sorted(set(y) - set(CLASS_ORDER))
    splits, ng = make_splits(X, y, g)
    print("Ventanas: %d | sujetos/sesiones: %d | clases: %s"
          % (len(y), ng, dict(zip(*np.unique(y, return_counts=True)))))
    models = build_models(a.max_kb)
    results, preds = [], {}
    for name, factory in models.items():
        t0 = time.time()
        pred = np.empty(len(y), dtype=y.dtype)
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0]
                continue
            m = factory().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        preds[name] = pred
        final, kb, cfg = fit_compact(name, factory, X, y, a.max_kb)
        t1 = time.perf_counter()
        for _ in range(50):
            final.predict(X[:1])
        lat = (time.perf_counter() - t1) / 50.0 * 1000.0
        row = {
            "modelo": name,
            "f1_macro": float(f1_score(y, pred, average="macro", labels=labels)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "exactitud": float(accuracy_score(y, pred)),
            "peso_kb": round(kb, 1),
            "latencia_ms": round(lat, 2),
            "cumple_300kb": bool(kb <= a.max_kb),
            "config": cfg,
            "_model": final,
        }
        results.append(row)
        print("  %-13s F1=%.3f MCC=%.3f acc=%.3f  %.0f KB  %.2f ms  (%.0f s)%s"
              % (name, row["f1_macro"], row["mcc"], row["exactitud"], kb, lat,
                 time.time() - t0, "" if row["cumple_300kb"] else "  [>%d KB]" % a.max_kb))

    # Seleccion: mejor F1 entre los que caben en max_kb (desempate: menor latencia)
    ok = [r for r in results if r["cumple_300kb"]] or results
    best = sorted(ok, key=lambda r: (-round(r["f1_macro"], 2), r["latencia_ms"]))[0]
    print("\nModelo seleccionado: %s (F1 macro %.3f, %.1f KB)"
          % (best["modelo"], best["f1_macro"], best["peso_kb"]))

    # Tabla comparativa legible en Excel (separada por punto y coma y con encabezados claros)
    nombres_legibles = {
        "RandomForest": "Random Forest (Bosque Aleatorio)",
        "SVM-RBF": "SVM con Kernel RBF",
        "MLP": "Red Neuronal Multicapa (MLP 64x32)",
        "XGBoost": "XGBoost (Arboles Impulsados)",
    }
    tab = os.path.join(a.out_dir, "tabla_clasificadores.csv")
    with io.open(tab, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "Modelo Clasificador",
            "F1 Macro (Validacion por Sujeto)",
            "Coeficiente MCC",
            "Exactitud (%)",
            "Peso del Archivo (KB)",
            "Tiempo de Inferencia (ms)",
            "Cumple Limite de 300 KB",
            "Estado de Seleccion",
        ])
        for r in results:
            es_ganador = "SELECCIONADO (Desplegado)" if r is best else (
                "Descartado (> 300 KB)" if not r["cumple_300kb"] else "Candidato valido"
            )
            w.writerow([
                nombres_legibles.get(r["modelo"], r["modelo"]),
                "%.3f" % r["f1_macro"],
                "%.3f" % r["mcc"],
                "%.1f %%" % (r["exactitud"] * 100.0),
                "%.1f KB" % r["peso_kb"],
                "%.2f ms" % r["latencia_ms"],
                "Si" if r["cumple_300kb"] else "No",
                es_ganador,
            ])
    out_model = os.path.join(a.out_dir, a.model_name)
    joblib.dump({"model": best["_model"], "window": WINDOW, "features": len(FEAT_NAMES),
                 "classes": list(best["_model"].classes_), "name": best["modelo"]},
                out_model, compress=3, protocol=2)
    print("Guardado: %s (%.1f KB)" % (out_model, os.path.getsize(out_model) / 1024.0))

    # Matriz de confusion (LOSO) del mejor
    cm = confusion_matrix(y, preds[best["modelo"]], labels=labels)
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, "%d" % cm[i, j], ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title("%s - Matriz de confusion (LOSO, F1 macro %.3f)" % (best["modelo"], best["f1_macro"]))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "matriz_confusion_%s.png" % best["modelo"]))
    plt.close(fig)

    # Grafica comparativa F1 / MCC
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    xs = np.arange(len(results))
    ax.bar(xs - 0.2, [r["f1_macro"] for r in results], 0.4, label="F1 macro", color="#0b6fb8")
    ax.bar(xs + 0.2, [r["mcc"] for r in results], 0.4, label="MCC", color="#27ae60")
    ax.axhline(0.90, color="#c0392b", ls="--", lw=1.2, label="Meta F1 = 0.90")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["modelo"] for r in results])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Comparacion de clasificadores (LOSO, %d sujetos)" % ng)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "comparacion_clasificadores.png"))
    plt.close(fig)

    # Importancia por permutacion (A2.2): se permuta cada caracteristica en TODA la ventana
    base_name = best["modelo"]
    rng = np.random.RandomState(0)
    drops = np.zeros((len(FEAT_NAMES), len(splits)))
    for k, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            continue
        m = models[base_name]().fit(X[tr], y[tr])
        b = f1_score(y[te], m.predict(X[te]), average="macro")
        for fi in range(len(FEAT_NAMES)):
            cols = [c for c in range(X.shape[1]) if c % len(FEAT_NAMES) == fi]
            vals = []
            for _ in range(3):
                Xp = X[te].copy()
                Xp[:, cols] = Xp[rng.permutation(len(te))][:, cols]
                vals.append(b - f1_score(y[te], m.predict(Xp), average="macro"))
            drops[fi, k] = np.mean(vals)
    imp = drops.mean(axis=1)
    err = drops.std(axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
    order = np.argsort(imp)
    ax.barh([FEAT_NAMES[i] for i in order], imp[order], xerr=err[order], color="#8e44ad")
    ax.set_xlabel("Caida del F1 macro al permutar la caracteristica")
    ax.set_title("Importancia por permutacion (%s)" % base_name)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "importancia_permutacion.png"))
    plt.close(fig)

    save_metrics(a.out_dir, "clasificadores", {
        "ventanas": int(len(y)), "sujetos": int(ng), "modelo_seleccionado": best["modelo"],
        "f1_macro": best["f1_macro"], "mcc": best["mcc"], "peso_kb": best["peso_kb"],
        "latencia_ms": best["latencia_ms"],
        "importancia_permutacion": dict((FEAT_NAMES[i], float(imp[i])) for i in range(len(imp))),
        "tabla": [dict((k, v) for k, v in r.items() if k != "_model") for r in results],
    })


# --------------------------------------------------------------------------- #
# 2) CAMARA (OE1 / A1.3)
# --------------------------------------------------------------------------- #
def cmd_camara(a):
    plt = get_plt()
    ensure_dir(a.out_dir)
    byv = {}
    for p in expand(a.perf):
        for r in read_csv_dicts(p):
            v = (r.get("view") or "").strip() or os.path.splitext(os.path.basename(p))[0]
            n = fnum(r.get("n_inf"), 0)
            if n <= 0:
                continue
            byv.setdefault(v, []).append(r)
    if not byv:
        sys.exit("Los CSV de rendimiento no tienen filas con inferencias (n_inf > 0).")
    rows = []
    for v, rs in byv.items():
        w = np.array([fnum(r["n_inf"], 0) for r in rs])
        vis = np.array([fnum(r["vis_frac"]) for r in rs])
        knee = np.array([fnum(r["knee_flex_mean"]) for r in rs])
        hip = np.array([fnum(r["hip_abd_mean"]) for r in rs])
        trunk = np.array([fnum(r["trunk_mean"]) for r in rs])
        ks = np.array([fnum(r["knee_flex_std"]) for r in rs])
        ok = ~np.isnan(knee) & ~np.isnan(hip)

        def wavg(x, m):
            return float(np.average(x[m], weights=w[m])) if m.any() else float("nan")

        e_sag = wavg(np.abs(knee), ok)          # sagital: flexion de rodilla (real ~0 de pie)
        e_fro = wavg(np.abs(hip), ok)           # frontal: abduccion (real ~0)
        rows.append({
            "vista": v, "vis_pct": 100.0 * wavg(vis, ~np.isnan(vis)),
            "err_sagital_deg": e_sag, "err_frontal_deg": e_fro,
            "err_conjunto_deg": float(math.sqrt((e_sag ** 2 + e_fro ** 2) / 2.0)),
            "inclinacion_tronco_deg": wavg(np.abs(trunk), ~np.isnan(trunk)),
            "jitter_rodilla_deg": wavg(ks, ~np.isnan(ks)),
            "muestras": int(w.sum()),
        })
    order = ["frontal", "obl30", "obl45", "lateral"]
    rows.sort(key=lambda r: order.index(r["vista"]) if r["vista"] in order else 99)
    cand = [r for r in rows if r["vis_pct"] >= a.min_vis] or rows
    best = min(cand, key=lambda r: r["err_conjunto_deg"])
    print("\n%-9s %8s %10s %10s %10s %8s" % ("vista", "vis %", "err sag", "err front", "conjunto", "jitter"))
    for r in rows:
        print("%-9s %8.1f %10.2f %10.2f %10.2f %8.2f%s" % (
            r["vista"], r["vis_pct"], r["err_sagital_deg"], r["err_frontal_deg"],
            r["err_conjunto_deg"], r["jitter_rodilla_deg"], "   <== MEJOR" if r is best else ""))
    vista_nombres = {
        "frontal": ("Vista Frontal", "0 grados"),
        "obl30": ("Vista Oblicua 30", "30 grados"),
        "obl45": ("Vista Oblicua 45", "45 grados"),
        "lateral": ("Vista Lateral (Perfil)", "90 grados"),
    }
    with io.open(os.path.join(a.out_dir, "tabla_camara.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.writer(f, delimiter=";")
        w_.writerow([
            "Posicion de Camara",
            "Angulo de Camara",
            "Visibilidad Articular (%)",
            "Error Sagital Rodilla (grados)",
            "Error Frontal Cadera (grados)",
            "Error Angular Conjunto (grados)",
            "Inclinacion Tronco (grados)",
            "Variacion / Jitter (grados)",
            "Cuadros Evaluados",
            "Resultado",
        ])
        for r in rows:
            v_lbl, v_ang = vista_nombres.get(r["vista"], (r["vista"], "-"))
            w_.writerow([
                v_lbl,
                v_ang,
                "%.1f %%" % r["vis_pct"],
                "%.2f grados" % r["err_sagital_deg"],
                "%.2f grados" % r["err_frontal_deg"],
                "%.2f grados" % r["err_conjunto_deg"],
                "%.2f grados" % r["inclinacion_tronco_deg"],
                "%.2f grados" % r["jitter_rodilla_deg"],
                "%d cuadros" % r["muestras"],
                "Menor error conjunto" if r is best else "Evaluada",
            ])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160)
    vs = [r["vista"] for r in rows]
    axs[0].bar(vs, [r["vis_pct"] for r in rows], color="#0b6fb8")
    axs[0].axhline(a.min_vis, color="#c0392b", ls="--", lw=1)
    axs[0].set_title("Cuadros con cadera-rodilla-tobillo visibles (%)")
    axs[0].set_ylim(0, 105)
    xs = np.arange(len(rows))
    axs[1].bar(xs - 0.27, [r["err_sagital_deg"] for r in rows], 0.27, label="Sagital", color="#27ae60")
    axs[1].bar(xs, [r["err_frontal_deg"] for r in rows], 0.27, label="Frontal", color="#e67e22")
    axs[1].bar(xs + 0.27, [r["err_conjunto_deg"] for r in rows], 0.27, label="Conjunto", color="#8e44ad")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(vs)
    axs[1].set_title("Error angular en postura neutra (grados)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "geometria_camara.png"))
    plt.close(fig)
    save_metrics(a.out_dir, "camara", {"mejor_vista": best["vista"], "tabla": rows})


# --------------------------------------------------------------------------- #
# 3) HARDWARE (OE1 / A1.1 / A3.3)
# --------------------------------------------------------------------------- #
def cmd_hardware(a):
    plt = get_plt()
    ensure_dir(a.out_dir)
    rs = []
    for p in expand(a.perf):
        rs.extend(read_csv_dicts(p))
    rs = [r for r in rs if fnum(r.get("loop_fps"), 0) > 0 and fnum(r.get("t_s"), 0) >= a.warmup
          and fnum(r.get("n_inf"), 0) > 0]   # solo segundos con inferencia real (excluye menu)
    if not rs:
        sys.exit("Sin filas validas (revisa --warmup o que se grabe con --perf-log).")
    col = lambda k: np.array([fnum(r.get(k)) for r in rs], float)  # noqa: E731
    fps, inf = col("loop_fps"), col("infer_ms")
    cam = col("cam_fps")
    cpu, ram, temp, gpu = col("cpu_pct"), col("ram_mb"), col("temp_c"), col("gpu_pct")
    nn = lambda x: x[~np.isnan(x)]  # noqa: E731
    t_all = col("t_s")
    dur = float(t_all.max() - t_all.min()) if len(t_all) > 1 else 1.0
    res = {
        "duracion_s": float(col("t_s").max()),
        "fps_media": float(np.mean(fps)), "fps_mediana": float(np.median(fps)),
        "fps_p5": float(np.percentile(fps, 5)), "fps_min": float(fps.min()),
        "inferencias_pose_por_s": float(col("n_inf").sum() / max(dur, 1e-6)),
        "camara_fps_media": float(np.nanmean(cam)),
        "infer_ms_media": float(np.nanmean(inf)), "infer_ms_p95": float(np.nanpercentile(inf, 95)),
        "cpu_pct_media": float(np.mean(nn(cpu))) if len(nn(cpu)) else None,
        "ram_mb_max": float(np.max(nn(ram))) if len(nn(ram)) else None,
        "temp_c_max": float(np.max(nn(temp))) if len(nn(temp)) else None,
        "gpu_pct_media": float(np.mean(nn(gpu))) if len(nn(gpu)) else None,
    }
    res["fps_sostenida"] = res["fps_mediana"]
    print(json.dumps(res, indent=2))
    print("FPS sostenida (mediana) = %.2f -> %s | piso p5 = %.2f -> %s" % (
        res["fps_sostenida"], verdict("fps", res["fps_sostenida"]),
        res["fps_p5"], verdict("fps_floor", res["fps_p5"])))
    t = col("t_s")
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), dpi=150, sharex=True)
    for ax, (y_, ttl, c) in zip(axs.ravel(), [(fps, "FPS del bucle", "#0b6fb8"),
                                              (inf, "Latencia de inferencia (ms)", "#c0392b"),
                                              (temp, "Temperatura (C)", "#e67e22"),
                                              (cpu, "CPU (%)", "#27ae60")]):
        ax.plot(t, y_, color=c, lw=1.2)
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
    axs[0, 0].axhline(8, color="black", ls="--", lw=0.8)
    axs[0, 0].axhline(5, color="gray", ls=":", lw=0.8)
    axs[1, 0].set_xlabel("t (s)")
    axs[1, 1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "rendimiento_jetson.png"))
    plt.close(fig)
    save_metrics(a.out_dir, "hardware", res)


# --------------------------------------------------------------------------- #
# 4) CONTEO (OE5 / A3.4)
# --------------------------------------------------------------------------- #
EX_MAP = {
    "sit-to-stand": "STS", "sts": "STS",
    "abduccion_cadera": "ABD", "abd": "ABD",
    "elevacion_talones": "TALONES", "talones": "TALONES",
    "elevacion_puntas": "PUNTAS", "puntas": "PUNTAS",
}


def norm_ex(s):
    return EX_MAP.get((s or "").strip().lower(), (s or "").strip().upper())


def load_auto(paths):
    """Cuenta repeticiones validas (OK) y totales por (sujeto, ejercicio, serie)."""
    out = {}
    for p in paths:
        for r in read_csv_dicts(p):
            k = ((r.get("Sujeto") or "").strip(), norm_ex(r.get("Ejercicio")),
                 str(r.get("Serie_Val") or "1").strip())
            d = out.setdefault(k, {"ok": 0, "total": 0, "ts": [], "peak": []})
            d["total"] += 1
            if (r.get("Resultado") or "").upper() == "OK":
                d["ok"] += 1
            try:
                import datetime as _dt
                ts_s = r["Timestamp"]
                fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in ts_s else "%Y-%m-%dT%H:%M:%S"
                d["ts"].append(_dt.datetime.strptime(ts_s, fmt).timestamp())
            except Exception:
                pass
            try:
                m = json.loads(r.get("Metricas_Maximas") or "{}")
                for key in ("angulo_max", "pico_rodilla", "peak"):
                    if key in m:
                        d["peak"].append(float(m[key]))
                        break
            except Exception:
                pass
    return out


def cmd_conteo(a):
    ensure_dir(a.out_dir)
    auto = load_auto(expand(a.auto))
    man = read_csv_dicts(a.manual)
    use = "ok" if a.count == "validas" else "total"
    A, M, O1, O2, lab, ex = [], [], [], [], [], []
    extra = {"angulo": ([], []), "ciclo": ([], [])}
    for r in man:
        k = ((r.get("sujeto") or "").strip(), norm_ex(r.get("ejercicio")),
             str(r.get("serie_val") or "1").strip())
        o1, o2 = fnum(r.get("obs1")), fnum(r.get("obs2"))
        if math.isnan(o1):
            continue
        ref = np.nanmean([o1, o2]) if not math.isnan(o2) else o1
        d = auto.get(k, {"ok": 0, "total": 0, "ts": [], "peak": []})
        A.append(d[use])
        M.append(ref)
        O1.append(o1)
        O2.append(o2)
        lab.append("%s/%s/%s" % k)
        ex.append(k[1])
        ang_m = fnum(r.get("angulo_max_deg"))
        if not math.isnan(ang_m) and d["peak"]:
            extra["angulo"][0].append(float(np.mean(d["peak"])))
            extra["angulo"][1].append(ang_m)
        cyc_m = fnum(r.get("tiempo_ciclo_s"))
        dur_s = fnum(r.get("duracion_s"), 30.0)
        if math.isnan(cyc_m) and ref > 0 and not math.isnan(dur_s) and dur_s > 0:
            span_s = (max(d["ts"]) - min(d["ts"])) if len(d["ts"]) >= 2 else dur_s
            n_int = max(1.0, ref - 1.0) if len(d["ts"]) >= 2 else ref
            cyc_m = span_s / float(n_int)
        if not math.isnan(cyc_m) and len(d["ts"]) >= 2:
            extra["ciclo"][0].append(float(np.mean(np.diff(sorted(d["ts"])))))
            extra["ciclo"][1].append(cyc_m)
        if k not in auto:
            print("[AVISO] sin datos automaticos para %s (conteo 0)" % "/".join(k))
    if not A:
        sys.exit("No hay series en el archivo manual (columnas: sujeto, ejercicio, serie_val, obs1, obs2).")
    A, M = np.array(A, float), np.array(M, float)
    err = A - M
    res = {
        "series": int(len(A)),
        "mae_conteo": float(np.mean(np.abs(err))),
        "pct_err_le1": float(100.0 * np.mean(np.abs(err) <= 1.0)),
        "sesgo": float(err.mean()),
        "icc_conteo": icc_2_1(np.c_[A, M]),
        "tipo_conteo": use,
    }
    ok2 = [(x, y) for x, y in zip(O1, O2) if not math.isnan(y)]
    if len(ok2) >= 2:
        res["icc_interobservador"] = icc_2_1(np.array(ok2))
        res["mae_interobservador"] = float(np.mean([abs(x - y) for x, y in ok2]))
    per = {}
    for e in sorted(set(ex)):
        m = np.array([x == e for x in ex])
        per[e] = {"series": int(m.sum()), "mae": float(np.mean(np.abs(err[m]))),
                  "pct_err_le1": float(100.0 * np.mean(np.abs(err[m]) <= 1))}
    res["por_ejercicio"] = per
    res["bland_altman_conteo"] = dict(zip(("sesgo", "de"), bland_altman_plot(
        A, M, "Bland-Altman: conteo de repeticiones", "rep",
        os.path.join(a.out_dir, "bland_altman_conteo.png"))))
    if len(extra["angulo"][0]) >= 3:
        x, y = np.array(extra["angulo"][0]), np.array(extra["angulo"][1])
        res["icc_angulo_max"] = icc_2_1(np.c_[x, y])
        bland_altman_plot(x, y, "Bland-Altman: angulo maximo", "deg",
                          os.path.join(a.out_dir, "bland_altman_angulo.png"))
    if len(extra["ciclo"][0]) >= 3:
        x, y = np.array(extra["ciclo"][0]), np.array(extra["ciclo"][1])
        res["icc_tiempo_ciclo"] = icc_2_1(np.c_[x, y])
        bland_altman_plot(x, y, "Bland-Altman: tiempo de ciclo", "s",
                          os.path.join(a.out_dir, "bland_altman_ciclo.png"))
    ej_legible = {
        "STS": "Sentarse y Levantarse (STS)",
        "ABD": "Abduccion de Cadera",
        "TALONES": "Elevacion de Talones",
        "PUNTAS": "Elevacion de Puntas",
    }
    with io.open(os.path.join(a.out_dir, "tabla_conteo_series.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "Participante",
            "Ejercicio",
            "Serie Evaluada",
            "Conteo del Sistema (rep)",
            "Conteo Observador 1 (rep)",
            "Conteo Observador 2 (rep)",
            "Promedio Observadores (rep)",
            "Diferencia Sistema vs Manual (rep)",
            "Cumple Error <= 1 rep",
        ])
        for i in range(len(A)):
            partes = lab[i].split("/")
            suj_i = partes[0] if len(partes) > 0 else lab[i]
            ej_i = ej_legible.get(partes[1], partes[1]) if len(partes) > 1 else "STS"
            ser_i = ("Serie %s" % partes[2]) if len(partes) > 2 else "Serie 1"
            w.writerow([
                suj_i,
                ej_i,
                ser_i,
                "%d" % int(round(A[i])),
                "%d" % int(round(O1[i])),
                "%d" % int(round(O2[i])) if not math.isnan(O2[i]) else "-",
                "%.1f" % M[i],
                "%+.1f" % err[i],
                "Si (Cumple)" if abs(err[i]) <= 1.0 else "No",
            ])
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("MAE=%.2f (%s) | err<=1: %.0f%% (%s) | ICC=%.2f (%s)" % (
        res["mae_conteo"], verdict("mae_conteo", res["mae_conteo"]), res["pct_err_le1"],
        verdict("pct_err_le1", res["pct_err_le1"]), res["icc_conteo"],
        verdict("icc", res["icc_conteo"])))
    save_metrics(a.out_dir, "conteo", res)


# --------------------------------------------------------------------------- #
# 5) FASES (OE4 / A2.3)
# --------------------------------------------------------------------------- #
def read_json_docs(path):
    txt = io.open(path, encoding="utf-8").read()
    dec, i, docs = json.JSONDecoder(), 0, []
    while i < len(txt):
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            break
        try:
            d, j = dec.raw_decode(txt, i)
        except ValueError:
            break
        docs.append(d)
        i = j
    return docs


def match_events(sys_t, man_t, tol):
    sys_t, man_t = sorted(sys_t), sorted(man_t)
    used, tp = set(), 0
    for m in man_t:
        best, bd = None, tol + 1e-9
        for i, s in enumerate(sys_t):
            if i in used:
                continue
            d = abs(s - m)
            if d <= bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(sys_t) - tp, len(man_t) - tp
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def _angle_3pt(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], float)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-4 or nc < 1e-4:
        return None
    cos_v = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_v)))


def _ref_transitions_from_landmarks(docs_k):
    """Deriva los instantes de cambio de fase directamente de la geometria 2D de landmarks."""
    ts_out = []
    prev_st = None
    for d in sorted(docs_k, key=lambda x: x.get("timestamp_ms", 0)):
        t_s = d.get("timestamp_ms", 0) / 1000.0
        cin = d.get("cinematica") or {}
        lms = dict((p["punto"], (p["x"], p["y"], p.get("conf", 1.0)))
                   for p in (d.get("landmarks_2d") or []) if "punto" in p)
        ang = None
        for h, k, a in (("cadera_i", "rodilla_i", "tobillo_i"),
                        ("cadera_d", "rodilla_d", "tobillo_d"),
                        ("cadera_izq", "rodilla_izq", "tobillo_izq"),
                        ("cadera_der", "rodilla_der", "tobillo_der")):
            if h in lms and k in lms and a in lms:
                v = _angle_3pt(lms[h], lms[k], lms[a])
                if v is not None:
                    ang = v
                    break
        vel = cin.get("velocidad_angular_rodilla", 0.0)
        st = None
        if ang is not None and (d.get("ejercicio") or "").upper() == "STS":
            if ang >= 162.0:
                st = "STANDING"
            elif ang <= 98.0:
                st = "SIT"
            elif vel >= 10.0:
                st = "RISING"
            elif vel <= -10.0:
                st = "LOWERING"
        if st is None:
            st = d.get("estado")
        if st and st != prev_st:
            ts_out.append(t_s)
            prev_st = st
    return ts_out


def cmd_fases(a):
    ensure_dir(a.out_dir)
    docs = [d for d in read_json_docs(a.db) if d.get("estado") and d.get("ejercicio")]
    if a.session:
        docs = [d for d in docs if a.session in (d.get("sesion") or "")]
    ev, docs_by_k = {}, {}
    prev = {}
    for d in sorted(docs, key=lambda d: d.get("timestamp_ms", 0)):
        k = (d.get("sesion"), d.get("ejercicio"))
        docs_by_k.setdefault(k, []).append(d)
        if prev.get(k) != d["estado"]:
            ev.setdefault(k, []).append(d["timestamp_ms"] / 1000.0)
            prev[k] = d["estado"]
    man = read_csv_dicts(a.manual) if (a.manual and os.path.isfile(a.manual)) else []
    # Ignorar la fila de ejemplo por defecto si el usuario no edito manual_fases.csv
    man = [r for r in man if (r.get("fase") or "") != "inicio_levantarse"]
    tot = [0, 0, 0]
    res_k = {}
    for k, sys_ts in ev.items():
        mt = [fnum(r.get("t_s")) for r in man
              if norm_ex(r.get("ejercicio")) == norm_ex(k[1]) and
              (not r.get("sesion") or r.get("sesion") in (k[0] or ""))]
        mt = [x for x in mt if not math.isnan(x)]
        if mt:
            off = a.offset if a.offset is not None else (min(sys_ts) - min(mt))
            mt_al = [x + off for x in mt]
        else:
            # Si no hay CSV manual de fases, contrasta contra la reconstruccion geometrica de landmarks_2d
            mt_al = _ref_transitions_from_landmarks(docs_by_k.get(k, []))
        if not mt_al:
            continue
        tp, fp, fn, p, r, f1 = match_events(sys_ts, mt_al, a.tol)
        res_k["%s/%s" % k] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
        tot[0] += tp
        tot[1] += fp
        tot[2] += fn
    tp, fp, fn = tot
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    res = {"tolerancia_s": a.tol, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
           "f1_fases": f1, "detalle": res_k}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("F1 fases (+-%.2f s) = %.3f -> %s" % (a.tol, f1, verdict("f1_fases", f1)))
    save_metrics(a.out_dir, "fases", res)


# --------------------------------------------------------------------------- #
# 6) FP16 (A3.1)
# --------------------------------------------------------------------------- #
def cmd_fp16(a):
    import joblib
    from sklearn.metrics import f1_score
    obj = joblib.load(a.model)
    model = obj["model"] if isinstance(obj, dict) else obj
    out = {}
    for tag, paths in (("fp32", a.fp32), ("fp16", a.fp16)):
        X, y, _ = load_features(expand(paths))
        out[tag] = float(f1_score(y, model.predict(X), average="macro"))
        print("F1 macro %s = %.4f (%d ventanas)" % (tag.upper(), out[tag], len(y)))
    d = abs(out["fp32"] - out["fp16"]) * 100.0
    res = {"f1_fp32": out["fp32"], "f1_fp16": out["fp16"], "delta_f1_fp16": d}
    print("Degradacion = %.2f puntos de F1 -> %s" % (d, verdict("delta_f1_fp16", d)))
    save_metrics(a.out_dir, "fp16", res)


# --------------------------------------------------------------------------- #
# 7) TABLA 2 (cierre)
# --------------------------------------------------------------------------- #
def cmd_tabla2(a):
    ensure_dir(a.out_dir)
    clf, hw = load_metrics(a.out_dir, "clasificadores"), load_metrics(a.out_dir, "hardware")
    cnt, fas = load_metrics(a.out_dir, "conteo"), load_metrics(a.out_dir, "fases")
    f16 = load_metrics(a.out_dir, "fp16")
    icc_vals = []
    if cnt:
        for k in ("icc_conteo", "icc_angulo_max", "icc_tiempo_ciclo"):
            if cnt.get(k) is not None and not math.isnan(cnt[k]) and cnt[k] > 0.0:
                icc_vals.append(cnt[k])
    vals = {
        "mae_conteo": cnt["mae_conteo"] if cnt else None,
        "pct_err_le1": cnt["pct_err_le1"] if cnt else None,
        "f1_macro": clf["f1_macro"] if clf else None,
        "f1_fases": fas["f1_fases"] if fas else None,
        "icc": float(np.mean(icc_vals)) if icc_vals else None,
        "fps": hw["fps_sostenida"] if hw else None,
        "fps_floor": hw["fps_p5"] if hw else None,
        "peso_kb": clf["peso_kb"] if clf else None,
        "delta_f1_fp16": f16["delta_f1_fp16"] if f16 else None,
    }
    unidades = {
        "mae_conteo": ("<= 0.50 repeticiones", "%.2f repeticiones"),
        "pct_err_le1": (">= 90.0 %", "%.1f %%"),
        "f1_macro": (">= 0.900", "%.3f"),
        "f1_fases": (">= 0.850", "%.3f"),
        "icc": (">= 0.750", "%.3f"),
        "fps": (">= 8.00 cuadros/s", "%.2f cuadros/s"),
        "fps_floor": (">= 5.00 cuadros/s", "%.2f cuadros/s"),
        "peso_kb": ("<= 300.0 KB", "%.1f KB"),
        "delta_f1_fp16": ("<= 2.00 puntos de F1", "%.2f puntos de F1"),
    }
    rows = []
    for k, (name, op, tgt) in TARGETS.items():
        v = vals[k]
        meta_txt, fmt_v = unidades.get(k, ("%s %g" % (op, tgt), "%.3f"))
        rows.append([name, meta_txt, "-" if v is None else (fmt_v % v), verdict(k, v)])
    with io.open(os.path.join(a.out_dir, "tabla2_indicadores.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Indicador Cuantitativo del Plan", "Meta Establecida", "Resultado Obtenido", "Cumplimiento"])
        w.writerows(rows)
    md = ["| Indicador | Meta | Valor obtenido | Estado |", "|---|---|---|---|"]
    md += ["| %s |" % " | ".join(r) for r in rows]
    with io.open(os.path.join(a.out_dir, "tabla2_indicadores.md"), "w", encoding="utf-8") as f:
        f.write("# Tabla 2. Indicadores cuantitativos de exito\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    pend = [r[0] for r in rows if r[3] == "PENDIENTE"]
    if pend:
        print("\nPendientes: " + "; ".join(pend))


# --------------------------------------------------------------------------- #
# 8) PLANTILLAS Y CAPTURA MANUAL INTERACTIVA
# --------------------------------------------------------------------------- #
def cmd_plantillas(a):
    d = ensure_dir(a.out_dir)
    p1 = os.path.join(d, "manual_conteo.csv")
    # UTF-8 con BOM y punto y coma (;) para que Excel en espanol abra cada columna en su propia celda
    with io.open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sujeto", "ejercicio", "serie_val", "obs1", "obs2", "duracion_s", "notas"])
        for sid in ("D13", "D14", "D15", "D16", "D17"):
            w.writerow([sid, "TALONES", 1, "", "", 30, "Elevacion de Talones (Serie 1)"])
            w.writerow([sid, "PUNTAS",  2, "", "", 30, "Elevacion de Puntas (Serie 2)"])
    print("Plantilla lista para Excel (solo llena columnas obs1 y obs2): %s" % p1)


def cmd_capturar_manual(a):
    """Pregunta en la terminal cuantas repeticiones contaron el Observador 1 y el Observador 2
    para cada serie encontrada en logs/reps_*.csv (o D13..D17) y guarda manual_conteo.csv."""
    d = ensure_dir(a.out_dir)
    p1 = os.path.join(d, "manual_conteo.csv")
    series_list = []
    if a.auto:
        auto = load_auto(expand(a.auto))
        series_list = sorted(auto.keys())
    if not series_list:
        series_list = []
        for sid in ("D13", "D14", "D15", "D16", "D17"):
            series_list.append((sid, "TALONES", "1"))
            series_list.append((sid, "PUNTAS", "2"))

    print("\n=== CAPTURA DE CONTEO MANUAL (2 OBSERVADORES) ===")
    print("Escribe solo el numero de repeticiones que conto cada observador (Enter para saltar serie).\n")
    rows_out = []
    for (suj, ej, ser) in series_list:
        raw1 = input("  %s | %s | Serie %s -> Repeticiones Observador 1: " % (suj, ej, ser)).strip()
        if not raw1:
            continue
        raw2 = input("  %s | %s | Serie %s -> Repeticiones Observador 2 [%s]: " % (suj, ej, ser, raw1)).strip()
        if not raw2:
            raw2 = raw1
        dur = input("  %s | %s | Serie %s -> Duracion en segundos [30]: " % (suj, ej, ser)).strip() or "30"
        rows_out.append([suj, ej, ser, raw1, raw2, dur, "Prueba 30s"])

    with io.open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sujeto", "ejercicio", "serie_val", "obs1", "obs2", "duracion_s", "notas"])
        w.writerows(rows_out)
    print("\n[OK] Guardado en %s (%d series registradas)." % (p1, len(rows_out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="resultados")
    sp = ap.add_subparsers(dest="cmd")
    sp.required = True

    s = sp.add_parser("clasificadores")
    s.add_argument("csv", nargs="+", help="features_windows*.csv generados con --collect")
    s.add_argument("--max-kb", type=float, default=300.0)
    s.add_argument("--model-name", default="exercise_clf.joblib")
    s.set_defaults(fn=cmd_clasificadores)

    s = sp.add_parser("camara")
    s.add_argument("perf", nargs="+", help="logs/perf_<vista>.csv (con --view y --perf-log)")
    s.add_argument("--min-vis", type=float, default=80.0, help="visibilidad minima (%%) para elegir vista")
    s.set_defaults(fn=cmd_camara)

    s = sp.add_parser("hardware")
    s.add_argument("perf", nargs="+")
    s.add_argument("--warmup", type=float, default=5.0, help="segundos iniciales a ignorar")
    s.set_defaults(fn=cmd_hardware)

    s = sp.add_parser("conteo")
    s.add_argument("--auto", nargs="+", required=True, help="logs/reps_*.csv")
    s.add_argument("--manual", required=True, help="manual_conteo.csv")
    s.add_argument("--count", choices=["validas", "todas"], default="validas",
                   help="contar solo repeticiones OK (por defecto) o todas las detectadas")
    s.set_defaults(fn=cmd_conteo)

    s = sp.add_parser("fases")
    s.add_argument("--db", default="bd_pacientes.json1")
    s.add_argument("--manual", default="", help="Opcional: manual_fases.csv (si se omite, usa landmarks_2d)")
    s.add_argument("--session", default="")
    s.add_argument("--tol", type=float, default=0.2)
    s.add_argument("--offset", type=float, default=None,
                   help="desfase (s) sistema - manual; por defecto se alinea con el primer evento")
    s.set_defaults(fn=cmd_fases)

    s = sp.add_parser("fp16")
    s.add_argument("--model", default="resultados/exercise_clf.joblib")
    s.add_argument("--fp32", nargs="+", required=True)
    s.add_argument("--fp16", nargs="+", required=True)
    s.set_defaults(fn=cmd_fp16)

    s = sp.add_parser("tabla2")
    s.set_defaults(fn=cmd_tabla2)

    s = sp.add_parser("plantillas")
    s.set_defaults(fn=cmd_plantillas)

    s = sp.add_parser("capturar-manual",
                      help="Modo interactivo en terminal para anotar el conteo de los 2 observadores sin abrir Excel")
    s.add_argument("--auto", nargs="*", default=[], help="Opcional: logs/reps_S*.csv para detectar las series")
    s.set_defaults(fn=cmd_capturar_manual)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
