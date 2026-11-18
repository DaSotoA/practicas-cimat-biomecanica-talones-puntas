#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena el clasificador temporal de ejercicio (STS / ABD_I / ABD_D / REST / TRANSITION)
que usa hrnet_jetson_pipeline_STSV2.py.

PASO 1 - RECOLECTAR (en la Jetson, con el pipeline en modo recoleccion)
    Sin modelo el pipeline entra solo en recoleccion; tambien se fuerza con --collect.
    Cada ventana de 24 muestras (~3 s) se guarda como una fila de 96 valores.

    python3 hrnet_jetson_pipeline_STSV2.py --engine hrnet_wb_fp16.engine --collect \
            --collect-out dataset/sesion1.csv --collect-label REST

    Teclas durante la sesion (cambian la etiqueta de las filas que se graban):
        1 = STS   2 = ABD_I   3 = ABD_D   4 = REST   5 = TRANSITION   0 = pausa
    Receta: ~2-3 min por clase y por sesion. Para STS y ABD haz repeticiones normales,
    lentas, rapidas y "malas" (con trampa); para REST quedate quieto de pie/sentado;
    para TRANSITION camina, cambia de postura, sal y entra del encuadre.
    Varia camara (perfil / 45 grados / frente), ropa y personas. Una sesion = un archivo.

PASO 2 - ENTRENAR (en la Jetson o en la PC; ver nota de versiones)
    python3 train_exercise_classifier.py dataset/*.csv --out exercise_clf.joblib

PASO 3 - USAR
    python3 hrnet_jetson_pipeline_STSV2.py --engine hrnet_wb_fp16.engine --model exercise_clf.joblib

NOTA DE VERSIONES: el .joblib solo se carga con versiones compatibles de scikit-learn/numpy.
JetPack 4.6 / Python 3.6 admite como maximo scikit-learn 0.24.2. Lo mas seguro es entrenar
en la propia Jetson (el modelo es pequeno) o en la PC con `pip install scikit-learn==0.24.2`
en un entorno Python 3.6-3.8.
"""
import argparse
import csv
import glob
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

try:
    import joblib
except ImportError:                       # sklearn antiguo
    from sklearn.externals import joblib

WINDOW, N_FEAT = 24, 6


def load(paths):
    X, y, groups = [], [], []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r)                             # encabezado
                # Columnas de caracteristicas = las que terminan en _tNN (admite CSV con o sin
                # metadatos subject/view/light/band)
                fi = [i for i, h in enumerate(header) if h[-4:-2] == "_t" and h[-2:].isdigit()]
                si = header.index("subject") if "subject" in header else None
                for row in r:
                    X.append([float(row[i]) for i in fi])    # 96 valores
                    y.append(row[2])
                    subj = row[si].strip() if si is not None else ""
                    groups.append(subj if subj else path + ":" + row[1])   # sujeto o sesion
    if not X:
        sys.exit("No se encontraron filas. Revisa las rutas.")
    return np.array(X, np.float32), np.array(y), np.array(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="CSV(s) generados con --collect (admite comodines)")
    ap.add_argument("--out", default="exercise_clf.joblib")
    ap.add_argument("--trees", type=int, default=60, help="pocos arboles = inferencia rapida en la Jetson")
    ap.add_argument("--depth", type=int, default=12)
    a = ap.parse_args()

    X, y, g = load(a.csv)
    assert X.shape[1] == WINDOW * N_FEAT, "Se esperaban %d columnas de caracteristicas" % (WINDOW * N_FEAT)
    labels, counts = np.unique(y, return_counts=True)
    print("Filas: %d | clases: %s" % (len(y), dict(zip(labels, counts))))

    def new_model():
        # n_jobs=1: en la Jetson Nano evita el sobrecosto de hilos para predecir 1 muestra
        return RandomForestClassifier(n_estimators=a.trees, max_depth=a.depth, min_samples_leaf=2,
                                      class_weight="balanced", n_jobs=1, random_state=0)

    # Validacion por SESION (las ventanas solapadas de una misma sesion no deben separarse)
    n_groups = len(np.unique(g))
    if n_groups >= 2:
        k = min(5, n_groups)
        pred = np.empty_like(y)
        for tr, te in GroupKFold(n_splits=k).split(X, y, g):
            pred[te] = new_model().fit(X[tr], y[tr]).predict(X[te])
        print("\nValidacion cruzada por sesion (%d pliegues):" % k)
        print(classification_report(y, pred, digits=3))
        print(confusion_matrix(y, pred, labels=labels))
    else:
        print("\n[AVISO] Solo 1 sesion: sin validacion honesta. Graba varias sesiones/personas.")

    model = new_model().fit(X, y)
    joblib.dump({"model": model, "window": WINDOW, "features": N_FEAT}, a.out, protocol=2)
    print("\nModelo guardado en %s (clases: %s)" % (a.out, list(model.classes_)))


if __name__ == "__main__":
    main()
