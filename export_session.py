#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta o gestiona las sesiones de TODOS los pacientes guardadas en Redis Streams
(o convierte un archivo con --in-file) hacia un UNICO archivo unificado NoSQL
(por defecto bd_pacientes.json1).

Uso en la Jetson:
  # Listar todos los pacientes y sesiones en Redis:
  python3 export_session.py --list

  # Exportar TODOS los pacientes en un solo archivo unificado:
  python3 export_session.py --out bd_pacientes.json1

  # Exportar solo un paciente especifico:
  python3 export_session.py --patient Paciente_02 --out paciente_02.json1

  # Borrar una sesion especifica o vaciar todo Redis:
  python3 export_session.py --delete 20261028_122747
  python3 export_session.py --clear-all
"""
import argparse
import csv
import json

try:
    import redis
except ImportError:
    redis = None

ACTIVE_STATES = ("RISING", "STANDING", "LOWERING", "UP", "PEAK", "DOWN")
RELEVANT_KPTS = (
    (5, "hombro_i"), (6, "hombro_d"),
    (11, "cadera_i"), (12, "cadera_d"),
    (13, "rodilla_i"), (14, "rodilla_d"),
    (15, "tobillo_i"), (16, "tobillo_d"),
)
RELEVANT_NAMES = set(name for _, name in RELEVANT_KPTS)
IGNORE_ALERTS = set((
    "Repeticion OK",
    "Junta los pies para terminar",
    "Sientate para iniciar",
    "Ponte de pie con las piernas juntas",
    "Ponte dentro del cuadro",
    "Estira la rodilla",
))


def normalize_frame_doc(raw, dt_override=None, eval_override=None, patient_default="Paciente_01", session_default=""):
    ej = (raw.get("ejercicio") or "").upper()
    if "cinematica" in raw:
        cin = raw["cinematica"]
    else:
        ang = raw.get("angulos") or {}
        if ej == "STS":
            cin = {
                "angulo_rodilla": round(float(ang.get("knee", 0.0)), 1),
                "velocidad_angular_rodilla": round(float(ang.get("omega", 0.0)), 1),
                "inclinacion_tronco": round(float(ang.get("trunk", 0.0)), 1),
            }
        elif ej == "ABD":
            cin = {
                "angulo_abduccion_i": round(float(ang.get("abd_I", 0.0)), 1),
                "angulo_abduccion_d": round(float(ang.get("abd_D", 0.0)), 1),
                "velocidad_angular_cadera": round(float(ang.get("omega", 0.0)), 1),
                "inclinacion_tronco": round(float(ang.get("trunk", 0.0)), 1),
            }
        else:
            cin = dict((k, round(float(v), 1)) for k, v in ang.items())

    lms = []
    if "landmarks_2d" in raw:
        for p in raw["landmarks_2d"]:
            if p.get("punto") in RELEVANT_NAMES:
                lms.append(p)
    else:
        kpts = raw.get("keypoints") or []
        for idx, name in RELEVANT_KPTS:
            if idx < len(kpts):
                pt = kpts[idx]
                lms.append({
                    "punto": name,
                    "x": round(float(pt[0]), 1),
                    "y": round(float(pt[1]), 1),
                    "conf": round(float(pt[2]), 2),
                })

    ts = raw.get("timestamp_ms", raw.get("ts_ms"))
    dt_val = dt_override if dt_override is not None else raw.get("delta_t_ms", raw.get("dt_ms"))
    out = {
        "paciente": raw.get("paciente") or patient_default or "Paciente_01",
        "sesion": raw.get("sesion") or raw.get("session_id") or session_default,
        "timestamp_ms": ts,
        "delta_t_ms": round(float(dt_val), 1) if dt_val is not None else 0.0,
        "frame_id": raw.get("frame_id", raw.get("frame")),
        "ejercicio": ej or None,
        "estado": raw.get("estado"),
        "cinematica": cin,
        "landmarks_2d": lms,
    }
    ev = eval_override if eval_override is not None else raw.get("evaluacion")
    if ev:
        out["evaluacion"] = ev
    return out


def format_frame_doc(d):
    cin_lines = ['    "%s": %s' % (k, json.dumps(v)) for k, v in d["cinematica"].items()]
    lm_lines = [
        '    {"punto": "%s", "x": %s, "y": %s, "conf": %s}'
        % (p["punto"], json.dumps(p["x"]), json.dumps(p["y"]), json.dumps(p["conf"]))
        for p in d["landmarks_2d"]
    ]
    ev_block = ""
    if d.get("evaluacion"):
        ev = d["evaluacion"]
        ev_block = (
            '  "evaluacion": {\n'
            '    "repeticion": %s,\n'
            '    "resultado": %s,\n'
            '    "motivo": %s\n'
            '  },\n' % (
                json.dumps(ev.get("repeticion")),
                json.dumps(ev.get("resultado")),
                json.dumps(ev.get("motivo"), ensure_ascii=False),
            )
        )
    hdr_block = ""
    if d.get("paciente"):
        hdr_block += '  "paciente": %s,\n' % json.dumps(d.get("paciente"), ensure_ascii=False)
    if d.get("sesion"):
        hdr_block += '  "sesion": %s,\n' % json.dumps(d.get("sesion"), ensure_ascii=False)
    return (
        "{\n"
        "%s"
        '  "timestamp_ms": %s,\n'
        '  "delta_t_ms": %s,\n'
        '  "frame_id": %s,\n'
        '  "ejercicio": %s,\n'
        '  "estado": %s,\n'
        "%s"
        '  "cinematica": {\n%s\n  },\n'
        '  "landmarks_2d": [\n%s\n  ]\n'
        "}" % (
            hdr_block,
            json.dumps(d["timestamp_ms"]),
            json.dumps(d["delta_t_ms"]),
            json.dumps(d["frame_id"]),
            json.dumps(d["ejercicio"]),
            json.dumps(d["estado"]),
            ev_block,
            ",\n".join(cin_lines),
            ",\n".join(lm_lines),
        )
    )


def filter_state_changes(raw_docs, keep_all=False, patient_default="Paciente_01", session_default=""):
    if raw_docs and any("evaluacion" in d for d in raw_docs):
        return [normalize_frame_doc(d, patient_default=patient_default, session_default=session_default)
                for d in raw_docs]

    docs = []
    rep_buf = []
    cycle_reasons = []
    prev_ej, prev_st, prev_ts = None, None, None
    prev_ok, prev_fail = 0, 0

    for raw in raw_docs:
        ej = (raw.get("ejercicio") or "").upper()
        st = raw.get("estado")
        trk = raw.get("tracking", True)
        has_kpts = bool(raw.get("landmarks_2d") or raw.get("keypoints"))
        if not (trk and ej in ("STS", "ABD") and has_kpts):
            prev_ej, prev_st, prev_ts = None, None, None
            rep_buf, cycle_reasons = [], []
            continue

        cur_ok = int(raw.get("ok", prev_ok))
        cur_fail = int(raw.get("fallidas", prev_fail))
        if st == "IDLE" or ej != prev_ej:
            prev_ok, prev_fail = cur_ok, cur_fail

        if st in ACTIVE_STATES or st in ("SIT", "REST"):
            for al in raw.get("alertas") or []:
                if al not in IGNORE_ALERTS and al not in cycle_reasons:
                    cycle_reasons.append(al)

        if keep_all:
            if st in ACTIVE_STATES:
                docs.append(normalize_frame_doc(raw, patient_default=patient_default, session_default=session_default))
            continue

        if (ej, st) == (prev_ej, prev_st):
            prev_ok, prev_fail = cur_ok, cur_fail
            continue

        is_active = st in ACTIVE_STATES
        is_closing = (st in ("SIT", "REST")) and (prev_ej == ej) and (prev_st in ACTIVE_STATES)
        ts = raw.get("timestamp_ms", raw.get("ts_ms"))

        if is_active:
            if prev_ej != ej or prev_st not in ACTIVE_STATES or prev_ts is None or ts is None:
                dt = 0.0
                rep_buf = []
                cycle_reasons = [al for al in (raw.get("alertas") or []) if al not in IGNORE_ALERTS]
            else:
                dt = float(ts - prev_ts)
            rep_buf.append(normalize_frame_doc(raw, dt_override=dt,
                                               patient_default=patient_default, session_default=session_default))
            prev_ts = ts
        elif is_closing:
            dt = float(ts - prev_ts) if (ts is not None and prev_ts is not None) else 0.0
            d_ok = cur_ok - prev_ok
            d_fail = cur_fail - prev_fail
            ev = raw.get("evaluacion")
            if ev is None and (d_ok > 0 or d_fail > 0):
                ok_rep = d_ok > 0
                ev = {
                    "repeticion": cur_ok + cur_fail,
                    "resultado": "CORRECTA" if ok_rep else "INCORRECTA",
                    "motivo": "Ejecucion correcta" if ok_rep else (
                        "; ".join(cycle_reasons) if cycle_reasons else "Falla biomecanica en la repeticion"
                    ),
                }
            if ev is not None:
                docs.extend(rep_buf)
                docs.append(normalize_frame_doc(raw, dt_override=dt, eval_override=ev,
                                                patient_default=patient_default, session_default=session_default))
            rep_buf, cycle_reasons = [], []
            prev_ts = ts

        prev_ej, prev_st = ej, st
        prev_ok, prev_fail = cur_ok, cur_fail
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="redis://127.0.0.1:6379/0")
    ap.add_argument("--list", action="store_true", help="Listar sesiones en Redis")
    ap.add_argument("--delete", default="", help="Borrar una sesion especifica de Redis (ej. 20261028_122747)")
    ap.add_argument("--clear-all", action="store_true", help="Borrar TODAS las sesiones guardadas en Redis")
    ap.add_argument("--session", default="", help="Exportar solo este ID de sesion (por defecto exporta TODAS)")
    ap.add_argument("--patient", default="", help="Filtrar por nombre de paciente (ej. Paciente_02)")
    ap.add_argument("--in-file", default="", help="Convertir/filtrar un archivo .jsonl/.json1 local")
    ap.add_argument("--all-frames", action="store_true",
                    help="Guardar todos los cuadros activos en vez de solo los cambios de estado")
    ap.add_argument("--reps", action="store_true", help="Exportar resumen de repeticiones")
    ap.add_argument("--out", default="bd_pacientes.json1", help="Archivo de salida (.json1, .jsonl, .json o .csv)")
    a = ap.parse_args()

    docs = []
    if a.in_file:
        raw_docs = []
        with open(a.in_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        dec = json.JSONDecoder()
        idx, n = 0, len(content)
        while idx < n:
            while idx < n and content[idx].isspace():
                idx += 1
            if idx >= n:
                break
            obj, idx = dec.raw_decode(content, idx)
            raw_docs.append(obj)
        print("%d documentos leidos de %s" % (len(raw_docs), a.in_file))
        docs = filter_state_changes(raw_docs, keep_all=a.all_frames,
                                    patient_default=a.patient or "Paciente_01")
    else:
        if redis is None:
            raise RuntimeError("Instala redis (pip3 install redis==3.5.3) o usa --in-file")
        r = redis.Redis.from_url(a.url, decode_responses=True)
        sessions = sorted(r.smembers("sts:sessions"))

        if a.clear_all:
            for sid in sessions:
                r.delete("sts:%s:frames" % sid, "sts:%s:reps" % sid, "sts:%s:meta" % sid)
            r.delete("sts:sessions")
            print("Se borraron TODAS las sesiones (%d) de Redis." % len(sessions))
            return

        if a.delete:
            sid = a.delete
            r.delete("sts:%s:frames" % sid, "sts:%s:reps" % sid, "sts:%s:meta" % sid)
            r.srem("sts:sessions", sid)
            print("Sesion '%s' eliminada de Redis." % sid)
            return

        if a.list:
            if not sessions:
                print("No hay sesiones guardadas en Redis.")
            for sid in sessions:
                meta = r.hgetall("sts:%s:meta" % sid)
                print("%s | cuadros=%d reps=%d | paciente=%s | inicio_ms=%s" % (
                    sid, r.xlen("sts:%s:frames" % sid), r.xlen("sts:%s:reps" % sid),
                    meta.get("patient", ""), meta.get("started_ms", "?")))
            return

        target_sessions = [a.session] if a.session else sessions
        if not target_sessions:
            print("No hay sesiones guardadas en Redis.")
            return

        total_raw = 0
        for sid in target_sessions:
            meta = r.hgetall("sts:%s:meta" % sid)
            pat = meta.get("patient") or "Paciente_01"
            if a.patient and pat != a.patient:
                continue
            key = "sts:%s:%s" % (sid, "reps" if a.reps else "frames")
            raw_sid, last = [], "-"
            while True:
                chunk = r.xrange(key, min=last, count=1000)
                if last != "-":
                    chunk = chunk[1:]
                if not chunk:
                    break
                for _id, f in chunk:
                    raw_sid.append(json.loads(f["doc"]))
                last = chunk[-1][0]
            total_raw += len(raw_sid)
            if a.reps:
                docs.extend(raw_sid)
            else:
                docs.extend(filter_state_changes(raw_sid, keep_all=a.all_frames,
                                                 patient_default=pat, session_default=sid))
        print("%d documentos exportados de %d sesion(es) (%d cuadros leidos)"
              % (len(docs), len(target_sessions), total_raw))

    if a.reps:
        with open(a.out, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
        print("Guardado en %s" % a.out)
        return

    if a.out.endswith(".csv"):
        cols = ["paciente", "sesion", "timestamp_ms", "delta_t_ms", "frame_id", "ejercicio",
                "estado", "evaluacion", "cinematica", "landmarks_2d"]
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for d in docs:
                w.writerow([json.dumps(d.get(c), ensure_ascii=False)
                            if isinstance(d.get(c), (dict, list)) else d.get(c)
                            for c in cols])
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(format_frame_doc(d) + "\n")
    print("Guardado en %s" % a.out)


if __name__ == "__main__":
    main()
