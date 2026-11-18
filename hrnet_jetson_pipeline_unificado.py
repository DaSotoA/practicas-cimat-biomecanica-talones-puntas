#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Unificado HRNetPose COCO-WholeBody (TensorRT FP16) para NVIDIA Jetson Nano 4GB
(JetPack 4.6, Python 3.6). Camara: ELP USBFHD05MT-KL36IR (UVC, MJPEG)

Unifica lo mejor de hrnet_jetson_pipeline_STSV5.py y hrnet_jetson_pipeline_Heelsv4.py:
  1. Ejercicios soportados en un mismo pipeline (4 ejercicios):
     - [S] STS     : Sit-to-Stand (vista lateral/oblicua, correccion de perspectiva sagital)
     - [A] ABD     : Abduccion de Cadera (vista frontal/oblicua, rango adaptado por edad)
     - [H] TALONES : Elevacion de Talones (calibracion de linea base, alineacion y control de bajada)
     - [T] PUNTAS  : Elevacion de Puntas (calibracion de linea base, alineacion y control de bajada)
     - [G] AUTO    : Auto-deteccion con clasificador temporal (sklearn)
  2. ProtocolManager (protocolos.json editable + compatibilidad con pacientes.json):
     - Asignacion automatica por edad de dias de la semana (ej. Lunes-Miercoles-Viernes o Lunes-Jueves),
       nivel (Avanzado / Intermedio / Basico), series, repeticiones, descanso y rango minimo de ABD.
     - Deteccion automatica del dia de la semana ("DIA DE DESCANSO", "EJERCICIOS COMPLETOS / SESION
       COMPLETADA" o "EJERCICIOS PENDIENTES").
     - Ingreso interactivo de paciente y edad en pantalla con [I] (tanto en el Home como en camara).
  3. HUD completo en pantalla:
     - Encabezado superior con el Plan del Paciente, nivel, estado de hoy y avance por ejercicio.
     - Panel biomecanico del ejercicio activo (angulos, velocidad angular deg/s, fase, repeticiones
       buenas/malas/caidas, calibracion y alertas).
     - HUD tecnico inferior izquierdo con Camera FPS, Loop FPS, tiempo de inferencia (ms),
       estado TRACK/SEARCH y conteo de keypoints validos (`kpts`).
     - Barra inferior de controles rapidos.
  4. Base de datos NoSQL no bloqueante (FrameDB -> Redis Streams + bd_pacientes.json1) y CSV (SessionLogger):
     - Guarda los 8 keypoints corporales en STS/ABD y los 14 keypoints (cuerpo + pies) en TALONES/PUNTAS.

Uso:
  python3 hrnet_jetson_pipeline_unificado.py --engine hrnet_wb_fp16.engine \
      --cam nvv4l2 --width 1024 --height 768 --fps 30 --protocols protocolos.json
"""

import argparse
from collections import deque
import csv
import datetime
import json
import os
import pickle
import queue
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (crea el contexto CUDA primario)
import pycuda.driver as cuda
import tensorrt as trt

try:                     # opcional: solo hace falta para cargar el clasificador sklearn
    import joblib
except ImportError:      # se usara pickle como alternativa
    joblib = None

try:                     # opcional: base de datos no relacional (Redis Streams)
    import redis as redis_lib
except ImportError:
    redis_lib = None

N_KPTS = 23   # 17 cuerpo + 6 pie (los canales 23..132 son cara/manos y se ignoran)

TRACKED_LANDMARKS = [
    (5,  "hombro_i"),  (6,  "hombro_d"),
    (11, "cadera_i"),  (12, "cadera_d"),
    (13, "rodilla_i"), (14, "rodilla_d"),
    (15, "tobillo_i"), (16, "tobillo_d"),
    (17, "puntaG_i"),  (18, "puntaP_i"), (19, "talon_i"),
    (20, "puntaG_d"),  (21, "puntaP_d"), (22, "talon_d"),
]

BODY_LANDMARKS = TRACKED_LANDMARKS[:8]

EX_IDX = {
    "I": dict(shoulder=5,  hip=11, knee=13, ankle=15, toe_big=17, toe_small=18, heel=19),
    "D": dict(shoulder=6,  hip=12, knee=14, ankle=16, toe_big=20, toe_small=21, heel=22),
}

GREEN, ORANGE, RED = (0, 255, 0), (0, 165, 255), (0, 0, 255)


# --------------------------------------------------------------------------- #
# 1. CAPTURA DE CAMARA (hilo independiente)
# --------------------------------------------------------------------------- #


class USBCameraStream(object):
    """Captura continua en un hilo. read() devuelve siempre el frame mas reciente."""

    def __init__(self, device="/dev/video0", width=1280, height=720, fps=30,
                 backend="v4l2"):
        self.device = device
        self.width, self.height, self.req_fps = width, height, fps
        self.backend = backend      # v4l2 | nvv4l2 | nvjpeg | jpegdec

        self.cap = None
        self.fps = 0.0                      # FPS reales de captura (medidos)
        self.alive = False

        self._frame = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = False
        self._thread = None

    def _gst_pipeline(self, w=None, h=None, strict_fps=True):
        """drop=1 + max-buffers=1 + sync=false: nunca se acumulan buffers viejos."""
        w = w or self.width
        h = h or self.height
        if strict_fps:
            src = ("v4l2src device={dev} ! "
                   "image/jpeg, width={w}, height={h}, framerate={fps}/1 ! ").format(
                       dev=self.device, w=w, h=h, fps=self.req_fps)
        else:
            src = ("v4l2src device={dev} ! "
                   "image/jpeg, width={w}, height={h} ! ").format(
                       dev=self.device, w=w, h=h)
        sink = "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        pipes = {
            "nvv4l2": src + "jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! "
                            "video/x-raw, format=BGRx ! " + sink,
            "nvjpeg": src + "nvjpegdec ! video/x-raw, format=I420 ! " + sink,
            "jpegdec": src + "jpegdec ! " + sink,
        }
        return pipes[self.backend]

    def _open(self):
        if self.backend != "v4l2":
            res_trials = [(self.width, self.height)]
            if (self.width, self.height) != (1280, 720):
                res_trials.append((1280, 720))
            for rw, rh in res_trials:
                for strict_fps in (True, False):
                    pipeline = self._gst_pipeline(w=rw, h=rh, strict_fps=strict_fps)
                    print("[CAM] GStreamer (%s):\n      %s" % (self.backend, pipeline))
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        return cap
                    cap.release()
                    time.sleep(0.3)
            print("[CAM] GStreamer fallo; usando V4L2 directo con MJPG.")

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)   # Re-aplicar MJPG: V4L2 suele resetear a YUYV (3-5 FPS) al cambiar W/H
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara %s" % self.device)
        self._running = True
        self.alive = True
        self._thread = threading.Thread(target=self._update, name="cam")
        self._thread.daemon = True
        self._thread.start()
        return self

    def _update(self):
        count, t0, fails = 0, time.time(), 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            count += 1
            dt = time.time() - t0
            if dt >= 1.0:
                self.fps = count / dt
                count, t0 = 0, time.time()
        self.alive = False
        with self._cond:
            self._cond.notify_all()

    def read(self, last_id=-1, timeout=0.2):
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout)
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame, self._frame_id

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- #
# 2. INFERENCIA TENSORRT
# --------------------------------------------------------------------------- #


class TRTInference(object):
    """Wrapper de un engine TensorRT con 1 input y 1 output (shapes estaticos)."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("No se pudo deserializar %s (version de TRT distinta?)"
                               % engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host, self.dev = {}, {}
        self.input_idx = self.output_idx = None

        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            if -1 in shape:
                raise RuntimeError("Engine con shapes dinamicos; reconstruir con shapes estaticos.")
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            self.host[i] = host_mem.reshape(shape)
            self.dev[i] = dev_mem
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idx = i

        self.input_view = self.host[self.input_idx]
        self.input_shape = self.input_view.shape
        self.output_shape = self.host[self.output_idx].shape
        self.input_dtype = self.input_view.dtype
        self.output_dtype = self.host[self.output_idx].dtype
        self._out_flat = self.host[self.output_idx].reshape(-1)
        self._spatial_size = int(self.output_shape[2] * self.output_shape[3]) if len(self.output_shape) == 4 else 0

    def infer(self, max_kpts=None):
        """Copia solo los primeros max_kpts canales (23 de 133) en D2H para ahorrar ~83% de ancho de banda."""
        i, o = self.input_idx, self.output_idx
        cuda.memcpy_htod_async(self.dev[i], self.host[i], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        if max_kpts is not None and self._spatial_size > 0 and max_kpts < self.output_shape[1]:
            n_elems = int(max_kpts) * self._spatial_size
            cuda.memcpy_dtoh_async(self._out_flat[:n_elems], self.dev[o], self.stream)
        else:
            cuda.memcpy_dtoh_async(self.host[o], self.dev[o], self.stream)
        self.stream.synchronize()
        return self.host[o]


# --------------------------------------------------------------------------- #
# 3. PRE / POST PROCESADO
# --------------------------------------------------------------------------- #


class PosePreprocessor(object):
    """warpAffine(ROI -> entrada de la red) -> BGR2RGB -> (x-mean)/std -> HWC->CHW."""

    NORMS = {
        "imagenet": (np.array([0.485, 0.456, 0.406], np.float32) * 255.0,
                     np.array([0.229, 0.224, 0.225], np.float32) * 255.0),
        "zero_one": (np.zeros(3, np.float32), np.full(3, 255.0, np.float32)),
        "minus1_1": (np.full(3, 127.5, np.float32), np.full(3, 127.5, np.float32)),
        "raw": (np.zeros(3, np.float32), np.ones(3, np.float32)),
    }

    def __init__(self, in_w, in_h, to_rgb=True, norm="imagenet"):
        self.in_w, self.in_h = in_w, in_h
        self.to_rgb = to_rgb
        self._warp = np.zeros((in_h, in_w, 3), np.uint8)
        self._rgb = np.empty((in_h, in_w, 3), np.uint8)
        self._f32 = np.empty((in_h, in_w, 3), np.float32)
        self._chw = np.empty((3, in_h, in_w), np.float32)
        mean, std = self.NORMS[norm]
        self.MEAN = mean
        self._inv_std = (1.0 / std).astype(np.float32)
        self._M = np.zeros((2, 3), np.float64)

    def run(self, frame, roi, dst_chw):
        x0, y0, rw, rh = roi
        sx = self.in_w / float(rw)
        sy = self.in_h / float(rh)
        M = self._M
        M[0, 0], M[0, 1], M[0, 2] = sx, 0.0, -x0 * sx
        M[1, 0], M[1, 1], M[1, 2] = 0.0, sy, -y0 * sy
        cv2.warpAffine(frame, M, (self.in_w, self.in_h), dst=self._warp,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))

        if self.to_rgb:
            cv2.cvtColor(self._warp, cv2.COLOR_BGR2RGB, dst=self._rgb)
            np.copyto(self._f32, self._rgb)
        else:
            np.copyto(self._f32, self._warp)

        self._f32 -= self.MEAN
        self._f32 *= self._inv_std
        np.copyto(self._chw, self._f32.transpose(2, 0, 1))
        np.copyto(dst_chw[0], self._chw)


class OneEuroFilter(object):
    """Filtro 1 Euro vectorizado (N,2) con suavizado reforzado en puntos de pie (17..22)."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0, foot_smoothing=True):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.foot_smoothing = bool(foot_smoothing)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        mc = np.full_like(dx_hat, self.min_cutoff)
        bt = np.full_like(dx_hat, self.beta)
        if self.foot_smoothing and x.shape[0] >= 23:
            # Los puntos 17..22 (puntas y talones) tienen mas ruido de cuantizacion en heatmaps 64x48
            mc[17:23] = self.min_cutoff * 0.42
            bt[17:23] = self.beta * 0.25
        cutoff = mc + bt * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = (a * x + (1.0 - a) * self.x_prev).astype(np.float32)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def decode_heatmaps(hm, roi):
    """Heatmaps (K,h,w) -> keypoints (K,2) en coordenadas del FRAME + confianza (K,).
    Convierte primero el slice (23,64,48) a float32 contiguo para usar SIMD NEON en argmax."""
    hm32 = np.ascontiguousarray(hm, dtype=np.float32)
    K, h, w = hm32.shape
    flat = hm32.reshape(K, -1)
    idx = flat.argmax(axis=1)
    ar = np.arange(K)
    conf = flat[ar, idx].copy()
    conf[~np.isfinite(conf)] = 0.0

    xs = idx % w
    ys = idx // w

    xl = hm32[ar, ys, np.clip(xs - 1, 0, w - 1)]
    xr = hm32[ar, ys, np.clip(xs + 1, 0, w - 1)]
    yu = hm32[ar, np.clip(ys - 1, 0, h - 1), xs]
    yd = hm32[ar, np.clip(ys + 1, 0, h - 1), xs]
    dx = np.sign(xr - xl) * 0.25 * ((xs > 0) & (xs < w - 1))
    dy = np.sign(yd - yu) * 0.25 * ((ys > 0) & (ys < h - 1))

    x0, y0, rw, rh = roi
    kx = x0 + (xs + dx) * (rw / float(w))
    ky = y0 + (ys + dy) * (rh / float(h))
    return np.stack([kx, ky], axis=1).astype(np.float32), conf


def make_roi(cx, cy, h, aspect):
    h = float(h)
    w = h * aspect
    return (cx - w / 2.0, cy - h / 2.0, w, h)


class ROITracker(object):
    """Seguimiento de una persona en coordenadas globales del frame."""

    def __init__(self, aspect, thr=0.3, margin=1.30, alpha=0.6, min_kpts=6,
                 min_score=0.35, lost_tolerance=5, enabled=True):
        self.aspect, self.thr, self.margin, self.alpha = aspect, thr, margin, alpha
        self.min_kpts, self.min_score = min_kpts, min_score
        self.lost_tolerance, self.enabled = lost_tolerance, enabled
        self.state = None
        self.lost = 0
        self._search_i = 0
        self.n_good = 0

    @property
    def tracking(self):
        return self.state is not None

    def _search_roi(self, fw, fh):
        cands = [
            (0.50 * fw, 0.5 * fh, fh),
            (0.50 * fw, 0.5 * fh, 1.5 * fh),
            (0.28 * fw, 0.5 * fh, fh),
            (0.72 * fw, 0.5 * fh, fh),
            (0.50 * fw, 0.5 * fh, fw / self.aspect),
        ]
        cx, cy, h = cands[self._search_i % len(cands)]
        self._search_i += 1
        return make_roi(cx, cy, h, self.aspect)

    def initial(self, fw, fh):
        self.state = None
        self.lost = 0
        self._search_i = 0
        return self._search_roi(fw, fh)

    def update(self, kpts, conf, fw, fh):
        if not self.enabled:
            return make_roi(fw / 2.0, fh / 2.0, fh, self.aspect)

        inside = (kpts[:, 0] >= 0) & (kpts[:, 0] < fw) & (kpts[:, 1] >= 0) & (kpts[:, 1] < fh)
        good = (conf > self.thr) & inside
        n = int(good.sum())
        self.n_good = n
        score = float(conf[good].mean()) if n > 0 else 0.0
        h_max = fw / self.aspect

        if n >= self.min_kpts and score >= self.min_score:
            pts = kpts[good]
            mn, mx = pts.min(axis=0), pts.max(axis=0)
            cx, cy = (mn + mx) / 2.0
            h = max(mx[1] - mn[1], (mx[0] - mn[0]) / self.aspect) * self.margin
            h = min(max(h, 0.30 * fh), h_max)
            new = np.array([cx, cy, h], np.float64)
            if self.state is None:
                self.state = new
            else:
                self.state = self.alpha * new + (1.0 - self.alpha) * self.state
            self.lost = 0
        else:
            self.lost += 1
            if self.state is not None and self.lost <= self.lost_tolerance:
                self.state[2] = min(self.state[2] * 1.15, h_max)
            else:
                self.state = None
                return self._search_roi(fw, fh)

        return make_roi(self.state[0], self.state[1], self.state[2], self.aspect)


# --------------------------------------------------------------------------- #
# 3b. PIE PLANO / LEVANTADO
# --------------------------------------------------------------------------- #


class FootMonitor(object):
    """Decide si cada pie esta plano usando el angulo del vector talon -> puntas."""

    SIDES = (("I", 19, 17, 18), ("D", 22, 20, 21))

    def __init__(self, max_deg=12.0, thr=0.3, min_len=10.0, hold=3):
        self.max_deg, self.thr, self.min_len, self.hold = max_deg, thr, min_len, hold
        self.reset()

    def reset(self):
        self.state = dict((s[0], None) for s in self.SIDES)
        self._cnt = dict((s[0], 0) for s in self.SIDES)

    def update(self, kpts, conf):
        res = {}
        for side, h, t1, t2 in self.SIDES:
            toes = [kpts[i] for i in (t1, t2) if conf[i] > self.thr]
            raw, ang = None, 0.0
            if conf[h] > self.thr and toes:
                v = np.mean(toes, axis=0) - kpts[h]
                if float(np.hypot(v[0], v[1])) >= self.min_len:
                    ang = float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]) + 1e-6)))
                    raw = ang < self.max_deg

            if raw is None:
                self.state[side] = None
                self._cnt[side] = 0
            elif self.state[side] is None:
                self.state[side] = raw
                self._cnt[side] = 0
            elif raw == self.state[side]:
                self._cnt[side] = 0
            else:
                self._cnt[side] += 1
                if self._cnt[side] >= self.hold:
                    self.state[side] = raw
                    self._cnt[side] = 0
            res[side] = (self.state[side], ang)
        return res


# --------------------------------------------------------------------------- #
# 3c. CONTADORES DE REPETICIONES (STS, ABD, TALONES, PUNTAS)
# --------------------------------------------------------------------------- #


def joint_angle(a, b, c):
    v1 = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    v2 = np.asarray(c, np.float64) - np.asarray(b, np.float64)
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def tilt_from_vertical(p_low, p_up):
    v = np.asarray(p_up, np.float64) - np.asarray(p_low, np.float64)
    if float(np.hypot(v[0], v[1])) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


def leg_abduction_deg(hip, ankle):
    v = np.asarray(ankle, np.float64) - np.asarray(hip, np.float64)
    if float(np.hypot(v[0], v[1])) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(abs(v[0]), v[1])))


class SavGolDeriv(object):
    """Savitzky-Golay causal (ventana N, orden 2) con tiempos reales."""

    def __init__(self, window=5, order=2, max_gap=0.8):
        self.buf = deque(maxlen=window)
        self.order, self.max_gap = order, max_gap

    def reset(self):
        self.buf.clear()

    def push(self, t, v):
        if self.buf and (t - self.buf[-1][0]) > self.max_gap:
            self.buf.clear()
        self.buf.append((t, float(v)))
        n = len(self.buf)
        if n < 3:
            return float(v), 0.0
        ts = np.array([p[0] for p in self.buf], np.float64) - t
        vs = np.array([p[1] for p in self.buf], np.float64)
        deg = min(self.order, n - 2)
        c = np.polyfit(ts, vs, deg)
        return float(c[-1]), float(c[-2])


class Feedback(object):
    """Alertas de texto con caducidad."""

    def __init__(self, ttl=2.5):
        self.ttl = ttl
        self.items = {}

    def add(self, text, t, level="bad"):
        self.items[text] = (t + self.ttl, level)

    def active(self, t):
        for k in [k for k, (e, _) in self.items.items() if e < t]:
            del self.items[k]
        return [(k, v[1]) for k, v in sorted(self.items.items(), key=lambda kv: kv[1][0])]


class RepCounterBase(object):
    """Infraestructura comun: contadores, banderas de violacion, refractario, perdida de pose."""

    TITLE = ""

    def __init__(self, refractory_ms=400.0, kpt_thr=0.3, lost_reset_s=3.0, timeout_s=20.0):
        self.refractory_s = refractory_ms / 1000.0
        self.thr, self.lost_reset_s, self.timeout_s = kpt_thr, lost_reset_s, timeout_s
        self.listeners = []
        self.reset_counts()

    def reset_counts(self):
        self.ok = self.fail = 0
        self.last_result = "-"
        self.feedback = Feedback()
        self.cur = {}
        self.rep_eval = None
        self._hard_reset(0.0)

    def _hard_reset(self, t):
        self.state, self.state_t = "IDLE", t
        self.flags = []
        self.streaks = {}
        self.refractory_until = 0.0
        self.lost_since = None
        self._reset_filters()

    def _reset_filters(self):
        pass

    def _set_state(self, s, t):
        self.state, self.state_t = s, t

    def _streak(self, key, cond, n):
        self.streaks[key] = self.streaks.get(key, 0) + 1 if cond else 0
        return self.streaks[key] >= n

    def _flag(self, reason, t):
        if reason not in self.flags:
            self.flags.append(reason)
        self.feedback.add(reason, t, "bad")

    def _new_cycle(self):
        self.flags = []
        self.streaks = {}
        self.rep_eval = None

    def _finish(self, t, info):
        reasons = list(self.flags)
        if reasons:
            self.fail += 1
            self.last_result = "FALLIDA: " + "; ".join(reasons)
        else:
            self.ok += 1
            self.last_result = "OK"
            self.feedback.add("Repeticion OK", t, "ok")
        self.rep_eval = {
            "repeticion": int(self.ok + self.fail),
            "resultado": "CORRECTA" if not reasons else "INCORRECTA",
            "motivo": "Ejecucion correcta" if not reasons else "; ".join(reasons),
        }
        self.refractory_until = t + self.refractory_s
        print("[REP %s] #%d %s | %s%s" % (
            self.TITLE, self.ok + self.fail, "OK" if not reasons else "FALLIDA",
            " ".join("%s=%.1f" % kv for kv in sorted(info.items())),
            "" if not reasons else " | " + "; ".join(reasons)))
        for cb in self.listeners:
            try:
                cb(self.TITLE, not reasons, reasons, info)
            except Exception as e:
                print("[LOG] error en listener: %s" % e)
        self.flags = []
        self.streaks = {}

    def notify_lost(self, t):
        if self.lost_since is None:
            self.lost_since = t
        self.feedback.add("Ponte dentro del cuadro", t, "warn")
        if (t - self.lost_since) > self.lost_reset_s and self.state != "IDLE":
            self._hard_reset(t)

    def _timeout(self, t):
        if self.state not in ("IDLE", "SIT", "REST", "CALIB", "BASE") and (t - self.state_t) > self.timeout_s:
            self._flag("Tiempo excedido en la repeticion", t)
            self._finish(t, {})
            self._reset_filters()
            self._set_state("IDLE", t)
            return True
        return False

    def hud_lines(self):
        return []


class STSCounter(RepCounterBase):
    """Sit-to-Stand (vista LATERAL / sagital). Ciclo: SIT -> RISING -> STANDING -> LOWERING -> SIT."""

    TITLE = "STS"
    SIDES = {"I": (5, 11, 13, 15), "D": (6, 12, 14, 16)}

    def __init__(self, sit_enter=95.0, sit_exit=105.0, stand_enter=165.0, stand_exit=155.0,
                 trunk_max=30.0, trunk_upright=10.0, omega_max=150.0, min_attempt=125.0, **kw):
        self.sit_enter, self.sit_exit = sit_enter, sit_exit
        self.stand_enter, self.stand_exit = stand_enter, stand_exit
        self.trunk_max, self.trunk_upright = trunk_max, trunk_upright
        self.omega_max, self.min_attempt = omega_max, min_attempt
        self.side = None
        self.knee_f, self.trunk_f = SavGolDeriv(5), SavGolDeriv(5)
        super(STSCounter, self).__init__(**kw)

    def _reset_filters(self):
        self.knee_f.reset()
        self.trunk_f.reset()
        self.side = None
        self._new_cycle_vars()

    def _new_cycle_vars(self):
        self.knee_peak, self.trunk_peak = 0.0, 0.0
        self.trunk_stand_min, self.down_peak = 999.0, 0.0

    def _pick_side(self, conf):
        sc = {}
        for s, idx in self.SIDES.items():
            c = [conf[i] for i in idx]
            if min(c) > self.thr:
                sc[s] = float(np.mean(c))
        if not sc:
            return None
        best = max(sc, key=sc.get)
        if self.side in sc and sc[self.side] >= sc[best] - 0.10:
            return self.side
        return best

    def update(self, kpts, conf, t, planes=None):
        P = planes if planes is not None else IDENTITY_PLANES
        side = self._pick_side(conf)
        if side is None:
            self.notify_lost(t)
            return
        if side != self.side:
            self.knee_f.reset()
            self.trunk_f.reset()
            self.side = side
        self.lost_since = None

        sh, hip, kn, an = [kpts[i] for i in self.SIDES[side]]
        knee_raw = angle_between(P.sagittal(hip - kn), P.sagittal(an - kn))
        trunk_raw = tilt_plane(P.sagittal(sh - hip))
        if knee_raw is None or trunk_raw is None:
            self.notify_lost(t)
            return
        k, w = self.knee_f.push(t, knee_raw)
        tr, _ = self.trunk_f.push(t, trunk_raw)
        self.cur = {"knee": k, "omega": w, "trunk": tr}
        st = self.state
        if self._timeout(t):
            return

        if st == "IDLE":
            if self._streak("sit", k < self.sit_enter, 2):
                self._set_state("SIT", t)
            else:
                self.feedback.add("Sientate para iniciar", t, "warn")

        elif st == "SIT":
            if t >= self.refractory_until and k > self.sit_exit and w > 0:
                self._new_cycle()
                self._new_cycle_vars()
                self._set_state("RISING", t)

        else:
            self.knee_peak = max(self.knee_peak, k)
            self.trunk_peak = max(self.trunk_peak, tr)
            if self._streak("trunk", tr > self.trunk_max, 2):
                self._flag("Espalda muy inclinada", t)

            if st == "RISING":
                if k >= self.stand_enter:
                    self.trunk_stand_min = 999.0
                    self._set_state("STANDING", t)
                elif k < self.sit_enter:
                    if self.knee_peak >= self.min_attempt:
                        self._flag("Levantada incompleta", t)
                        self._finish(t, self._info())
                    self._set_state("SIT", t)

            elif st == "STANDING":
                if t > self.state_t:
                    self.trunk_stand_min = min(self.trunk_stand_min, tr)
                if k < self.stand_exit:
                    if 999.0 > self.trunk_stand_min > self.trunk_upright:
                        self._flag("No te erguiste al estar de pie", t)
                    self.down_peak = 0.0
                    self._set_state("LOWERING", t)

            elif st == "LOWERING":
                self.down_peak = max(self.down_peak, -w)
                if self._streak("fall", -w > self.omega_max, 1):
                    self._flag("Caida brusca: controla el descenso", t)
                if k < self.sit_enter:
                    self._finish(t, self._info())
                    self._set_state("SIT", t)
                elif k >= self.stand_enter:
                    self._set_state("STANDING", t)

    def _info(self):
        return {"knee_peak": self.knee_peak, "trunk_peak": self.trunk_peak,
                "trunk_stand_min": min(self.trunk_stand_min, 999.0),
                "omega_down_peak": self.down_peak}

    def hud_lines(self):
        c = self.cur
        if not c:
            return ["STS [%s] buscando pose lateral..." % self.state]
        return [
            "STS [%s] lado=%s | REPS OK: %d  FALLIDAS: %d" % (
                self.state, self.side or "?", self.ok, self.fail),
            "rodilla %.0f deg   w %+.0f deg/s   tronco %.0f deg" % (
                c["knee"], c["omega"], c["trunk"]),
        ]


class AbdCounter(RepCounterBase):
    """Abduccion de cadera de pie (vista FRONTAL). Ciclo: REST -> UP -> PEAK -> DOWN -> REST."""

    TITLE = "ABD"
    LEGS = {"I": (11, 13, 15), "D": (12, 14, 16)}
    NEED = (5, 6, 11, 12, 15, 16)

    def __init__(self, rest_enter=15.0, rest_exit=20.0, peak_min=45.0, peak_down=40.0,
                 trunk_max=15.0, knee_min=165.0, knee_margin=15.0, omega_max=120.0,
                 ratio_tol=0.30, attempt_min=25.0, **kw):
        self.rest_enter, self.rest_exit = rest_enter, rest_exit
        self.peak_min, self.peak_down = peak_min, peak_down
        self.trunk_max, self.knee_min, self.knee_margin = trunk_max, knee_min, knee_margin
        self.omega_max, self.ratio_tol, self.attempt_min = omega_max, ratio_tol, attempt_min
        self.abd_f = dict((s, SavGolDeriv(5)) for s in self.LEGS)
        self.knee_f = dict((s, SavGolDeriv(5)) for s in self.LEGS)
        self.trunk_f, self.ratio_f = SavGolDeriv(5), SavGolDeriv(5)
        super(AbdCounter, self).__init__(**kw)

    def _reset_filters(self):
        for f in list(self.abd_f.values()) + list(self.knee_f.values()):
            f.reset()
        self.trunk_f.reset()
        self.ratio_f.reset()
        self.base_buf = deque(maxlen=12)
        self.active = None
        self._new_cycle_vars()

    def _new_cycle_vars(self):
        self.peak, self.knee_min_seen, self.trunk_peak, self.down_peak = 0.0, 999.0, 0.0, 0.0

    def update(self, kpts, conf, t, planes=None):
        P = planes if planes is not None else IDENTITY_PLANES
        if any(conf[i] <= self.thr for i in self.NEED):
            self.notify_lost(t)
            return
        hipw = float(np.hypot(*(kpts[11] - kpts[12])))
        if hipw < 1.0:
            self.notify_lost(t)
            return
        self.lost_since = None

        a, kn, om = {}, {}, {}
        for s, (h, k, an) in self.LEGS.items():
            raw = abd_plane(P.frontal(kpts[an] - kpts[h]))
            if raw is None:
                self.notify_lost(t)
                return
            a[s], om[s] = self.abd_f[s].push(t, raw)
            kr = (angle_between(P.frontal(kpts[h] - kpts[k]), P.frontal(kpts[an] - kpts[k]))
                  if conf[k] > self.thr else None)
            if kr is None:
                self.knee_f[s].reset()
                kn[s] = None
            else:
                kn[s], _ = self.knee_f[s].push(t, kr)
        trunk_raw = tilt_plane(P.frontal((kpts[5] + kpts[6]) / 2.0 - (kpts[11] + kpts[12]) / 2.0))
        if trunk_raw is None:
            self.notify_lost(t)
            return
        tr, _ = self.trunk_f.push(t, trunk_raw)
        ratio, _ = self.ratio_f.push(t, float(np.hypot(*(kpts[15] - kpts[16]))) / hipw)

        st = self.state
        top = max(a, key=a.get)
        self.cur = {"abd_I": a["I"], "abd_D": a["D"], "trunk": tr, "ratio": ratio,
                    "omega": om[self.active or top]}
        if self._timeout(t):
            return

        if st == "IDLE":
            if self._streak("rest", a[top] <= self.rest_enter, 3):
                self.base_buf.clear()
                self._set_state("REST", t)
            else:
                self.feedback.add("Ponte de pie con las piernas juntas", t, "warn")

        elif st == "REST":
            if a[top] <= self.rest_exit:
                self.base_buf.append(ratio)
            if t >= self.refractory_until and a[top] > self.rest_exit:
                self.active = top
                self._new_cycle()
                self._new_cycle_vars()
                self._set_state("UP", t)

        else:
            s = self.active
            ang, w, k = a[s], om[s], kn[s]
            self.peak = max(self.peak, ang)
            self.trunk_peak = max(self.trunk_peak, tr)
            if self._streak("trunk", tr > self.trunk_max, 2):
                self._flag("Tronco inclinado lateralmente", t)
            if k is not None:
                self.knee_min_seen = min(self.knee_min_seen, k)
                if k < self.knee_min:
                    self.feedback.add("Estira la rodilla", t, "warn")
                if self._streak("knee", k < self.knee_min - self.knee_margin, 2):
                    self._flag("Rodilla flexionada", t)

            if st == "UP":
                if ang >= self.peak_min:
                    self._set_state("PEAK", t)
                elif ang <= self.rest_enter:
                    self._close(t)
                elif w < 0 and ang < self.peak - 6.0:
                    self._set_state("DOWN", t)
            elif st == "PEAK":
                if ang < self.peak_down:
                    self.down_peak = 0.0
                    self._set_state("DOWN", t)
            elif st == "DOWN":
                self.down_peak = max(self.down_peak, -w)
                if self._streak("fall", -w > self.omega_max, 1):
                    self._flag("Pierna cae bruscamente: controla el descenso", t)
                if ang >= self.peak_min:
                    self._set_state("PEAK", t)
                elif ang <= self.rest_enter:
                    if self._ankles_back(ratio):
                        self._close(t)
                    else:
                        self.feedback.add("Junta los pies para terminar", t, "warn")

    def _ankles_back(self, ratio):
        if len(self.base_buf) < 3:
            return True
        return ratio <= float(np.median(self.base_buf)) * (1.0 + self.ratio_tol)

    def _close(self, t):
        if self.peak >= self.attempt_min:
            if self.peak < self.peak_min:
                self._flag("Rango insuficiente: levanta mas la pierna", t)
            self._finish(t, {"peak": self.peak, "trunk_peak": self.trunk_peak,
                             "knee_min": min(self.knee_min_seen, 999.0),
                             "omega_down_peak": self.down_peak})
        else:
            self.flags, self.streaks = [], {}
        self.active = None
        self._set_state("REST", t)

    def hud_lines(self):
        c = self.cur
        if not c:
            return ["ABD [%s] buscando pose frontal..." % self.state]
        return [
            "ABD [%s] activa=%s (meta>=%.0f deg) | REPS OK: %d  FALLIDAS: %d" % (
                self.state, self.active or "-", self.peak_min, self.ok, self.fail),
            "abd I %.0f  D %.0f deg   w %+.0f deg/s   tronco %.0f deg   tob/cad %.2f" % (
                c["abd_I"], c["abd_D"], c["omega"], c["trunk"], c["ratio"]),
        ]


# --------------------------------------------------------------------------- #
# 3d. EVALUACION BIOMECANICA DE TALONES Y PUNTAS (de Heelsv4.py)
# --------------------------------------------------------------------------- #


class ExerciseConfig(object):
    def __init__(self, **kw):
        self.kpt_thr                = 0.25
        self.max_shoulder_sep_ratio = 0.65
        self.max_hip_sep_ratio      = 0.65
        self.min_foot_len_ratio     = 0.14
        self.max_torso_incl_deg     = 22.0
        self.min_knee_angle_deg     = 148.0
        self.check_both_knees       = False
        self.max_omega_dps          = 240.0
        self.omega_alpha            = 0.25
        self.omega_min_dt           = 1e-3
        self.omega_max_dt           = 0.8
        self.calib_frames           = 6
        self.calib_max_omega_dps    = 65.0
        self.calib_max_pitch_deg    = 45.0
        self.calib_max_std_deg      = 6.5
        self.baseline_fallback_deg  = 0.0
        self.heel_up_deg            = 8.5
        self.heel_down_deg          = 4.0
        self.toe_up_deg             = 5.5
        self.toe_down_deg           = 2.5
        self.toe_min_dy_ratio       = 0.035
        self.toe_hold               = 2
        self.min_dy_floor_px        = 2.0
        self.hold                   = 2
        self.pitch_alpha            = 0.50
        self.min_dy_px              = 4.0
        self.min_foot_px            = 10.0
        self.toe_off                = {}
        self.max_bad_frames         = 10
        self.max_bad_ratio          = 0.65
        self.drop_omega_dps         = 25.0
        self.min_desc_s             = 0.4
        self.reject_drops           = False
        self.event_show_s           = 3.0
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


class AngularVelocity(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.theta_prev = self.t_prev = None
        self.omega = 0.0

    def update(self, theta, t):
        c = self.cfg
        if self.theta_prev is None:
            self.theta_prev, self.t_prev, self.omega = theta, t, 0.0
            return 0.0, False
        dt = t - self.t_prev
        if dt > c.omega_max_dt:
            self.reset()
            self.theta_prev, self.t_prev = theta, t
            return 0.0, False
        if dt < c.omega_min_dt:
            return self.omega, abs(self.omega) > c.max_omega_dps
        raw = (theta - self.theta_prev) / dt
        self.omega = c.omega_alpha * raw + (1.0 - c.omega_alpha) * self.omega
        self.theta_prev, self.t_prev = theta, t
        return self.omega, abs(self.omega) > c.max_omega_dps


class ExerciseState(object):
    def __init__(self, cfg=None):
        self.cfg = cfg or ExerciseConfig()
        self.omega_est = AngularVelocity(self.cfg)
        self.reset()

    def reset(self):
        self.omega_est.reset()
        self.calib        = []
        self.calib_dy     = []
        self.calib_all    = []
        self.calib_dy_all = []
        self.baseline     = None
        self.phase        = "CALIB"
        self.reps         = 0
        self.rejected     = 0
        self._cnt         = 0
        self._bad_frames  = 0
        self.side         = None
        self.pitch_f      = self.dy_f = None
        self.pitch_med    = deque(maxlen=3)
        self.dy_med       = deque(maxlen=3)
        self.baseline_dy  = None
        self.peak         = 0.0
        self.t_peak       = self.t_hi = self.t_rise0 = self.t_below = None
        self.up_frames    = 0
        self.max_desc     = 0.0
        self.attempt      = self.attempt_dy = 0.0
        self.bad_reasons  = []
        self.drops        = 0
        self.event        = ""
        self.event_col    = (0, 255, 0)
        self.event_t      = -1e9
        self.last_rep     = None
        self.hold_until   = -1e9
        self.peak_logged  = False


def _pt(kpts, conf, i, thr):
    return np.asarray(kpts[i], np.float64) if conf[i] > thr else None


def angle_3pts(a, b, c):
    v1, v2 = np.asarray(a, np.float64) - b, np.asarray(c, np.float64) - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def select_side(kpts, conf, side="auto", prev_side=None):
    if side in EX_IDX:
        return side
    score = dict((s, sum(float(conf[i]) for i in EX_IDX[s].values())) for s in EX_IDX)
    best = max(score, key=score.get)
    # Histeresis de lado ("pegajoso"): evita que cambie I<->D a mitad de la calibracion
    if prev_side in score and score[prev_side] >= score[best] - 0.65:
        return prev_side
    return best


def _foot_vec(kpts, conf, side, cfg):
    k, thr = EX_IDX[side], cfg.kpt_thr
    heel   = _pt(kpts, conf, k["heel"],      thr)
    tb     = _pt(kpts, conf, k["toe_big"],   thr)
    ts     = _pt(kpts, conf, k["toe_small"], thr)
    if heel is None:
        return None
    o = cfg.toe_off.get(side)
    if tb is not None and ts is not None:
        off = ts - tb
        cfg.toe_off[side] = off if o is None else 0.9 * o + 0.1 * off
        toe = 0.5 * (tb + ts)
    elif o is None:
        return None
    elif tb is not None:
        toe = tb + 0.5 * o
    elif ts is not None:
        toe = ts - 0.5 * o
    else:
        return None
    return toe - heel


def foot_pitch_deg(kpts, conf, side, cfg):
    v = _foot_vec(kpts, conf, side, cfg)
    if v is None:
        return None, None
    length = float(np.hypot(v[0], v[1]))
    if length < cfg.min_foot_px:
        return None, None
    pitch = float(np.degrees(np.arctan2(v[1], abs(v[0]) + 1e-6)))
    return pitch, length


def foot_dy_px(kpts, conf, side, cfg):
    v = _foot_vec(kpts, conf, side, cfg)
    return None if v is None else float(v[1])


def validar_orientacion_lateral(kpts, conf, side, cfg):
    thr, k = cfg.kpt_thr, EX_IDX[side]
    info, why = {}, []
    sh = _pt(kpts, conf, k["shoulder"], thr)
    hp = _pt(kpts, conf, k["hip"],      thr)
    if sh is not None and hp is not None:
        torso = float(np.linalg.norm(sh - hp))
        if torso > 10.0:
            info["torso_px"] = torso
            sl = _pt(kpts, conf, 5, thr)
            sr = _pt(kpts, conf, 6, thr)
            if sl is not None and sr is not None:
                r = abs(sl[0] - sr[0]) / torso
                info["shoulder_sep"] = r
                if r > cfg.max_shoulder_sep_ratio:
                    why.append("gira mas de perfil (hombros)")
            _, flen = foot_pitch_deg(kpts, conf, side, cfg)
            if flen is not None:
                r = flen / torso
                info["foot_len_ratio"] = r
                if r < cfg.min_foot_len_ratio:
                    why.append("pie muy de frente")
    pitch, _ = foot_pitch_deg(kpts, conf, side, cfg)
    if pitch is None:
        why.append("pie no visible")
    return len(why) == 0, info, why


def validar_alineacion_articular(kpts, conf, side, cfg):
    thr, k = cfg.kpt_thr, EX_IDX[side]
    info, why = {}, []
    sh = _pt(kpts, conf, k["shoulder"], thr)
    hp = _pt(kpts, conf, k["hip"],      thr)
    if sh is not None and hp is not None:
        v = sh - hp
        incl = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))
        info["torso_incl"] = incl
        if incl > cfg.max_torso_incl_deg:
            why.append("espalda inclinada (%.0f > %.0f)" % (incl, cfg.max_torso_incl_deg))
    h  = _pt(kpts, conf, k["hip"],   thr)
    kn = _pt(kpts, conf, k["knee"],  thr)
    an = _pt(kpts, conf, k["ankle"], thr)
    if h is not None and kn is not None and an is not None:
        ang = angle_3pts(h, kn, an)
        if ang is not None:
            info["knee_angle"] = ang
            if ang < cfg.min_knee_angle_deg:
                why.append("rodilla flexionada (%.0f < %.0f)" % (ang, cfg.min_knee_angle_deg))
    return len(why) == 0, info, why


def _event(state, text, col, t):
    state.event, state.event_col, state.event_t = text, col, t


def _pack(res, state, t):
    res.update(phase=state.phase, reps=state.reps, rejected=state.rejected,
               drops=state.drops, last_rep=state.last_rep,
               calibrated=state.baseline is not None,
               event=(state.event if (t - state.event_t) < state.cfg.event_show_s else ""),
               event_col=state.event_col)
    return res


def _note_bad(state, res):
    for r in res["reasons"]:
        tag = r.split(" (")[0].split(" |")[0]
        if tag not in state.bad_reasons:
            state.bad_reasons.append(tag)


def diagnose_foot(kpts, conf, side, cfg):
    k, thr = EX_IDX[side], cfg.kpt_thr
    if conf[k["heel"]] <= thr:
        return "No veo el talon (conf %.2f<%.2f)" % (conf[k["heel"]], thr)
    tc = max(conf[k["toe_big"]], conf[k["toe_small"]])
    if tc <= thr:
        return "No veo las puntas (conf %.2f<%.2f)" % (tc, thr)
    return "Pie no medible: muy corto en imagen"


def _why_not_counted(state, up_deg, min_dy):
    if state.attempt < up_deg:
        return "NO CONTADA: subiste %.1f deg, minimo %.1f deg" % (state.attempt, up_deg)
    if state.attempt_dy < min_dy:
        return "NO CONTADA: poco desplazamiento (%.0f px): acercate" % state.attempt_dy
    return "NO CONTADA: sosten arriba un instante"


def extract_snapshot(kpts, conf, res):
    cinematica = {
        "angulo_pie":        round(float(res["metric"] if res["metric"] is not None else 0.0), 1),
        "velocidad_angular": round(float(res.get("omega", 0.0)), 1),
        "angulo_rodilla":    round(float(res.get("knee_angle", 0.0) or 0.0), 1),
        "inclinacion_tronco": round(float(res.get("torso_incl", 0.0) or 0.0), 1),
    }
    landmarks = []
    for idx, name in TRACKED_LANDMARKS:
        if idx < len(kpts):
            landmarks.append({
                "punto": name,
                "x":    round(float(kpts[idx][0]), 1),
                "y":    round(float(kpts[idx][1]), 1),
                "conf": round(float(conf[idx]),    2),
            })
    return cinematica, landmarks


def _evaluar(kpts, conf, t, state, side, sign, up_deg, down_deg, hold, dy_ratio=None,
             db_logger=None, frame_id=0, ex_label="TALONES"):
    cfg  = state.cfg
    kpts = np.asarray(kpts)
    side = select_side(kpts, conf, side, prev_side=state.side)
    if state.side is None:
        state.side = side
    elif state.side != side and state.phase == "CALIB":
        state.side = side
        state.calib, state.calib_dy = [], []
        state.calib_all, state.calib_dy_all = [], []
        state.pitch_f = state.dy_f = None
        state.pitch_med.clear()
        state.dy_med.clear()

    if t < state.hold_until:
        res = dict(side=state.side, valid=False, foot_angle=None, delta=None,
                   metric=None, omega=0.0, abrupt=False, knee_angle=None,
                   torso_incl=None, reasons=[], msg="Ponte con el pie plano...")
        return _pack(res, state, t)

    res = dict(side=state.side, valid=False, foot_angle=None, delta=None,
               metric=None, omega=0.0, abrupt=False, knee_angle=None,
               torso_incl=None, reasons=[], msg="")

    ok_lat, _, why_lat = validar_orientacion_lateral(kpts, conf, state.side, cfg)
    ok_ali, info_ali, why_ali = validar_alineacion_articular(kpts, conf, state.side, cfg)
    res["knee_angle"]  = info_ali.get("knee_angle")
    res["torso_incl"]  = info_ali.get("torso_incl")
    res["reasons"]     = why_lat + why_ali
    posture_ok         = ok_lat and ok_ali

    pitch_raw, length = foot_pitch_deg(kpts, conf, state.side, cfg)
    dy_raw = foot_dy_px(kpts, conf, state.side, cfg)
    if pitch_raw is None or dy_raw is None:
        res["msg"] = diagnose_foot(kpts, conf, state.side, cfg)
        return _pack(res, state, t)

    # Filtro anti-pico (mediana 3 en CALIB + limitador de salto y EMA en ejecucion)
    state.pitch_med.append(pitch_raw)
    state.dy_med.append(dy_raw)
    if state.baseline is None:
        p_in = float(np.median(state.pitch_med))
        d_in = float(np.median(state.dy_med))
    else:
        p_in = state.pitch_f + float(np.clip(pitch_raw - state.pitch_f, -18.0, 18.0))
        d_in = state.dy_f + float(np.clip(dy_raw - state.dy_f, -25.0, 25.0))
    if state.pitch_f is None:
        state.pitch_f, state.dy_f = p_in, d_in
    else:
        a = cfg.pitch_alpha
        state.pitch_f += a * (p_in - state.pitch_f)
        state.dy_f    += a * (d_in - state.dy_f)
    pitch, dy = state.pitch_f, state.dy_f
    res["foot_angle"] = pitch

    omega, abrupt = state.omega_est.update(pitch, t)
    res["omega"], res["abrupt"] = omega, abrupt
    if abrupt:
        res["reasons"].append("muy rapido |w|=%.0f > %.0f" % (abs(omega), cfg.max_omega_dps))

    if state.baseline is None:
        # Calibracion robusta con mediana: NO borra el progreso si un frame aislado tiene ruido
        if abs(pitch) < cfg.calib_max_pitch_deg:
            state.calib_all.append(pitch)
            state.calib_dy_all.append(dy)
            if abs(omega) < cfg.calib_max_omega_dps:
                state.calib.append(pitch)
                state.calib_dy.append(dy)
            n_prog = max(len(state.calib), len(state.calib_all) // 2)
            res["msg"] = "Calibrando pie plano (%d/%d)..." % (
                min(n_prog, cfg.calib_frames), cfg.calib_frames)
            ready_clean = (len(state.calib) >= cfg.calib_frames and
                           float(np.std(state.calib[-cfg.calib_frames:])) < cfg.calib_max_std_deg)
            ready_fallback = (len(state.calib_all) >= cfg.calib_frames + 4)
            if ready_clean or ready_fallback:
                src_p = state.calib[-cfg.calib_frames:] if len(state.calib) >= 3 else state.calib_all
                src_d = state.calib_dy[-cfg.calib_frames:] if len(state.calib_dy) >= 3 else state.calib_dy_all
                state.baseline    = float(np.median(src_p))
                state.baseline_dy = float(np.median(src_d))
                state.phase       = "BASE"
                _event(state, "PIE CALIBRADO (base %.1f deg): inicia tus repeticiones" % state.baseline,
                       GREEN, t)
        else:
            res["msg"] = "Calibrando: apoya el pie en el suelo (%d/%d)" % (
                len(state.calib), cfg.calib_frames)
        return _pack(res, state, t)

    delta     = pitch - state.baseline
    metric    = sign * delta
    metric_dy = sign * (dy - state.baseline_dy)
    rate      = sign * omega
    min_dy    = cfg.min_dy_px if dy_ratio is None else max(cfg.min_dy_floor_px, dy_ratio * length)
    res["delta"], res["metric"] = delta, metric

    if state.phase == "BASE" and abs(metric) < down_deg and abs(omega) < 35.0:
        state.baseline    = 0.97 * state.baseline    + 0.03 * pitch
        state.baseline_dy = 0.97 * state.baseline_dy + 0.03 * dy

    if state.phase == "BASE":
        if metric < 0.25 * up_deg:
            state.t_rise0 = t
            if state.attempt >= 0.5 * up_deg:
                _event(state, _why_not_counted(state, up_deg, min_dy), ORANGE, t)
            state.attempt = state.attempt_dy = 0.0
            state._cnt = 0
        else:
            state.attempt    = max(state.attempt,    metric)
            state.attempt_dy = max(state.attempt_dy, metric_dy)
            if metric >= up_deg and metric_dy >= min_dy:
                state._cnt += 1
                if state._cnt >= hold:
                    state.phase       = "UP"
                    state._cnt        = 0
                    state.peak        = metric
                    state.t_peak      = state.t_hi = t
                    state.max_desc    = 0.0
                    state.up_frames   = 0
                    state.bad_reasons = []
                    state._bad_frames = 0
                    state.peak_logged = False
                    if not posture_ok or abrupt:
                        state._bad_frames = 1
                        _note_bad(state, res)
                    state.attempt = state.attempt_dy = 0.0
                    if db_logger is not None:
                        cin, lms = extract_snapshot(kpts, conf, res)
                        db_logger.log_event(frame_id, ex_label, "UP", cin, lms)
            else:
                state._cnt = 0
                if metric < up_deg:
                    res["msg"] = "Sube mas: %.1f de %.1f deg" % (metric, up_deg)
                else:
                    res["msg"] = "Poco desplazamiento: acercate a la camara"

    elif state.phase == "UP":
        state.up_frames += 1
        if not posture_ok or abrupt:
            state._bad_frames += 1
            _note_bad(state, res)
        if metric > state.peak:
            state.peak, state.t_peak = metric, t
        if metric >= 0.9 * state.peak:
            state.t_hi = t
        if rate < -5.0 and not state.peak_logged:
            state.peak_logged = True
            if db_logger is not None:
                cin, lms = extract_snapshot(kpts, conf, res)
                db_logger.log_event(frame_id, ex_label, "PEAK", cin, lms)
        if rate < 0:
            state.max_desc = max(state.max_desc, -rate)
            if -rate > cfg.drop_omega_dps:
                res["msg"] = "Bajas muy rapido (%.0f deg/s): controla la bajada" % (-rate)

        if metric <= down_deg:
            if state._cnt == 0:
                state.t_below = t
                if db_logger is not None:
                    cin, lms = extract_snapshot(kpts, conf, res)
                    db_logger.log_event(frame_id, ex_label, "DOWN", cin, lms)
            state._cnt += 1
            if state._cnt >= hold:
                t_desc   = max(state.t_below - state.t_hi, 1e-3)
                drop_deg = max(0.9 * state.peak - down_deg, 1.0)
                w_down   = drop_deg / t_desc
                t_asc    = None if state.t_rise0 is None else state.t_peak - state.t_rise0
                w_up     = (state.peak / t_asc) if (t_asc is not None and t_asc > 1e-3) else None
                dropped  = (w_down > cfg.drop_omega_dps) or (t_desc < cfg.min_desc_s)
                state.last_rep = dict(peak=state.peak, w_down=w_down,
                                      t_desc=t_desc, w_up=w_up, dropped=dropped)

                bad_ratio = state._bad_frames / float(max(state.up_frames, 1))
                bad_post  = (state._bad_frames > cfg.max_bad_frames and
                             bad_ratio > cfg.max_bad_ratio)
                rep_num   = state.reps + state.rejected + 1

                if bad_post:
                    state.rejected += 1
                    motivo    = ", ".join(state.bad_reasons[:2]) or "mala postura"
                    resultado = "INCORRECTA"
                    _event(state, "RECHAZADA: " + motivo, RED, t)
                elif dropped and cfg.reject_drops:
                    state.rejected += 1
                    motivo    = "Pierna cae bruscamente: controla el descenso"
                    resultado = "INCORRECTA"
                    _event(state, "RECHAZADA: te dejaste caer (%.0f deg/s)" % w_down, RED, t)
                else:
                    state.reps += 1
                    if dropped:
                        state.drops += 1
                        resultado = "INCORRECTA"
                        motivo    = "Pierna cae bruscamente: controla el descenso"
                        _event(state, "REP %d contada, pero TE DEJASTE CAER (%.0f deg/s)"
                               % (state.reps, w_down), ORANGE, t)
                    else:
                        resultado = "CORRECTA"
                        motivo    = "Ejecucion correcta"
                        _event(state, "REP %d OK: bajada controlada (%.0f deg/s, %.1fs)"
                               % (state.reps, w_down, t_desc), GREEN, t)

                if db_logger is not None:
                    eval_data = {"repeticion": rep_num,
                                 "resultado":  resultado,
                                 "motivo":     motivo}
                    cin, lms = extract_snapshot(kpts, conf, res)
                    db_logger.log_event(frame_id, ex_label, "REST", cin, lms, evaluacion=eval_data)

                state.phase = "BASE"
                state._cnt  = state._bad_frames = 0
                state.peak  = 0.0
                state.attempt = state.attempt_dy = 0.0
                state.peak_logged = False
        else:
            state._cnt = 0

    res["valid"] = posture_ok and not abrupt
    return _pack(res, state, t)


def evaluar_elevacion_talones(kpts, conf, t, state, side="auto",
                              db_logger=None, frame_id=0):
    c = state.cfg
    return _evaluar(kpts, conf, t, state, side, +1.0,
                    c.heel_up_deg, c.heel_down_deg, c.hold,
                    None, db_logger, frame_id, "TALONES")


def evaluar_elevacion_puntas(kpts, conf, t, state, side="auto",
                             db_logger=None, frame_id=0):
    c = state.cfg
    return _evaluar(kpts, conf, t, state, side, -1.0,
                    c.toe_up_deg, c.toe_down_deg, c.toe_hold,
                    c.toe_min_dy_ratio, db_logger, frame_id, "PUNTAS")


class HeelToeCounter(RepCounterBase):
    """Adaptador unificado de Elevacion de Talones (TALONES) y Elevacion de Puntas (PUNTAS)
    compatible con ExerciseRouter, ProtocolManager, SessionLogger y FrameDB."""

    def __init__(self, kind="talones", cfg=None, side="auto", **kw):
        self.kind = kind.lower()
        self.TITLE = "TALONES" if self.kind == "talones" else "PUNTAS"
        self.cfg = cfg or ExerciseConfig()
        self.side_pref = side
        self.ex_state = ExerciseState(self.cfg)
        self.last_res = None
        self.frame_id = 0
        self._pending_events = []
        super(HeelToeCounter, self).__init__(kpt_thr=self.cfg.kpt_thr, **kw)

    def _hard_reset(self, t):
        super(HeelToeCounter, self)._hard_reset(t)
        if hasattr(self, "ex_state"):
            self.ex_state.reset()
            if t > 0.0:
                self.ex_state.hold_until = t + 0.25
            self.state = self.ex_state.phase
            self.last_res = None
            self._pending_events = []

    def log_event(self, frame_id, ejercicio, estado, cinematica, landmarks, evaluacion=None):
        """Recibe las transiciones UP / PEAK / DOWN / REST desde _evaluar."""
        self._pending_events.append({
            "frame_id": frame_id,
            "ejercicio": ejercicio,
            "estado": estado,
            "cinematica": cinematica,
            "landmarks": landmarks,
            "evaluacion": evaluacion,
        })

    def update(self, kpts, conf, t, planes=None):
        self.lost_since = None
        self.frame_id += 1
        self._pending_events = []
        if self.kind == "talones":
            res = evaluar_elevacion_talones(
                kpts, conf, t, self.ex_state, side=self.side_pref,
                db_logger=self, frame_id=self.frame_id
            )
        else:
            res = evaluar_elevacion_puntas(
                kpts, conf, t, self.ex_state, side=self.side_pref,
                db_logger=self, frame_id=self.frame_id
            )
        self.last_res = res
        self.state = res.get("phase", "CALIB")
        self.cur = {
            "angulo_pie": float(res["metric"] if res.get("metric") is not None else 0.0),
            "velocidad_angular": float(res.get("omega", 0.0) or 0.0),
            "angulo_rodilla": float(res.get("knee_angle", 0.0) or 0.0),
            "inclinacion_tronco": float(res.get("torso_incl", 0.0) or 0.0),
        }

        for ev in self._pending_events:
            st_ev = ev["estado"]
            self.state = st_ev
            if st_ev == "REST" and ev.get("evaluacion") is not None:
                ev_data = ev["evaluacion"]
                is_ok = (ev_data.get("resultado") == "CORRECTA")
                motivo = ev_data.get("motivo", "")
                reasons = [] if is_ok else [motivo]
                if is_ok:
                    self.ok += 1
                    self.last_result = "OK"
                    self.feedback.add("Repeticion OK", t, "ok")
                else:
                    self.fail += 1
                    self.last_result = "FALLIDA: " + motivo
                    self.feedback.add(motivo, t, "bad")
                self.rep_eval = dict(ev_data)
                lrep = self.ex_state.last_rep or {}
                info = {
                    "peak": float(lrep.get("peak", 0.0) or 0.0),
                    "omega_down_peak": float(lrep.get("w_down", 0.0) or 0.0),
                }
                for cb in self.listeners:
                    try:
                        cb(self.TITLE, is_ok, reasons, info)
                    except Exception as e:
                        print("[LOG] error en listener (%s): %s" % (self.TITLE, e))


# --------------------------------------------------------------------------- #
# 3e. PERSPECTIVA ANATOMICA Y CLASIFICADOR TEMPORAL (de STSV5.py)
# --------------------------------------------------------------------------- #

WS_REF = 0.72
WH_REF = 0.38
MIN_GAIN = 0.5


class AnatomicalPlanes(object):
    def __init__(self, e_lat=(1.0, 0.0), cos_psi=1.0, identity=False):
        self.identity = identity
        self.ex, self.ey = float(e_lat[0]), float(e_lat[1])
        self.cos_psi = float(np.clip(cos_psi, 0.0, 1.0))
        self.yaw_deg = float(np.degrees(np.arccos(self.cos_psi)))
        sin_psi = float(np.sqrt(max(0.0, 1.0 - self.cos_psi ** 2)))
        self.kf = 1.0 if identity else max(self.cos_psi, MIN_GAIN)
        self.ks = 1.0 if identity else max(sin_psi, MIN_GAIN)

    def _map(self, v, k):
        X = float(v[0]) / (k * self.ex)
        Y = -float(v[1]) + (self.ey / self.ex) * float(v[0])
        return X, Y

    def frontal(self, v):
        return self._map(v, self.kf)

    def sagittal(self, v):
        return self._map(v, self.ks)


IDENTITY_PLANES = AnatomicalPlanes(identity=True)


def project_to_anatomical_planes(kpts, conf, thr=0.3, prev_cos=None, alpha=0.35, ref_ratio=1.0):
    for i in (5, 6, 11, 12):
        if conf[i] <= thr:
            return None
    ms = (kpts[5] + kpts[6]) / 2.0
    mh = (kpts[11] + kpts[12]) / 2.0
    L = float(np.hypot(*(ms - mh)))
    if L < 20.0:
        return None
    cands = []
    for d, ref, c in ((kpts[6] - kpts[5], WS_REF, min(conf[5], conf[6])),
                      (kpts[12] - kpts[11], WH_REF, min(conf[11], conf[12]))):
        if d[0] < 0:
            d = -d
        cands.append((d / (ref * L), float(c)))
    (m0, c0), (m1, c1) = cands
    diff = float(np.hypot(*(m0 - m1)))
    scale = max(float(np.hypot(*m0)), float(np.hypot(*m1)), 1e-6)
    if diff <= 0.35 * scale + 0.08:
        m = (c0 * m0 + c1 * m1) / max(c0 + c1, 1e-6)
    else:
        m = m0 if c0 >= c1 else m1
    mag = float(np.hypot(*m))
    cos_raw = float(np.clip(mag / ref_ratio, 0.0, 1.0))
    cos_psi = cos_raw if prev_cos is None else alpha * cos_raw + (1.0 - alpha) * prev_cos
    e = (float(m[0]) / mag, float(m[1]) / mag) if mag > 0.15 else (1.0, 0.0)
    if e[0] < 0.2:
        e = (1.0, 0.0)
    return AnatomicalPlanes(e, cos_psi)


class PlaneEstimator(object):
    def __init__(self, thr=0.3, ref_ratio=1.0):
        self.thr, self.ref_ratio = thr, ref_ratio
        self.prev_cos = None
        self.planes = None

    def update(self, kpts, conf):
        p = project_to_anatomical_planes(kpts, conf, self.thr, self.prev_cos,
                                         ref_ratio=self.ref_ratio)
        if p is not None:
            self.prev_cos = p.cos_psi
            self.planes = p
        return self.planes


def angle_between(u, v):
    n = float(np.hypot(u[0], u[1]) * np.hypot(v[0], v[1]))
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip((u[0] * v[0] + u[1] * v[1]) / n, -1.0, 1.0))))


def tilt_plane(v):
    if float(np.hypot(v[0], v[1])) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(abs(v[0]), v[1])))


def abd_plane(v):
    if float(np.hypot(v[0], v[1])) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(abs(v[0]), -v[1])))


WINDOW = 24
FEAT_NAMES = [
    "foot_pitch_deg",
    "tibiotarsal_deg",
    "heel_toe_dy_norm",
    "foot_omega_dps",
    "knee_flexion",
    "trunk_lean",
]
CLASS_LABELS = ["TALONES", "PUNTAS", "REST", "TRANSITION", "STS", "ABD_I", "ABD_D"]


class FeatureExtractor(object):
    """Extractor de caracteristicas angulares y cinematicas para Elevacion de Talones y Puntas
    (OE3 y Etapa E2 del Plan de David Soto), normalizadas por la longitud del segmento
    rodilla-tobillo (tibia) e invariantes a escala y traslacion."""

    def __init__(self, thr=0.25):
        self.thr = min(float(thr), 0.25)
        self.omega_f = SavGolDeriv(5)
        self.L_tibia_ema = None
        self.side = None

    def reset(self):
        self.omega_f.reset()
        self.L_tibia_ema = None
        self.side = None

    def update(self, kpts, conf, t):
        if any(conf[i] <= self.thr for i in (5, 6, 11, 12)):
            return None
        ms, mh = (kpts[5] + kpts[6]) / 2.0, (kpts[11] + kpts[12]) / 2.0
        L_torso = float(np.hypot(*(ms - mh)))
        if L_torso < 20.0:
            return None

        self.side = select_side(kpts, conf, side="auto", prev_side=self.side)
        idx = EX_IDX[self.side]
        kn = kpts[idx["knee"]] if conf[idx["knee"]] > self.thr else None
        an = kpts[idx["ankle"]] if conf[idx["ankle"]] > self.thr else None
        hl = kpts[idx["heel"]] if conf[idx["heel"]] > self.thr else None
        toes = [kpts[i] for i in (idx["toe_big"], idx["toe_small"]) if conf[i] > self.thr]
        toe = np.mean(toes, axis=0) if toes else None

        L_tib = float(np.hypot(*(an - kn))) if (kn is not None and an is not None) else (0.75 * L_torso)
        if L_tib >= 10.0:
            self.L_tibia_ema = L_tib if self.L_tibia_ema is None else 0.9 * self.L_tibia_ema + 0.1 * L_tib
        denom_tib = max(float(self.L_tibia_ema or L_torso), 15.0)

        foot_pitch = 0.0
        tibiotarsal = 90.0
        dy_norm = 0.0
        if hl is not None and toe is not None:
            v_foot = toe - hl
            if float(np.hypot(v_foot[0], v_foot[1])) >= 6.0:
                # Positivo cuando el talon sube (TALONES, hl[1] < toe[1] -> v_foot[1] > 0)
                # Negativo cuando la punta sube (PUNTAS, toe[1] < hl[1] -> v_foot[1] < 0)
                foot_pitch = float(np.degrees(np.arctan2(float(v_foot[1]), abs(float(v_foot[0])) + 1e-6)))
                dy_norm = float(v_foot[1]) / denom_tib
                if kn is not None and an is not None:
                    ang_tt = angle_between(an - kn, v_foot)
                    if ang_tt is not None:
                        tibiotarsal = float(ang_tt)

        _, foot_omega = self.omega_f.push(t, foot_pitch)

        angs = []
        for h, k, a in ((11, 13, 15), (12, 14, 16)):
            if min(conf[h], conf[k], conf[a]) > self.thr:
                v = joint_angle(kpts[h], kpts[k], kpts[a])
                if v is not None:
                    angs.append(v)
        knee_flex = 180.0 - float(np.mean(angs)) if angs else 0.0
        trunk = tilt_from_vertical(mh, ms)
        return np.array([
            foot_pitch,
            tibiotarsal,
            dy_norm * 100.0,
            foot_omega,
            knee_flex,
            trunk if trunk is not None else 0.0,
        ], np.float32)


class TemporalClassifier(object):
    def __init__(self, model_path=None, window=WINDOW, predict_every=4, hold=3, min_conf=0.55):
        self.window, self.predict_every, self.min_conf = window, predict_every, min_conf
        self.buf = deque(maxlen=window)
        self.votes = deque(maxlen=max(1, hold))
        self.n = 0
        self.model = None
        self.label, self.prob, self.stable = None, 0.0, None
        self._load(model_path)

    def _load(self, path):
        if not path or not os.path.isfile(path):
            print("[CLF] Modelo '%s' no encontrado: la auto-deteccion no estara disponible." % path)
            return
        try:
            if joblib is not None:
                obj = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
        except Exception as e:
            print("[CLF] No se pudo cargar el modelo (%s: %s)." % (type(e).__name__, e))
            return
        if isinstance(obj, dict):
            obj = obj.get("model")
        nf = getattr(obj, "n_features_", None) or getattr(obj, "n_features_in_", None)
        if obj is None or (nf is not None and nf != self.window * len(FEAT_NAMES)):
            print("[CLF] Modelo incompatible (n_features=%s, esperado %d)."
                  % (str(nf), self.window * len(FEAT_NAMES)))
            return
        self.model = obj
        print("[CLF] Modelo cargado: %s (clases: %s)"
              % (path, ", ".join(str(c) for c in getattr(obj, "classes_", []))))

    def push(self, t, feat):
        if self.buf and (t - self.buf[-1][0]) > 1.0:
            self.buf.clear()
            self.votes.clear()
        self.buf.append((t, feat))
        self.n += 1
        if self.model is not None and len(self.buf) == self.window and self.n % self.predict_every == 0:
            self._predict()

    def window_vector(self):
        if len(self.buf) < self.window:
            return None
        return np.array([f for _, f in self.buf], np.float32).reshape(-1)

    def _predict(self):
        x = self.window_vector().reshape(1, -1)
        try:
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(x)[0]
                i = int(np.argmax(p))
                label, prob = str(self.model.classes_[i]), float(p[i])
            else:
                label, prob = str(self.model.predict(x)[0]), 1.0
        except Exception as e:
            print("[CLF] error de prediccion: %s" % e)
            return
        if prob < self.min_conf:
            label = "TRANSITION"
        self.label, self.prob = label, prob
        self.votes.append(label)
        if len(self.votes) == self.votes.maxlen and len(set(self.votes)) == 1:
            self.stable = label

    def stable_exercise(self):
        if self.stable == "STS":
            return "sts"
        if self.stable in ("ABD_I", "ABD_D"):
            return "abd"
        if self.stable in ("TALONES", "HEELS"):
            return "talones"
        if self.stable in ("PUNTAS", "TOES"):
            return "puntas"
        return None


class DataCollector(object):
    KEYS = {ord("1"): "STS", ord("2"): "ABD_I", ord("3"): "ABD_D",
            ord("4"): "REST", ord("5"): "TRANSITION", ord("0"): None,
            ord("6"): "TALONES", ord("7"): "PUNTAS"}
    META_COLS = ["subject", "view", "light", "band"]

    def __init__(self, path, window=WINDOW, stride=4, label=None, meta=None):
        self.stride, self.label, self.rows, self.n = stride, label, 0, 0
        meta = meta or {}
        self.meta = [str(meta.get(k, "")) for k in self.META_COLS]
        self.session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new = (not os.path.isfile(path)) or os.path.getsize(path) == 0
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["timestamp", "session", "label"] + self.META_COLS +
                            ["%s_t%02d" % (nm, i) for i in range(window) for nm in FEAT_NAMES])
        print("[REC] Recolectando en %s (sesion %s)" % (path, self.session))

    def set_label(self, label):
        self.label = label
        print("[REC] etiqueta = %s" % label)

    def push(self, t, vec):
        if vec is None or self.label is None:
            return
        self.n += 1
        if self.n % self.stride:
            return
        self.w.writerow([datetime.datetime.now().isoformat(), self.session, self.label] + self.meta +
                        ["%.4f" % v for v in vec])
        self.fh.flush()
        self.rows += 1

    def close(self):
        self.fh.close()


# --------------------------------------------------------------------------- #
# 3f. GESTOR DE PROTOCOLOS POR EDAD Y PLAN SEMANAL UNIFICADO (4 EJERCICIOS)
# --------------------------------------------------------------------------- #


class ProtocolManager(object):
    """Gestiona `protocolos.json` (editable) para los 4 ejercicios (STS, ABD, TALONES, PUNTAS):
    - Asigna plan semanal (ej. Lunes-Miercoles-Viernes o Lunes-Jueves), nivel, series,
      repeticiones por serie, descanso y angulo minimo de abduccion (abd_peak_deg)
      segun la edad del paciente.
    - Detecta el dia de la semana actual (o simulado con --day / tecla [D]).
    - Determina si hoy para el paciente es:
        * "DIA_DESCANSO" -> Hoy no le toca ejercicio.
        * "COMPLETO"     -> Ya completo todas las series y repeticiones de hoy.
        * "PENDIENTE"    -> Tiene series/repeticiones por realizar hoy.
    """

    WEEKDAYS_ES = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")
    WEEKDAYS_LABEL = {
        "lunes": "Lunes", "martes": "Martes", "miercoles": "Miercoles",
        "jueves": "Jueves", "viernes": "Viernes", "sabado": "Sabado", "domingo": "Domingo",
    }
    EXERCISE_ORDER = ("STS", "ABD", "TALONES", "PUNTAS")
    EXERCISE_LABELS = {
        "STS": "Sit-to-Stand (STS)",
        "ABD": "Abduccion de Cadera (ABD)",
        "TALONES": "Elevacion de Talones (TALONES)",
        "PUNTAS": "Elevacion de Puntas (PUNTAS)",
    }

    DEFAULT_CONFIG = {
        "configuracion_general": {
            "descripcion": (
                "Protocolos clinicos unificados basados en evidencia (ACSM/AHA, Fragala et al. 2019, "
                "Izquierdo et al. 2021, Soucie et al. 2011) para Sit-to-Stand (STS), Abduccion de "
                "Cadera (ABD), Elevacion de Talones (TALONES) y Elevacion de Puntas (PUNTAS)."
            ),
            "contar_solo_repeticiones_correctas": False,
            "auto_cerrar_al_completar_todo": False,
        },
        "protocolos_por_edad": {
            "adulto_18_59": {
                "edad_min": 18, "edad_max": 59,
                "etiqueta": "Adulto joven / mediana edad (18-59 anos)",
                "nivel": "Avanzado",
                "frecuencia_semanal": 3,
                "dias_asignados": ["lunes", "miercoles", "viernes"],
                "descanso_entre_series_s": 60,
                "abd_peak_deg": 45.0,
                "sts_trunk_max_deg": 30.0,
                "abd_trunk_max_deg": 15.0,
                "ejercicios": {
                    "STS": {"series": 3, "repeticiones": 10},
                    "ABD": {"series": 3, "repeticiones": 10},
                    "TALONES": {"series": 3, "repeticiones": 15},
                    "PUNTAS": {"series": 3, "repeticiones": 15},
                },
            },
            "adulto_mayor_60_69": {
                "edad_min": 60, "edad_max": 69,
                "etiqueta": "Adulto mayor (60-69 anos)",
                "nivel": "Intermedio",
                "frecuencia_semanal": 3,
                "dias_asignados": ["lunes", "miercoles", "viernes"],
                "descanso_entre_series_s": 60,
                "abd_peak_deg": 40.0,
                "sts_trunk_max_deg": 30.0,
                "abd_trunk_max_deg": 15.0,
                "ejercicios": {
                    "STS": {"series": 2, "repeticiones": 10},
                    "ABD": {"series": 2, "repeticiones": 10},
                    "TALONES": {"series": 3, "repeticiones": 12},
                    "PUNTAS": {"series": 3, "repeticiones": 12},
                },
            },
            "adulto_mayor_70_79": {
                "edad_min": 70, "edad_max": 79,
                "etiqueta": "Adulto mayor (70-79 anos)",
                "nivel": "Basico-Intermedio",
                "frecuencia_semanal": 3,
                "dias_asignados": ["lunes", "miercoles", "viernes"],
                "descanso_entre_series_s": 90,
                "abd_peak_deg": 40.0,
                "sts_trunk_max_deg": 30.0,
                "abd_trunk_max_deg": 15.0,
                "ejercicios": {
                    "STS": {"series": 2, "repeticiones": 8},
                    "ABD": {"series": 2, "repeticiones": 8},
                    "TALONES": {"series": 2, "repeticiones": 10},
                    "PUNTAS": {"series": 2, "repeticiones": 10},
                },
            },
            "adulto_mayor_80_plus": {
                "edad_min": 80, "edad_max": 120,
                "etiqueta": "Adulto mayor longevo / fragil (80+ anos)",
                "nivel": "Basico",
                "frecuencia_semanal": 2,
                "dias_asignados": ["lunes", "jueves"],
                "descanso_entre_series_s": 90,
                "abd_peak_deg": 35.0,
                "sts_trunk_max_deg": 30.0,
                "abd_trunk_max_deg": 15.0,
                "ejercicios": {
                    "STS": {"series": 2, "repeticiones": 6},
                    "ABD": {"series": 2, "repeticiones": 6},
                    "TALONES": {"series": 2, "repeticiones": 8},
                    "PUNTAS": {"series": 2, "repeticiones": 8},
                },
            },
        },
        "pacientes": {
            "Paciente_01": {
                "edad": 65,
                "grupo_protocolo": "adulto_mayor_60_69",
                "nivel": "Intermedio",
                "dias_asignados": ["lunes", "miercoles", "viernes"],
                "descanso_entre_series_s": 60,
                "abd_peak_deg": 40.0,
                "ejercicios": {
                    "STS": {"series": 2, "repeticiones": 10},
                    "ABD": {"series": 2, "repeticiones": 10},
                    "TALONES": {"series": 3, "repeticiones": 12},
                    "PUNTAS": {"series": 3, "repeticiones": 12},
                },
                "progreso_diario": {},
            },
        },
    }

    def __init__(self, path="protocolos.json", override_day=None):
        self.path = path
        self.override_day = self.normalize_day(override_day) if override_day else None
        self.forced_sessions = set()
        self._status_cache = {}
        self.data = {}
        self.load()

    @classmethod
    def normalize_day(cls, d):
        if not d:
            return None
        s = str(d).strip().lower()
        for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
            s = s.replace(a, b)
        return s if s in cls.WEEKDAYS_ES else None

    @staticmethod
    def clean_name(name):
        s = str(name or "").strip().replace(" ", "_")
        return s if s else "Paciente_01"

    def load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except Exception as e:
                print("[PROTO] Error leyendo %s (%s) -> usando valores por defecto." % (self.path, e))
                self.data = json.loads(json.dumps(self.DEFAULT_CONFIG))
        else:
            self.data = json.loads(json.dumps(self.DEFAULT_CONFIG))
            self.save()
            print("[PROTO] Archivo %s creado con protocolos clinicos por defecto." % self.path)
        if "configuracion_general" not in self.data:
            self.data["configuracion_general"] = dict(self.DEFAULT_CONFIG["configuracion_general"])
        if "protocolos_por_edad" not in self.data:
            self.data["protocolos_por_edad"] = json.loads(
                json.dumps(self.DEFAULT_CONFIG["protocolos_por_edad"]))
        if "pacientes" not in self.data:
            self.data["pacientes"] = {}

        # Asegurar que todos los grupos de edad tengan los 4 ejercicios y nivel
        for gname, def_g in self.DEFAULT_CONFIG["protocolos_por_edad"].items():
            if gname in self.data["protocolos_por_edad"]:
                g_cur = self.data["protocolos_por_edad"][gname]
                g_cur.setdefault("nivel", def_g.get("nivel", "Intermedio"))
                ejs = g_cur.setdefault("ejercicios", {})
                for ek, ev in def_g["ejercicios"].items():
                    ejs.setdefault(ek, dict(ev))

    def save(self):
        self._status_cache.clear()
        try:
            d = os.path.dirname(self.path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, ensure_ascii=False)
            if os.path.isfile(self.path):
                os.replace(tmp, self.path)
            else:
                os.rename(tmp, self.path)
        except Exception as e:
            print("[PROTO] No se pudo guardar %s: %s" % (self.path, e))

    def resolve_age_group(self, age):
        try:
            age_val = int(age)
        except Exception:
            age_val = 65
        grupos = self.data.get("protocolos_por_edad") or self.DEFAULT_CONFIG["protocolos_por_edad"]
        ordered = sorted(grupos.items(), key=lambda kv: int(kv[1].get("edad_min", 0)))
        for gname, gcfg in ordered:
            lo = int(gcfg.get("edad_min", 0))
            hi = int(gcfg.get("edad_max", 200))
            if lo <= age_val <= hi:
                return gname, gcfg
        if ordered:
            if age_val < int(ordered[0][1].get("edad_min", 18)):
                return ordered[0]
            return ordered[-1]
        return "adulto_mayor_60_69", self.DEFAULT_CONFIG["protocolos_por_edad"]["adulto_mayor_60_69"]

    def ensure_patient(self, name, age=None):
        pname = self.clean_name(name)
        pacientes = self.data.setdefault("pacientes", {})
        existing = pacientes.get(pname)
        if existing is None:
            age_int = int(age) if age is not None else 65
            gname, gcfg = self.resolve_age_group(age_int)
            pacientes[pname] = {
                "edad": age_int,
                "grupo_protocolo": gname,
                "nivel": gcfg.get("nivel", "Intermedio"),
                "dias_asignados": list(gcfg.get("dias_asignados", ["lunes", "miercoles", "viernes"])),
                "descanso_entre_series_s": int(gcfg.get("descanso_entre_series_s", 60)),
                "abd_peak_deg": float(gcfg.get("abd_peak_deg", 40.0)),
                "ejercicios": json.loads(json.dumps(gcfg.get("ejercicios", {
                    "STS": {"series": 2, "repeticiones": 10},
                    "ABD": {"series": 2, "repeticiones": 10},
                    "TALONES": {"series": 3, "repeticiones": 12},
                    "PUNTAS": {"series": 3, "repeticiones": 12},
                }))),
                "progreso_diario": {},
            }
            self.save()
            print("[PROTO] Nuevo paciente '%s' (%d anos -> %s, nivel %s): dias=%s, ABD>=%.0f deg"
                  % (pname, age_int, gname, pacientes[pname]["nivel"],
                     ",".join(pacientes[pname]["dias_asignados"]),
                     pacientes[pname]["abd_peak_deg"]))
        else:
            changed = False
            if age is not None and int(age) != int(existing.get("edad", 65)):
                age_int = int(age)
                gname, gcfg = self.resolve_age_group(age_int)
                existing["edad"] = age_int
                existing["grupo_protocolo"] = gname
                existing["nivel"] = gcfg.get("nivel", "Intermedio")
                existing["dias_asignados"] = list(
                    gcfg.get("dias_asignados", ["lunes", "miercoles", "viernes"]))
                existing["descanso_entre_series_s"] = int(gcfg.get("descanso_entre_series_s", 60))
                existing["abd_peak_deg"] = float(gcfg.get("abd_peak_deg", 40.0))
                existing["ejercicios"] = json.loads(json.dumps(gcfg.get("ejercicios", {
                    "STS": {"series": 2, "repeticiones": 10},
                    "ABD": {"series": 2, "repeticiones": 10},
                    "TALONES": {"series": 3, "repeticiones": 12},
                    "PUNTAS": {"series": 3, "repeticiones": 12},
                })))
                changed = True
                print("[PROTO] Edad actualizada para '%s' (%d anos -> %s)" % (pname, age_int, gname))
            else:
                age_int = int(existing.get("edad", 65))
                gname, gcfg = self.resolve_age_group(age_int)
                if "grupo_protocolo" not in existing:
                    existing["grupo_protocolo"] = gname
                    changed = True
                if "nivel" not in existing:
                    existing["nivel"] = gcfg.get("nivel", "Intermedio")
                    changed = True
                if "dias_asignados" not in existing:
                    existing["dias_asignados"] = list(
                        gcfg.get("dias_asignados", ["lunes", "miercoles", "viernes"]))
                    changed = True
                if "descanso_entre_series_s" not in existing:
                    existing["descanso_entre_series_s"] = int(gcfg.get("descanso_entre_series_s", 60))
                    changed = True
                if "abd_peak_deg" not in existing:
                    existing["abd_peak_deg"] = float(gcfg.get("abd_peak_deg", 40.0))
                    changed = True
                ejs = existing.setdefault("ejercicios", {})
                for ek, ev in gcfg.get("ejercicios", {}).items():
                    if ek not in ejs:
                        ejs[ek] = dict(ev)
                        changed = True
                if "progreso_diario" not in existing:
                    existing["progreso_diario"] = {}
                    changed = True
            if changed:
                self.save()
        return pname, pacientes[pname]

    def patient_list(self):
        pts = list(self.data.get("pacientes", {}).keys())
        return pts if pts else ["Paciente_01"]

    def step_patient(self, cur_patient, delta=1):
        pts = self.patient_list()
        if cur_patient not in pts:
            pts.append(cur_patient)
        idx = (pts.index(cur_patient) + delta) % len(pts)
        return pts[idx]

    def cycle_override_day(self):
        self._status_cache.clear()
        if self.override_day is None:
            self.override_day = self.WEEKDAYS_ES[0]
        else:
            idx = self.WEEKDAYS_ES.index(self.override_day)
            self.override_day = self.WEEKDAYS_ES[idx + 1] if idx + 1 < len(self.WEEKDAYS_ES) else None
        return self.override_day

    def get_day_and_date(self):
        now = datetime.datetime.now()
        real_day = self.WEEKDAYS_ES[now.weekday()]
        iso_date = now.strftime("%Y-%m-%d")
        if self.override_day and self.override_day != real_day:
            return self.override_day, "%s_%s" % (iso_date, self.override_day), True
        return real_day, iso_date, (self.override_day is not None)

    def _ensure_day_progress(self, pinfo, date_key, day_name):
        prog_all = pinfo.setdefault("progreso_diario", {})
        ej_plan = pinfo.get("ejercicios") or {}
        if date_key not in prog_all:
            prog_all[date_key] = {
                "dia_semana": day_name,
                "ejercicios": {},
                "completado": False,
                "ultima_actualizacion": datetime.datetime.now().isoformat(),
            }
        dprog = prog_all[date_key]
        dprog["dia_semana"] = day_name
        ej_prog = dprog.setdefault("ejercicios", {})
        for ej_key, cfg in ej_plan.items():
            s_meta = int(cfg.get("series", 2))
            r_meta = int(cfg.get("repeticiones", 10))
            if ej_key not in ej_prog:
                ej_prog[ej_key] = {
                    "series_meta": s_meta,
                    "repeticiones_meta": r_meta,
                    "series_completadas": 0,
                    "reps_en_serie_actual": 0,
                    "reps_correctas_hoy": 0,
                    "reps_incorrectas_hoy": 0,
                    "completado": False,
                }
            else:
                ej_prog[ej_key]["series_meta"] = s_meta
                ej_prog[ej_key]["repeticiones_meta"] = r_meta
                ej_prog[ej_key]["completado"] = (
                    int(ej_prog[ej_key].get("series_completadas", 0)) >= s_meta
                )
        dprog["completado"] = all(
            bool(ej_prog[k].get("completado")) for k in ej_plan.keys()
        ) if ej_plan else False
        return dprog

    def force_today(self, patient_name):
        pname, _ = self.ensure_patient(patient_name)
        _, date_key, _ = self.get_day_and_date()
        self.forced_sessions.add((pname, date_key))
        self._status_cache.clear()

    def reset_today(self, patient_name):
        pname, pinfo = self.ensure_patient(patient_name)
        day_name, date_key, _ = self.get_day_and_date()
        pinfo.setdefault("progreso_diario", {}).pop(date_key, None)
        self._ensure_day_progress(pinfo, date_key, day_name)
        self.save()

    def get_daily_status(self, patient_name):
        ckey = (patient_name, self.override_day)
        cached = self._status_cache.get(ckey)
        if cached is not None:
            return cached
        pname, pinfo = self.ensure_patient(patient_name)
        day_name, date_key, is_sim = self.get_day_and_date()
        dias_norm = [self.normalize_day(d) for d in pinfo.get("dias_asignados", [])]
        dias_norm = [d for d in dias_norm if d]
        es_dia_asig = (day_name in dias_norm)
        forzado = ((pname, date_key) in self.forced_sessions)
        dprog = self._ensure_day_progress(pinfo, date_key, day_name)

        if not es_dia_asig and not forzado:
            estado_dia = "DIA_DESCANSO"
        elif dprog.get("completado", False):
            estado_dia = "COMPLETO"
        else:
            estado_dia = "PENDIENTE"

        dias_str = ", ".join(self.WEEKDAYS_LABEL.get(d, d.capitalize()) for d in dias_norm)
        grupos = self.data.get("protocolos_por_edad") or {}
        gname = pinfo.get("grupo_protocolo", "")
        gcfg = grupos.get(gname) or {}
        glabel = gcfg.get("etiqueta", gname)
        nivel = pinfo.get("nivel") or gcfg.get("nivel", "Intermedio")

        res = {
            "paciente": pname,
            "edad": int(pinfo.get("edad", 65)),
            "nivel": nivel,
            "grupo_protocolo": gname,
            "grupo_etiqueta": glabel,
            "dia_semana": day_name,
            "dia_label": self.WEEKDAYS_LABEL.get(day_name, day_name.capitalize()),
            "fecha_key": date_key,
            "dia_simulado": is_sim,
            "dias_asignados": dias_norm,
            "dias_asignados_str": dias_str or "Ninguno",
            "es_dia_entrenamiento": es_dia_asig or forzado,
            "forzado_hoy": forzado,
            "estado_dia": estado_dia,
            "abd_peak_deg": float(pinfo.get("abd_peak_deg", 40.0)),
            "descanso_s": int(pinfo.get("descanso_entre_series_s", 60)),
            "ejercicios": dprog.get("ejercicios", {}),
        }
        self._status_cache[ckey] = res
        return res

    def next_pending_exercise(self, patient_name):
        st = self.get_daily_status(patient_name)
        if st["estado_dia"] != "PENDIENTE":
            return None
        for ej_key in self.EXERCISE_ORDER:
            ej_info = st["ejercicios"].get(ej_key)
            if ej_info and not ej_info.get("completado", False):
                return ej_key.lower()
        for ej_key, ej_info in st["ejercicios"].items():
            if not ej_info.get("completado", False):
                return ej_key.lower()
        return None

    def current_set_rep_labels(self, patient_name, exercise_title):
        st = self.get_daily_status(patient_name)
        ej = (exercise_title or "").upper()
        info = st["ejercicios"].get(ej)
        if not info:
            return None, None
        s_done = int(info.get("series_completadas", 0))
        s_meta = int(info.get("series_meta", 2))
        r_cur = int(info.get("reps_en_serie_actual", 0))
        r_meta = int(info.get("repeticiones_meta", 10))
        s_disp = min(s_done + 1, s_meta)
        return "%d/%d" % (s_disp, s_meta), "%d/%d" % (r_cur, r_meta)

    def on_rep(self, patient_name, exercise_title, ok, reasons, info):
        pname, pinfo = self.ensure_patient(patient_name)
        day_name, date_key, _ = self.get_day_and_date()
        dprog = self._ensure_day_progress(pinfo, date_key, day_name)
        ej_key = (exercise_title or "").upper()
        ej_prog = dprog["ejercicios"].get(ej_key)
        if ej_prog is None:
            return None

        if ok:
            ej_prog["reps_correctas_hoy"] = int(ej_prog.get("reps_correctas_hoy", 0)) + 1
        else:
            ej_prog["reps_incorrectas_hoy"] = int(ej_prog.get("reps_incorrectas_hoy", 0)) + 1

        only_ok = bool(
            (self.data.get("configuracion_general") or {}).get("contar_solo_repeticiones_correctas", False)
        )
        counted = (ok or not only_ok)
        serie_completada = False
        ejercicio_completado = False
        plan_completado = False

        if counted and not ej_prog.get("completado", False):
            ej_prog["reps_en_serie_actual"] = int(ej_prog.get("reps_en_serie_actual", 0)) + 1
            if ej_prog["reps_en_serie_actual"] >= int(ej_prog["repeticiones_meta"]):
                ej_prog["series_completadas"] = int(ej_prog.get("series_completadas", 0)) + 1
                ej_prog["reps_en_serie_actual"] = 0
                serie_completada = True
                if ej_prog["series_completadas"] >= int(ej_prog["series_meta"]):
                    ej_prog["completado"] = True
                    ejercicio_completado = True

        all_done = all(
            bool(v.get("completado", False)) for v in dprog["ejercicios"].values()
        ) if dprog["ejercicios"] else False
        if all_done and not dprog.get("completado", False):
            dprog["completado"] = True
            dprog["completado_en"] = datetime.datetime.now().isoformat()
            plan_completado = True
        elif all_done:
            dprog["completado"] = True

        dprog["ultima_actualizacion"] = datetime.datetime.now().isoformat()
        self.save()

        return {
            "paciente": pname,
            "ejercicio": ej_key,
            "contada": counted,
            "serie_completada": serie_completada,
            "ejercicio_completado": ejercicio_completado,
            "plan_completado": plan_completado,
            "series_completadas": int(ej_prog["series_completadas"]),
            "series_meta": int(ej_prog["series_meta"]),
            "reps_en_serie_actual": int(ej_prog["reps_en_serie_actual"]),
            "reps_meta": int(ej_prog["repeticiones_meta"]),
            "descanso_s": int(pinfo.get("descanso_entre_series_s", 60)),
        }


# --------------------------------------------------------------------------- #
# 3g. REGISTRO CSV Y BASE DE DATOS NoSQL (FrameDB)
# --------------------------------------------------------------------------- #


class SessionLogger(object):
    """Escucha el cierre de cada repeticion (OK o FALLIDA) de los 4 ejercicios y la agrega a un CSV."""

    HEADER = ["Timestamp", "Ejercicio", "Resultado", "Motivos_Falla", "Metricas_Maximas",
              "Sujeto", "Serie_Val", "Vista", "Iluminacion", "Banda"]
    NAMES = {
        "STS": "Sit-to-Stand",
        "ABD": "Abduccion_Cadera",
        "TALONES": "Elevacion_Talones",
        "PUNTAS": "Elevacion_Puntas",
    }
    ALIAS = {
        "knee_peak": "pico_rodilla",
        "peak": "angulo_max",
        "omega_down_peak": "vel_caida_dps",
    }

    def __init__(self, path, meta=None):
        meta = meta or {}
        self.meta = dict((k, str(meta.get(k, ""))) for k in ("subject", "view", "light", "band"))
        self.series_id = 1          # [K] en el pipeline lo incrementa: una "serie de validacion" por bloque
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        new = (not os.path.isfile(path)) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(self.HEADER)
            self.fh.flush()
        self.path = path
        print("[LOG] Registro de repeticiones en %s" % path)

    def on_rep(self, title, ok, reasons, info):
        metrics = dict((self.ALIAS.get(k, k), round(float(v), 1)) for k, v in info.items())
        self.w.writerow([datetime.datetime.now().isoformat(),
                         self.NAMES.get(title, title),
                         "OK" if ok else "FALLA",
                         "; ".join(reasons),
                         json.dumps(metrics, sort_keys=True),
                         self.meta["subject"], self.series_id, self.meta["view"],
                         self.meta["light"], self.meta["band"]])
        self.fh.flush()

    def next_series(self):
        self.series_id += 1
        return self.series_id

    def close(self):
        self.fh.close()


class PerfLogger(object):
    """Caracterizacion en dispositivo (OE1): cada ~1 s registra FPS de camara/bucle, latencia
    de inferencia, CPU, RAM, temperatura, GPU, fraccion de cuadros con cadera-rodilla-tobillo
    visibles (conf > umbral) y la cinematica media (para el experimento de geometria de camara,
    con el sujeto de pie en postura neutra). Solo usa /proc y /sys (sin dependencias)."""

    HEADER = ["t_s", "view", "light", "cam_fps", "loop_fps", "infer_ms", "cpu_pct", "ram_mb",
              "temp_c", "gpu_pct", "tracking", "vis_frac", "n_inf",
              "knee_flex_mean", "knee_flex_std", "hip_abd_mean", "hip_abd_std", "trunk_mean"]

    def __init__(self, path, meta=None, period=1.0):
        meta = meta or {}
        self.view, self.light = str(meta.get("view", "")), str(meta.get("light", ""))
        self.period = period
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        new = (not os.path.isfile(path)) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(self.HEADER)
        self.t0 = self.tlast = time.perf_counter()
        self.cpu_prev = self._cpu_times()
        self.vis, self.knee, self.hip, self.trunk, self.trk = [], [], [], [], []
        print("[PERF] Registro de rendimiento en %s" % path)

    @staticmethod
    def _cpu_times():
        try:
            with open("/proc/stat") as f:
                v = [float(x) for x in f.readline().split()[1:]]
            return sum(v), v[3] + (v[4] if len(v) > 4 else 0.0)
        except Exception:
            return None

    @staticmethod
    def _read(path, scale=1.0):
        try:
            with open(path) as f:
                return float(f.read().strip()) * scale
        except Exception:
            return float("nan")

    @staticmethod
    def _ram_mb():
        try:
            m = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")[0], float(line.split(":")[1].split()[0])
                    m[k] = v
            return (m["MemTotal"] - m.get("MemAvailable", m.get("MemFree", 0.0))) / 1024.0
        except Exception:
            return float("nan")

    def sample(self, kpts, conf, tracking, thr, feat):
        self.trk.append(1.0 if tracking else 0.0)
        if kpts is not None and conf is not None:
            # OE1 de David: visibilidad del segmento pierna-pie (rodilla, tobillo, talon y antepie)
            ok_i = (conf[13] > thr and conf[15] > thr and conf[19] > thr and max(conf[17], conf[18]) > thr)
            ok_d = (conf[14] > thr and conf[16] > thr and conf[22] > thr and max(conf[20], conf[21]) > thr)
            self.vis.append(1.0 if (ok_i or ok_d) else 0.0)
        if feat is not None:
            self.knee.append(float(feat[4]))
            self.hip.append(float(feat[0]))
            self.trunk.append(float(feat[5]))

    def tick(self, cam_fps, loop_fps, infer_ms):
        now = time.perf_counter()
        if now - self.tlast < self.period:
            return
        cur = self._cpu_times()
        cpu = float("nan")
        if cur is not None and self.cpu_prev is not None and cur[0] > self.cpu_prev[0]:
            cpu = 100.0 * (1.0 - (cur[1] - self.cpu_prev[1]) / (cur[0] - self.cpu_prev[0]))
        self.cpu_prev = cur
        temp = self._read("/sys/class/thermal/thermal_zone0/temp", 1e-3)
        gpu = self._read("/sys/devices/gpu.0/load", 0.1)

        def mean(x):
            return float(np.mean(x)) if x else float("nan")

        def std(x):
            return float(np.std(x)) if len(x) > 1 else float("nan")

        self.w.writerow([round(now - self.t0, 2), self.view, self.light, round(cam_fps, 2),
                         round(loop_fps, 2), round(infer_ms, 2), round(cpu, 1),
                         round(self._ram_mb(), 0), round(temp, 1), round(gpu, 1),
                         round(mean(self.trk), 3), round(mean(self.vis), 3), len(self.vis),
                         round(mean(self.knee), 2), round(std(self.knee), 2),
                         round(mean(self.hip), 2), round(std(self.hip), 2),
                         round(mean(self.trunk), 2)])
        self.fh.flush()
        self.tlast = now
        self.vis, self.knee, self.hip, self.trunk, self.trk = [], [], [], [], []

    def close(self):
        self.fh.close()


class FrameDB(object):
    """Base de datos NoSQL no bloqueante (Redis Streams + archivo JSONL `bd_pacientes.json1`)
    compatible con los 4 ejercicios (STS, ABD, TALONES, PUNTAS)."""

    _STOP = object()
    ACTIVE_STATES = ("RISING", "STANDING", "LOWERING", "UP", "PEAK", "DOWN")
    RELEVANT_KPTS = BODY_LANDMARKS
    FOOT_KPTS = TRACKED_LANDMARKS

    def __init__(self, session_id, backend="redis", url="redis://127.0.0.1:6379/0",
                 maxlen=0, out_dir="logs", patient="Paciente_01", client=None, only_active=True,
                 unified_file="bd_pacientes.json1", age=65, day_name="lunes", abd_peak=40.0):
        self.sid, self.backend, self.maxlen, self.out_dir = session_id, backend, maxlen, out_dir
        self.patient = patient or "Paciente_01"
        self.age = int(age)
        self.day_name = day_name
        self.abd_peak = float(abd_peak)
        self.only_active = only_active
        self.unified_file = unified_file
        self.t0_mono = time.perf_counter()
        self.q = queue.Queue(maxsize=4000)
        self.n = self.dropped = 0
        self.r = self.fh = None
        self.jsonl_path = None
        self._last_ms, self._seq, self._prev_mono = 0, 0, None
        self._prev_state = (None, None)
        self._rep_buf = []
        self.k_frames = "sts:%s:frames" % session_id
        self.k_reps = "sts:%s:reps" % session_id
        self.k_meta = "sts:%s:meta" % session_id
        if backend == "redis":
            self._connect(url, self.patient, client)
        self._open_jsonl()
        self.th = threading.Thread(target=self._run, name="framedb")
        self.th.daemon = True
        self.th.start()

    def set_patient(self, new_patient, age=None, day_name=None, abd_peak=None):
        self.patient = new_patient
        if age is not None:
            self.age = int(age)
        if day_name is not None:
            self.day_name = str(day_name)
        if abd_peak is not None:
            self.abd_peak = float(abd_peak)
        self.sid = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), new_patient)
        self.k_frames = "sts:%s:frames" % self.sid
        self.k_reps = "sts:%s:reps" % self.sid
        self.k_meta = "sts:%s:meta" % self.sid
        self._prev_state = (None, None)
        self._rep_buf = []
        if self.r is not None:
            try:
                self.r.sadd("sts:sessions", self.sid)
                self.r.hset(self.k_meta, mapping={
                    "session_id": self.sid, "patient": self.patient,
                    "age": self.age, "day_name": self.day_name or "",
                    "abd_peak_deg": self.abd_peak,
                    "started_ms": int(time.time() * 1000), "fps_nominal": 8
                })
            except Exception:
                pass
        print("[DB] Paciente activo -> %s (%d anos, %s, ABD>=%.0f deg, sesion %s)"
              % (self.patient, self.age, self.day_name, self.abd_peak, self.sid))

    def _connect(self, url, patient, client):
        if client is None and redis_lib is None:
            print("[DB] Modulo 'redis' no instalado -> JSONL local.")
            return
        try:
            r = client if client is not None else redis_lib.Redis.from_url(
                url, socket_timeout=1.0, socket_connect_timeout=1.0)
            r.ping()
            r.sadd("sts:sessions", self.sid)
            r.hset(self.k_meta, mapping={"session_id": self.sid, "patient": patient,
                                         "started_ms": int(time.time() * 1000), "fps_nominal": 8})
            self.r = r
            print("[DB] Redis Streams conectado (%s). Sesion '%s' (%s)." % (url, self.sid, patient))
        except Exception as e:
            print("[DB] Redis no disponible (%s: %s) -> JSONL local." % (type(e).__name__, e))

    def _open_jsonl(self):
        if self.fh is not None:
            return
        if self.unified_file:
            self.jsonl_path = self.unified_file
        else:
            if self.out_dir and not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)
            self.jsonl_path = os.path.join(self.out_dir, "frames_%s.jsonl" % self.sid)
        self.fh = open(self.jsonl_path, "a", encoding="utf-8")
        print("[DB] Base de datos unificada multi-paciente activa en: %s" % self.jsonl_path)

    @classmethod
    def format_doc(cls, d):
        if "cinematica" not in d or "landmarks_2d" not in d:
            return json.dumps(d, indent=2, ensure_ascii=False)
        cin_lines = ['    "%s": %s' % (k, json.dumps(v)) for k, v in d["cinematica"].items()]
        lm_lines = [
            '    {"punto": "%s", "x": %s, "y": %s, "conf": %s}'
            % (p["punto"], json.dumps(p["x"]), json.dumps(p["y"]), json.dumps(p["conf"]))
            for p in d["landmarks_2d"]
        ]
        ev_block = ""
        if d.get("evaluacion"):
            ev = d["evaluacion"]
            extra_ev = ""
            if ev.get("serie") is not None:
                extra_ev += '    "serie": %s,\n' % json.dumps(ev.get("serie"), ensure_ascii=False)
            if ev.get("rep_en_serie") is not None:
                extra_ev += '    "rep_en_serie": %s,\n' % json.dumps(
                    ev.get("rep_en_serie"), ensure_ascii=False)
            ev_block = (
                '  "evaluacion": {\n'
                '%s'
                '    "repeticion": %s,\n'
                '    "resultado": %s,\n'
                '    "motivo": %s\n'
                '  },\n' % (
                    extra_ev,
                    json.dumps(ev.get("repeticion")),
                    json.dumps(ev.get("resultado")),
                    json.dumps(ev.get("motivo"), ensure_ascii=False),
                )
            )
        hdr_block = ""
        if d.get("paciente"):
            hdr_block += '  "paciente": %s,\n' % json.dumps(d.get("paciente"), ensure_ascii=False)
        if d.get("edad") is not None:
            hdr_block += '  "edad": %s,\n' % json.dumps(d.get("edad"))
        if d.get("dia_semana"):
            hdr_block += '  "dia_semana": %s,\n' % json.dumps(d.get("dia_semana"), ensure_ascii=False)
        if d.get("sesion"):
            hdr_block += '  "sesion": %s,\n' % json.dumps(d.get("sesion"), ensure_ascii=False)
        if d.get("abd_rango_min_deg") is not None:
            hdr_block += '  "abd_rango_min_deg": %s,\n' % json.dumps(d.get("abd_rango_min_deg"))
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
                json.dumps(d.get("timestamp_ms")),
                json.dumps(d.get("delta_t_ms")),
                json.dumps(d.get("frame_id")),
                json.dumps(d.get("ejercicio")),
                json.dumps(d.get("estado")),
                ev_block,
                ",\n".join(cin_lines),
                ",\n".join(lm_lines),
            )
        )

    def _enqueue(self, kind, doc):
        if kind != "frame":
            doc["timestamp_ms"] = int(time.time() * 1000)
        try:
            self.q.put_nowait((kind, doc))
        except queue.Full:
            self.dropped += 1

    def log_frame(self, snap, frame_id, infer_ms, tracking, kpts=None, conf=None):
        ej_raw = snap.get("ejercicio")
        ej = (ej_raw or "").upper()
        st = snap.get("estado")
        if not tracking or not ej:
            self._prev_state = (None, None)
            self._rep_buf = []
            return

        # Si el ejercicio es TALONES o PUNTAS y viene con transiciones directas (_pending_events)
        heel_events = snap.get("heel_events") or []
        if ej in ("TALONES", "PUNTAS") and heel_events:
            ev_meta = snap.get("evaluacion")
            for hev in heel_events:
                st_h = hev["estado"]
                now = time.perf_counter()
                reset_dt = (st_h == "UP") or (self._prev_mono is None)
                dt_ms = 0.0 if reset_dt else round((now - self._prev_mono) * 1000.0, 1)
                self._prev_mono = now
                doc = {
                    "paciente": self.patient,
                    "edad": self.age,
                    "dia_semana": self.day_name,
                    "sesion": self.sid,
                    "timestamp_ms": int(time.time() * 1000),
                    "delta_t_ms": dt_ms,
                    "frame_id": int(frame_id),
                    "ejercicio": ej,
                    "estado": st_h,
                    "cinematica": hev["cinematica"],
                    "landmarks_2d": hev["landmarks"],
                }
                if st_h in ("UP", "PEAK", "DOWN"):
                    if st_h == "UP":
                        self._rep_buf = []
                    self._rep_buf.append(doc)
                elif st_h == "REST":
                    ev_use = ev_meta or hev.get("evaluacion")
                    if ev_use is not None:
                        doc["evaluacion"] = ev_use
                        for bdoc in self._rep_buf:
                            self._enqueue("frame", bdoc)
                        self._enqueue("frame", doc)
                    self._rep_buf = []
            return

        if ej in ("TALONES", "PUNTAS") and self.only_active:
            return

        prev_ej, prev_st = self._prev_state
        if self.only_active:
            if (ej, st) == self._prev_state:
                return
            is_active = st in self.ACTIVE_STATES
            is_closing = (st in ("SIT", "REST")) and (prev_ej == ej) and (prev_st in self.ACTIVE_STATES)
            self._prev_state = (ej, st)
            if not (is_active or is_closing):
                self._rep_buf = []
                return
            reset_dt = is_active and (prev_ej != ej or prev_st not in self.ACTIVE_STATES)
        else:
            is_active, is_closing = True, False
            reset_dt = (prev_ej != ej)
            self._prev_state = (ej, st)

        ang = snap.get("angulos") or {}
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

        kpt_list = self.FOOT_KPTS if ej in ("TALONES", "PUNTAS") else self.RELEVANT_KPTS
        lms = []
        if tracking and kpts is not None and conf is not None:
            for idx, name in kpt_list:
                if idx < len(kpts):
                    lms.append({
                        "punto": name,
                        "x": round(float(kpts[idx][0]), 1),
                        "y": round(float(kpts[idx][1]), 1),
                        "conf": round(float(conf[idx]), 2),
                    })

        now = time.perf_counter()
        if reset_dt or self._prev_mono is None:
            dt_ms = 0.0
        else:
            dt_ms = round((now - self._prev_mono) * 1000.0, 1)
        self._prev_mono = now

        doc = {
            "paciente": self.patient,
            "edad": self.age,
            "dia_semana": self.day_name,
            "sesion": self.sid,
            "abd_rango_min_deg": self.abd_peak,
            "timestamp_ms": int(time.time() * 1000),
            "delta_t_ms": dt_ms,
            "frame_id": int(frame_id),
            "ejercicio": ej or None,
            "estado": st,
            "cinematica": cin,
            "landmarks_2d": lms,
        }
        if not self.only_active:
            if snap.get("evaluacion"):
                doc["evaluacion"] = snap["evaluacion"]
            self._enqueue("frame", doc)
            return

        if is_active:
            if reset_dt:
                self._rep_buf = []
            self._rep_buf.append(doc)
        elif is_closing:
            ev = snap.get("evaluacion")
            if ev is not None:
                doc["evaluacion"] = ev
                for bdoc in self._rep_buf:
                    self._enqueue("frame", bdoc)
                self._enqueue("frame", doc)
            self._rep_buf = []

    def on_rep(self, title, ok, reasons, info):
        self._enqueue("rep", {"paciente": self.patient, "edad": self.age,
                              "dia_semana": self.day_name, "sesion": self.sid,
                              "ejercicio": title, "resultado": "OK" if ok else "FALLA",
                              "motivos": list(reasons),
                              "metricas": dict((k, round(float(v), 1)) for k, v in info.items())})

    def _run(self):
        batch, last = [], time.time()
        while True:
            try:
                item = self.q.get(timeout=0.25)
            except queue.Empty:
                item = None
            stop = item is self._STOP
            if item is not None and not stop:
                batch.append(item)
            if batch and (stop or item is None or len(batch) >= 20 or time.time() - last >= 0.5):
                self._flush(batch)
                batch, last = [], time.time()
            if stop:
                break

    def _flush(self, batch):
        if self.r is not None:
            try:
                pipe = self.r.pipeline(transaction=False)
                for kind, doc in batch:
                    ts = int(doc.get("timestamp_ms") or int(time.time() * 1000))
                    ms = max(ts, self._last_ms)
                    self._seq = self._seq + 1 if ms == self._last_ms else 0
                    self._last_ms = ms
                    fields = {"doc": json.dumps(doc, separators=(",", ":")), "ts_ms": ms,
                              "ej": doc.get("ejercicio") or "", "estado": doc.get("estado") or ""}
                    pipe.xadd(self.k_reps if kind == "rep" else self.k_frames, fields,
                              id="%d-%d" % (ms, self._seq),
                              maxlen=self.maxlen or None, approximate=bool(self.maxlen))
                pipe.execute()
            except Exception as e:
                print("[DB] Redis fallo (%s: %s) -> usando solo archivo unificado." % (type(e).__name__, e))
                self.r = None
        if self.fh is not None:
            for kind, doc in batch:
                if kind == "frame":
                    self.fh.write(self.format_doc(doc) + "\n")
            self.fh.flush()
        self.n += len(batch)

    def close(self):
        self.q.put(self._STOP)
        self.th.join(timeout=5.0)
        if self.r is not None:
            try:
                self.r.hset(self.k_meta, "ended_ms", int(time.time() * 1000))
            except Exception:
                pass
        if self.fh is not None:
            self.fh.close()
        print("[DB] Cerrado: %d documentos guardados, %d descartados (%s)."
              % (self.n, self.dropped, "Redis" if self.r is not None else self.jsonl_path))


# --------------------------------------------------------------------------- #
# 3h. ROUTER DE EJERCICIOS UNIFICADO (STS, ABD, TALONES, PUNTAS, AUTO)
# --------------------------------------------------------------------------- #


class ExerciseRouter(object):
    """Orquesta por cada inferencia: planos anatomicos -> caracteristicas -> clasificador ->
    (instanciar/activar) STSCounter, AbdCounter o HeelToeCounter -> counter.update()."""

    VALID_EXERCISES = ("sts", "abd", "talones", "puntas")

    def __init__(self, factory, forced, clf, collector, plane_est, thr=0.3, min_switch_s=2.0,
                 proto_mgr=None, patient_name="Paciente_01"):
        self.factory, self.forced = factory, forced
        self.clf, self.collector, self.plane_est = clf, collector, plane_est
        self.extractor = FeatureExtractor(thr)
        self.min_switch_s = min_switch_s
        self.counters, self.kind, self.counter, self.last_switch = {}, None, None, -1e9
        self.planes = None
        self.menu = False
        self.paused = False
        self.notice_txt, self.notice_until = "", 0.0
        self.notice_col = (0, 220, 255)
        self.proto_mgr = proto_mgr
        self.patient_name = patient_name
        self.rest_until = 0.0
        self.session_finished_auto = False
        self.input_mode = None      # None | 'name' | 'age'
        self.input_buf = ""
        self.pending_name = ""

    def notify(self, txt, t, duration=3.5, col=(0, 220, 255)):
        self.notice_txt = txt
        self.notice_until = t + duration
        self.notice_col = col

    def update_abd_protocol(self, abd_peak_deg):
        c = self.counters.get("abd")
        if c is not None:
            c.peak_min = float(abd_peak_deg)
            c.peak_down = max(20.0, float(abd_peak_deg) - 5.0)

    def can_start_exercise(self, mode, t):
        if self.proto_mgr is None:
            return True
        st = self.proto_mgr.get_daily_status(self.patient_name)
        if st["estado_dia"] == "DIA_DESCANSO":
            self.notify(
                "HOY ES DIA DE DESCANSO (%s). Presiona [F] si deseas autorizar sesion hoy."
                % st["dia_label"], t, 4.0, (0, 165, 255)
            )
            return False
        if st["estado_dia"] == "COMPLETO":
            self.notify(
                "EJERCICIOS COMPLETOS POR HOY. Sesion terminada ([R] reiniciar hoy | [Q] salir).",
                t, 4.0, (0, 255, 0)
            )
            return False
        if mode in self.VALID_EXERCISES:
            ej_info = st["ejercicios"].get(mode.upper())
            if ej_info and ej_info.get("completado", False):
                nxt = self.proto_mgr.next_pending_exercise(self.patient_name)
                nxt_lbl = nxt.upper() if nxt else "otro ejercicio"
                self.notify(
                    "%s ya esta completo hoy (%d/%d series). Realiza %s."
                    % (mode.upper(), ej_info["series_completadas"], ej_info["series_meta"], nxt_lbl),
                    t, 4.0, (0, 255, 180)
                )
                return False
        return True

    def set_mode(self, mode, t, check_plan=True):
        if mode == "menu":
            self.menu = True
            self.paused = False
            self.rest_until = 0.0
            return True
        if check_plan and not self.can_start_exercise(mode, t):
            self.menu = True
            return False
        if mode == "auto" and not (self.clf is not None and self.clf.model is not None):
            self.notify("Auto no disponible: falta el modelo (--model)", t, 3.0, (0, 0, 255))
            return False
        self.menu = False
        self.paused = False
        self.forced = None if mode == "auto" else mode
        self.kind, self.counter, self.last_switch = None, None, -1e9
        if mode == "auto":
            self.clf.stable = None
            self.clf.votes.clear()
        else:
            self._activate(mode, t)
        print("[MENU] modo -> %s" % mode.upper())
        return True

    def _activate(self, kind, t):
        if kind == "abd" and self.proto_mgr is not None:
            st = self.proto_mgr.get_daily_status(self.patient_name)
            self.update_abd_protocol(st["abd_peak_deg"])
        if kind == self.kind:
            return
        if self.kind is not None and (t - self.last_switch) < self.min_switch_s:
            return
        c = self.counters.get(kind)
        if c is None:
            c = self.counters[kind] = self.factory(kind)
        if kind == "abd" and self.proto_mgr is not None:
            st = self.proto_mgr.get_daily_status(self.patient_name)
            self.update_abd_protocol(st["abd_peak_deg"])
        c._hard_reset(t)
        self.kind, self.counter, self.last_switch = kind, c, t
        print("[EJERCICIO] activo -> %s" % kind.upper())

    def step(self, kpts, conf, t):
        if self.menu or self.paused or self.input_mode is not None:
            return
        if self.rest_until > t:
            return
        self.planes = self.plane_est.update(kpts, conf) if self.plane_est is not None else None
        if self.clf is not None:
            feat = self.extractor.update(kpts, conf, t)
            if feat is not None:
                self.clf.push(t, feat)
                if self.collector is not None:
                    self.collector.push(t, self.clf.window_vector())
        kind = self.forced if self.forced in self.VALID_EXERCISES else (
            self.clf.stable_exercise() if (self.forced is None and self.clf is not None) else None)
        if kind is not None:
            self._activate(kind, t)
        if self.counter is not None:
            self.counter.update(kpts, conf, t, self.planes)

    def snapshot(self, t):
        c = self.counter
        ev = getattr(c, "rep_eval", None) if c is not None else None
        heel_events = list(getattr(c, "_pending_events", []) or []) if c is not None else []
        if c is not None and hasattr(c, "_pending_events"):
            c._pending_events = []
        if c is not None and ev is not None:
            if self.proto_mgr is not None and self.kind and "serie" not in ev:
                s_lbl, r_lbl = self.proto_mgr.current_set_rep_labels(
                    self.patient_name, self.kind.upper())
                if s_lbl:
                    ev["serie"] = s_lbl
                if r_lbl:
                    ev["rep_en_serie"] = r_lbl
            c.rep_eval = None
        d = {"ejercicio": self.kind,
             "estado": c.state if c is not None else None,
             "angulos": dict((k, round(float(v), 2)) for k, v in c.cur.items()) if c is not None else {},
             "ok": c.ok if c is not None else 0,
             "fallidas": c.fail if c is not None else 0,
             "evaluacion": ev,
             "heel_events": heel_events,
             "alertas": [txt for txt, _ in c.feedback.active(t)] if c is not None else [],
             "yaw_camara": round(self.planes.yaw_deg, 1) if self.planes is not None else None}
        if self.clf is not None:
            d["clf"] = {"label": self.clf.label, "p": round(float(self.clf.prob), 3),
                        "estable": self.clf.stable}
        return d

    def lost(self, t):
        self.extractor.reset()
        if self.counter is not None:
            self.counter.notify_lost(t)

    def reset_counts(self):
        for c in self.counters.values():
            c.reset_counts()

    def close(self):
        for o in (self.collector, self.logger_ref):
            if o is not None:
                o.close()

    logger_ref = None


# --------------------------------------------------------------------------- #
# 4. DIBUJO Y HUD UNIFICADO (ENCABEZADO DE PLAN + METRICAS + FPS + CONTROLES)
# --------------------------------------------------------------------------- #


def draw_plan_header(img, router, t, patient_name):
    """Encabezado superior semitransparente con el Plan del Paciente y estado de los 4 ejercicios."""
    if router.proto_mgr is None:
        return 24
    st = router.proto_mgr.get_daily_status(patient_name)
    ej_map = st.get("ejercicios", {})
    n_ej = max(1, len(ej_map))
    box_h = 84 + 20 * n_ej

    h, w = img.shape[:2]
    bh = min(h, box_h)
    img[0:bh, :] >>= 1

    def put(txt, x, y, col=(255, 255, 0), scale=0.46, thick=1):
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_8)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, thick, cv2.LINE_AA)

    sim_tag = " (Simulado)" if st["dia_simulado"] else ""
    active_ej = (router.kind or "ninguno").upper()
    put("PLAN DE EJERCICIOS DEL PACIENTE   |   Ejercicio activo: %s" % active_ej,
        10, 20, (255, 255, 255), 0.50, 1)
    put("Paciente: %s (%d anos)  |  Dia: %s%s  |  Dias: %s  |  Nivel: %s"
        % (st["paciente"], st["edad"], st["dia_label"], sim_tag,
           st["dias_asignados_str"], st["nivel"]),
        10, 40, (0, 220, 255), 0.44, 1)

    estado_dia = st["estado_dia"]
    pause_tag = "  [PAUSADO]" if router.paused else ""
    if estado_dia == "COMPLETO":
        estado_txt = "ESTADO DE HOY (%s): EJERCICIOS COMPLETOS - SESION TERMINADA" % st["dia_label"].upper()
        col_estado = (0, 255, 0)
    elif estado_dia == "DIA_DESCANSO":
        estado_txt = "ESTADO DE HOY (%s): DIA DE DESCANSO ([F] autorizar hoy)" % st["dia_label"].upper()
        col_estado = (0, 165, 255)
    else:
        estado_txt = "ESTADO DE HOY (%s): EJERCICIOS PENDIENTES%s" % (st["dia_label"].upper(), pause_tag)
        col_estado = (0, 255, 255)
    put(estado_txt, 10, 61, col_estado, 0.46, 1)

    y_line = 82
    for ej_k in ProtocolManager.EXERCISE_ORDER:
        if ej_k not in ej_map:
            continue
        ej_d = ej_map[ej_k]
        done = bool(ej_d.get("completado", False))
        s_done = int(ej_d.get("series_completadas", 0))
        s_meta = int(ej_d.get("series_meta", 2))
        s_rem = max(0, s_meta - s_done)
        r_cur = int(ej_d.get("reps_en_serie_actual", 0))
        r_meta = int(ej_d.get("repeticiones_meta", 10))
        r_rem = max(0, r_meta - r_cur)
        is_cur = (active_ej == ej_k)
        prefix = ">>" if is_cur else " *"
        if done:
            st_line = "COMPLETADO (%d/%d series | OK:%d Malas:%d)" % (
                s_done, s_meta, ej_d.get("reps_correctas_hoy", 0), ej_d.get("reps_incorrectas_hoy", 0))
            col_l = (0, 255, 0)
        elif is_cur:
            st_line = "Serie %d/%d (faltan %d series) | Faltan %d/%d reps (OK:%d Malas:%d)" % (
                min(s_done + 1, s_meta), s_meta, s_rem, r_rem, r_meta,
                ej_d.get("reps_correctas_hoy", 0), ej_d.get("reps_incorrectas_hoy", 0))
            col_l = (0, 230, 255)
        else:
            st_line = "Faltan %d series (Serie %d/%d | Faltan %d/%d reps)" % (
                s_rem, min(s_done + 1, s_meta), s_meta, r_rem, r_meta)
            col_l = (190, 190, 190)
        put("%s %s: %s" % (prefix, ej_k, st_line), 14, y_line, col_l, 0.43, 1)
        y_line += 19

    if router.rest_until > t:
        rem = int(max(0.0, router.rest_until - t)) + 1
        put("DESCANSO ENTRE SERIES: %d s restantes ([Espacio] saltar)" % rem,
            14, y_line + 4, (0, 200, 255), 0.46, 1)
        y_line += 22

    if t < router.notice_until and router.notice_txt:
        put(">> " + router.notice_txt, 14, y_line + 4, router.notice_col, 0.45, 1)
        y_line += 22

    return y_line + 8


def draw_exercise_status(img, res, title, org=(10, 175)):
    """Panel biomecanico de Elevacion de Talones (TALONES) y Elevacion de Puntas (PUNTAS)."""
    if res is None:
        return
    col   = GREEN if res["valid"] else ORANGE
    m_txt = "-" if res["metric"] is None else "%+.1f deg" % res["metric"]
    k_txt = "-" if res.get("knee_angle") is None else "%.0f deg" % res["knee_angle"]
    t_txt = "-" if res.get("torso_incl") is None else "%.0f deg" % res["torso_incl"]
    lines = [
        ("%s [%s] %s | REPS OK: %d (rech: %d, caidas: %d)" % (
            title, res["side"], res["phase"],
            res["reps"], res["rejected"], res["drops"]), col),
        ("elev=%s   w=%.0f deg/s   rodilla=%s   tronco=%s" % (
            m_txt, res["omega"], k_txt, t_txt), col),
    ]
    if res.get("event"):
        lines.append((res["event"], res["event_col"]))
    if res.get("msg"):
        lines.append((res["msg"], (0, 255, 255)))
    lines += [(r, ORANGE) for r in res["reasons"][:2]]
    for i, (txt, c) in enumerate(lines):
        p = (org[0], org[1] + 22 * i)
        cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.50, c,         1, cv2.LINE_AA)


def draw_rep_hud(img, counter, t, y0=175, planes=None):
    """Panel biomecanico de Sit-to-Stand (STS) y Abduccion de Cadera (ABD)."""
    lines = [(txt, (255, 255, 0)) for txt in counter.hud_lines()]
    if planes is not None and not planes.identity:
        lines.append(("perspectiva yaw ~%.0f deg (cos=%.2f)" % (planes.yaw_deg, planes.cos_psi),
                      (200, 200, 140)))
    if counter.last_result == "OK":
        res_txt, res_col = "CORRECTA", (0, 255, 0)
    elif counter.last_result == "-":
        res_txt, res_col = "En espera...", (200, 200, 200)
    else:
        res_txt = counter.last_result.replace("FALLIDA:", "INCORRECTA -")[:65]
        res_col = (0, 0, 255)
    lines.append(("Ultima rep: " + res_txt, res_col))
    cols = {"bad": (0, 0, 255), "warn": (0, 200, 255), "ok": (0, 255, 0)}
    for text, lvl in counter.feedback.active(t):
        lines.append((">> " + text, cols.get(lvl, (0, 0, 255))))
    for i, (txt, col) in enumerate(lines):
        y = y0 + 21 * i
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def draw_patient_input_modal(img, router):
    """Modal interactivo en pantalla para ingresar Nombre y Edad del paciente con [I] / [E]."""
    if router.input_mode is None:
        return
    h, w = img.shape[:2]
    bx0, by0 = max(35, w // 7), max(65, h // 3)
    bx1, by1 = min(w - 35, w - w // 7), min(h - 50, by0 + 165)
    img[by0:by1, bx0:bx1] >>= 2
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 230, 255), 2, cv2.LINE_8)

    pts_list = router.proto_mgr.patient_list() if router.proto_mgr is not None else []
    pts_preview = ", ".join(pts_list[:6])
    if router.input_mode == "name":
        title_m = "ESCRIBE EL NOMBRE DEL PACIENTE Y PRESIONA [ENTER]"
        sub_m = "Nombre: %s_" % router.input_buf
        hint1 = "Guardados ([TAB] elegir): %s" % pts_preview
        hint2 = "[Enter] Confirmar  |  [Backspace] Borrar  |  [Esc] Cerrar"
    else:
        title_m = "PACIENTE '%s': ESCRIBE SU EDAD Y PRESIONA [ENTER]" % router.pending_name
        sub_m = "Edad (anos): %s_" % router.input_buf
        hint1 = "Se asignaran sus dias, nivel, series, reps y rango ABD segun la edad."
        hint2 = "[Enter] Guardar  |  [Backspace] Borrar  |  [Esc] Cancelar"
    cv2.putText(img, title_m, (bx0 + 16, by0 + 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(img, sub_m, (bx0 + 16, by0 + 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 255, 160), 2, cv2.LINE_AA)
    cv2.putText(img, hint1, (bx0 + 16, by0 + 116), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, hint2, (bx0 + 16, by0 + 144), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 200, 255), 1, cv2.LINE_AA)


def draw_menu(img, router, t, patient_name="Paciente_01"):
    """Pantalla Home interactiva con los 4 ejercicios y el estado diario del paciente."""
    h, w = img.shape[:2]
    img[:, :] >>= 2

    st = router.proto_mgr.get_daily_status(patient_name) if router.proto_mgr is not None else None
    items = []
    items.append(("PLAN UNIFICADO DE EJERCICIOS DEL PACIENTE (STS + ABD + TALONES + PUNTAS)",
                  (255, 255, 255), 0.52))

    if st is not None:
        sim_tag = " (Simulado)" if st["dia_simulado"] else ""
        items.append((
            "Paciente: %s (%d anos)  |  Nivel: %s  |  Dia: %s%s  |  Dias asignados: %s"
            % (st["paciente"], st["edad"], st["nivel"], st["dia_label"],
               sim_tag, st["dias_asignados_str"]),
            (0, 230, 255), 0.46
        ))

        estado_dia = st["estado_dia"]
        if estado_dia == "DIA_DESCANSO":
            items.append((
                "ESTADO DE HOY (%s): DIA DE DESCANSO" % st["dia_label"].upper(),
                (0, 165, 255), 0.56
            ))
            items.append((
                "Hoy no tocan ejercicios. Dias asignados: %s." % st["dias_asignados_str"],
                (210, 230, 255), 0.45
            ))
            items.append((
                "([F] Autorizar sesion hoy   |   [I] Otro paciente   |   [D] Simular dia   |   [Q] Salir)",
                (180, 180, 180), 0.43
            ))
        elif estado_dia == "COMPLETO":
            items.append((
                "ESTADO DE HOY (%s): EJERCICIOS COMPLETOS - SESION TERMINADA"
                % st["dia_label"].upper(),
                (0, 255, 0), 0.54
            ))
            for ej_k in ProtocolManager.EXERCISE_ORDER:
                ej_d = st["ejercicios"].get(ej_k)
                if ej_d:
                    label_ej = ProtocolManager.EXERCISE_LABELS.get(ej_k, ej_k)
                    items.append((
                        "  [OK] %s: %d/%d series completadas | Buenas: %d | Malas: %d"
                        % (label_ej, ej_d["series_completadas"], ej_d["series_meta"],
                           ej_d["reps_correctas_hoy"], ej_d["reps_incorrectas_hoy"]),
                        (120, 255, 120), 0.44
                    ))
            items.append((
                "([Q] Cerrar sesion   |   [I] Ingresar otro paciente   |   [R] Reiniciar hoy)",
                (190, 230, 190), 0.43
            ))
        else:
            items.append((
                "ESTADO DE HOY (%s): EJERCICIOS PENDIENTES (Rango ABD >= %.0f deg)"
                % (st["dia_label"].upper(), st["abd_peak_deg"]),
                (0, 255, 255), 0.50
            ))
            for ej_k in ProtocolManager.EXERCISE_ORDER:
                ej_d = st["ejercicios"].get(ej_k)
                if ej_d:
                    label_ej = ProtocolManager.EXERCISE_LABELS.get(ej_k, ej_k)
                    done = bool(ej_d.get("completado", False))
                    s_done = int(ej_d.get("series_completadas", 0))
                    s_meta = int(ej_d.get("series_meta", 2))
                    s_rem = max(0, s_meta - s_done)
                    r_cur = int(ej_d.get("reps_en_serie_actual", 0))
                    r_meta = int(ej_d.get("repeticiones_meta", 10))
                    r_rem = max(0, r_meta - r_cur)
                    status_tag = "COMPLETO" if done else (
                        "Faltan %d series (Serie %d/%d | Faltan %d/%d reps)"
                        % (s_rem, min(s_done + 1, s_meta), s_meta, r_rem, r_meta)
                    )
                    col_ej = (0, 255, 0) if done else (0, 220, 255)
                    items.append((
                        "  * %s: %s" % (label_ej, status_tag),
                        col_ej, 0.45
                    ))
            items.append((
                "[P] Iniciar Plan  |  [S] STS  |  [A] Abduccion  |  [H] Talones  |  [T] Puntas  |  [G] Auto",
                (0, 255, 128), 0.45
            ))

    items.append((
        "[I] Ingresar Paciente  [E] Edad  [N]/[B] Cambiar Paciente  [D] Dia  [R] Reiniciar  [Q] Salir",
        (185, 185, 185), 0.42
    ))

    if t < router.notice_until and router.notice_txt:
        items.append((">> " + router.notice_txt, router.notice_col, 0.46))

    y = max(32, h // 12)
    step_y = 24
    for txt, col, sc in items:
        cv2.putText(img, txt, (22, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(img, txt, (22, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        y += step_y

    draw_patient_input_modal(img, router)


COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
]
FOOT_SKELETON = [(15, 19), (19, 17), (19, 18), (16, 22), (22, 20), (22, 21)]
SKELETON = COCO_SKELETON + FOOT_SKELETON

LIMB_COLOR = (0, 255, 128)
FOOT_LIMB_COLOR = (255, 200, 0)
KPT_COLOR = (0, 128, 255)
FOOT_KPT_COLOR = (0, 0, 255)


def draw_pose(img, kpts, conf, thr=0.3):
    n = min(len(kpts), N_KPTS)
    pts = np.clip(np.round(kpts), -10000, 10000).astype(np.int32).tolist()
    for a, b in SKELETON:
        if a < n and b < n and conf[a] > thr and conf[b] > thr:
            col = FOOT_LIMB_COLOR if (a >= 17 or b >= 17) else LIMB_COLOR
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_8)
    for i in range(n):
        if conf[i] > thr:
            col = FOOT_KPT_COLOR if i >= 17 else KPT_COLOR
            cv2.circle(img, tuple(pts[i]), 4, col, -1, cv2.LINE_8)


def draw_foot_status(img, feet, kpts):
    """Dibuja el estado PLANO / LEVANTADO y los grados junto a cada talon."""
    heel = {"I": 19, "D": 22}
    for side, (state, ang) in feet.items():
        if state is None:
            txt, col = "%s: ?" % side, (160, 160, 160)
        elif state:
            txt, col = "%s: PLANO %.0f" % (side, ang), (0, 255, 0)
        else:
            txt, col = "%s: LEVANTADO %.0f" % (side, ang), (0, 0, 255)
        x, y = kpts[heel[side]]
        org = (max(int(np.clip(x, 0, 1e5)) - 40, 0),
               max(int(np.clip(y, 0, 1e5)) + 24, 12))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_8)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.50, col,         1, cv2.LINE_AA)


def put_hud(img, cam_fps, loop_fps, infer_ms, tracking, n_good):
    """HUD tecnico en la esquina inferior izquierda con Camera FPS, Loop FPS, infer ms y kpts."""
    h = img.shape[0]
    lines = [
        ("Camera FPS: %.1f" % cam_fps, (0, 255, 0)),
        ("Loop FPS: %.1f  (infer %.1f ms)" % (loop_fps, infer_ms), (0, 200, 255)),
        ("%s  kpts: %d" % ("TRACK" if tracking else "SEARCH", n_good),
         (0, 255, 0) if tracking else (0, 0, 255)),
    ]
    for i, (txt, col) in enumerate(lines):
        y = h - 105 + 26 * i
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_8)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col,       2, cv2.LINE_AA)


def draw_controls(img):
    """Barra inferior con todos los atajos rapidos de teclado."""
    h, w = img.shape[:2]
    controls = (
        "[S] STS  [A] Abd  [H] Talones  [T] Puntas  [G] Auto  "
        "[M] Menu  [P] Pausar  [R] Reset  [I] Paciente  [D] Dia  [Q] Salir"
    )
    cv2.putText(img, controls, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_8)
    cv2.putText(img, controls, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# 5. UTILIDADES DE DIAGNOSTICO (SWEEP / IMAGE TEST) Y BUCLE PRINCIPAL
# --------------------------------------------------------------------------- #

KPT_NAMES = ["nariz", "ojo_i", "ojo_d", "oreja_i", "oreja_d", "hombro_i", "hombro_d",
             "codo_i", "codo_d", "muneca_i", "muneca_d", "cadera_i", "cadera_d",
             "rodilla_i", "rodilla_d", "tobillo_i", "tobillo_d",
             "puntaG_i", "puntaP_i", "talon_i", "puntaG_d", "puntaP_d", "talon_d"]


def num_kpts(model, args):
    return max(1, min(args.num_kpts, model.output_shape[1]))


def full_frame_roi(fw, fh, aspect):
    return make_roi(fw / 2.0, fh / 2.0, max(float(fh), fw / aspect), aspect)


def candidate_rois(fw, fh, aspect):
    return [("centro", make_roi(fw / 2.0, fh / 2.0, fh, aspect)),
            ("completo", full_frame_roi(fw, fh, aspect))]


def run_once(model, frame, roi, to_rgb, norm, nk):
    _, _, in_h, in_w = model.input_shape
    K, hm_h, hm_w = model.output_shape[1:]
    pre = PosePreprocessor(in_w, in_h, to_rgb=to_rgb, norm=norm)
    pre.run(frame, roi, model.input_view)
    hm = model.infer(nk).reshape(K, hm_h, hm_w)[:nk]
    kpts, conf = decode_heatmaps(hm, roi)
    hmf = hm.astype(np.float32)
    return kpts, conf, float(np.nanmin(hmf)), float(np.nanmax(hmf))


def anatomy_score(kpts, conf, thr):
    pairs = [((0,), (5, 6)), ((5, 6), (11, 12)), ((11, 12), (13, 14)), ((13, 14), (15, 16))]
    ok = tot = 0
    for upper, lower in pairs:
        u = [kpts[i, 1] for i in upper if conf[i] > thr]
        l = [kpts[i, 1] for i in lower if conf[i] > thr]
        if u and l:
            tot += 1
            ok += int(np.mean(u) < np.mean(l))
    return ok, tot


def grab_frame(args):
    cam = USBCameraStream(args.device, args.width, args.height, args.fps,
                          backend=args.cam).start()
    frame, last_id = None, -1
    try:
        print("[TEST] Ponte de pie, cuerpo completo, centrado y a 2-3 m. Capturando en %d s..."
              % int(args.delay))
        t0 = time.time()
        while time.time() - t0 < args.delay:
            f, last_id = cam.read(last_id)
            if f is not None:
                frame = f.copy()
    finally:
        cam.stop()
    return frame


def image_test(model, frame, args):
    fh, fw = frame.shape[:2]
    aspect = model.input_shape[3] / float(model.input_shape[2])
    nk = num_kpts(model, args)
    os.makedirs("pose_out", exist_ok=True)
    for rname, roi in candidate_rois(fw, fh, aspect):
        kpts, conf, hmin, hmax = run_once(model, frame, roi, not args.bgr_input, args.norm, nk)
        print("\n[ROI %s] (%.0f,%.0f,%.0f,%.0f)  heatmap min=%.2f max=%.2f"
              % (rname, roi[0], roi[1], roi[2], roi[3], hmin, hmax))
        for i in range(min(len(kpts), len(KPT_NAMES))):
            print("  %-10s x=%7.1f y=%7.1f conf=%.2f" % (KPT_NAMES[i], kpts[i, 0], kpts[i, 1], conf[i]))
        img = frame.copy()
        draw_pose(img, kpts, conf, args.kpt_thr)
        if len(kpts) >= N_KPTS:
            mon = FootMonitor(args.flat_deg, args.kpt_thr)
            draw_foot_status(img, mon.update(kpts, conf), kpts)
        cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                      (255, 255, 0), 1)
        path = os.path.join("pose_out", "pose_%s.jpg" % rname)
        cv2.imwrite(path, img)
        print("  -> guardado %s" % path)


def sweep(model, frame, args):
    fh, fw = frame.shape[:2]
    aspect = model.input_shape[3] / float(model.input_shape[2])
    nk = num_kpts(model, args)
    os.makedirs("sweep_out", exist_ok=True)
    cv2.imwrite(os.path.join("sweep_out", "00_input.jpg"), frame)
    rows = []
    print("\n%-30s %9s %9s %7s %6s  %s"
          % ("variante", "conf_max", "conf_mean", "n>thr", "anat", "heatmap[min,max]"))
    for rname, roi in candidate_rois(fw, fh, aspect):
        for norm in ("imagenet", "zero_one", "minus1_1", "raw"):
            for to_rgb in (True, False):
                kpts, conf, hmin, hmax = run_once(model, frame, roi, to_rgb, norm, nk)
                n = int((conf > args.kpt_thr).sum())
                ok, tot = anatomy_score(kpts, conf, args.kpt_thr)
                tag = "%s_%s_%s" % (rname, "rgb" if to_rgb else "bgr", norm)
                print("%-30s %9.2f %9.2f %7d %3d/%-2d  [%.2f, %.2f]"
                      % (tag, float(conf.max()), float(conf.mean()), n, ok, tot, hmin, hmax))
                img = frame.copy()
                draw_pose(img, kpts, conf, args.kpt_thr)
                cv2.rectangle(img, (int(roi[0]), int(roi[1])),
                              (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (255, 255, 0), 1)
                cv2.imwrite(os.path.join("sweep_out", "sweep_%s.jpg" % tag), img)
                good = (tot >= 3 and ok == tot)
                rows.append((good, n, float(conf.mean()), tag, norm, to_rgb))
    rows.sort(reverse=True)
    print("\nMejores candidatas (anatomia coherente, mas puntos, mayor confianza media):")
    for good, n, mean, tag, norm, to_rgb in rows[:3]:
        print("  %-28s n>thr=%d mean=%.2f  ->  --norm %s%s"
              % (tag, n, mean, norm, "" if to_rgb else " --bgr-input"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline Unificado HRNet WholeBody (STS + ABD + TALONES + PUNTAS) en Jetson Nano"
    )
    p.add_argument("--engine", required=True, help="Ruta al .engine (FP16)")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam", default="v4l2", choices=["v4l2", "nvv4l2", "nvjpeg", "jpegdec"],
                   help="Captura: v4l2 (OpenCV directo, CPU) o GStreamer con nvv4l2 / nvjpeg / jpegdec")
    p.add_argument("--no-gst", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip", type=int, default=2,
                   help="Inferir 1 de cada N frames (2 por defecto en Jetson Nano para >=8-15 FPS)")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="Umbral de confianza")
    p.add_argument("--num-kpts", type=int, default=N_KPTS, help="Canales de heatmap a decodificar (23)")
    p.add_argument("--no-track", action="store_true", help="ROI fijo al centro")
    p.add_argument("--bgr-input", action="store_true", help="No convertir a RGB")
    p.add_argument("--norm", default="imagenet",
                   choices=["imagenet", "zero_one", "minus1_1", "raw"])
    p.add_argument("--flat-deg", type=float, default=22.0,
                   help="Angulo maximo talon->puntas para considerar el pie plano")
    p.add_argument("--flat-hold", type=int, default=3)
    p.add_argument("--no-foot", action="store_true", help="Desactivar etiqueta de pie plano")
    p.add_argument("--side", default="auto", choices=["auto", "I", "D"],
                   help="Lado del cuerpo para TALONES / PUNTAS (auto, I, D)")

    g = p.add_argument_group("Ejercicios y contadores de repeticiones")
    g.add_argument("--exercise", default="menu",
                   choices=["menu", "auto", "none", "sts", "abd", "talones", "puntas"],
                   help="menu = pantalla de inicio (por defecto); sts / abd / talones / puntas = "
                        "arrancar directo; auto = clasificador temporal; none = sin contador")
    g.add_argument("--refractory-ms", type=float, default=400.0)
    g.add_argument("--sit-deg", type=float, default=95.0, help="STS: rodilla < este valor = sedente")
    g.add_argument("--stand-deg", type=float, default=165.0, help="STS: rodilla > este valor = de pie")
    g.add_argument("--trunk-flex-max", type=float, default=30.0)
    g.add_argument("--trunk-upright", type=float, default=10.0)
    g.add_argument("--sts-omega-max", type=float, default=150.0)
    g.add_argument("--abd-rest", type=float, default=15.0)
    g.add_argument("--abd-peak", type=float, default=45.0)
    g.add_argument("--abd-trunk-max", type=float, default=15.0)
    g.add_argument("--abd-knee-min", type=float, default=165.0)
    g.add_argument("--abd-knee-margin", type=float, default=15.0)
    g.add_argument("--abd-omega-max", type=float, default=120.0)
    g.add_argument("--heel-up-deg", type=float, default=10.0, help="TALONES: elevacion minima (deg)")
    g.add_argument("--toe-up-deg", type=float, default=6.0, help="PUNTAS: elevacion minima (deg)")

    g2 = p.add_argument_group("Perspectiva, clasificador, recoleccion y registro CSV")
    g2.add_argument("--no-plane", action="store_true")
    g2.add_argument("--plane-ref", type=float, default=1.0)
    g2.add_argument("--model", default="exercise_clf.joblib")
    g2.add_argument("--min-conf", type=float, default=0.55)
    g2.add_argument("--clf-hold", type=int, default=3)
    g2.add_argument("--collect", action="store_true")
    g2.add_argument("--collect-label", default=None,
                    choices=["STS", "ABD_I", "ABD_D", "REST", "TRANSITION", "TALONES", "PUNTAS"])
    g2.add_argument("--collect-out", default="features_windows.csv")
    g2.add_argument("--collect-stride", type=int, default=4)
    g2.add_argument("--log-csv", default="")
    g2.add_argument("--no-log", action="store_true")

    g3 = p.add_argument_group("Protocolos por edad y Base de Datos NoSQL")
    g3.add_argument("--db", default="redis", choices=["redis", "jsonl", "none"])
    g3.add_argument("--db-url", default="redis://127.0.0.1:6379/0")
    g3.add_argument("--db-maxlen", type=int, default=0)
    g3.add_argument("--db-dir", default="logs")
    g3.add_argument("--db-all", action="store_true")
    g3.add_argument("--session-id", default="")
    p.add_argument("--patient", "--paciente", dest="patient", default="",
                   help="Nombre inicial del paciente (si se omite, se puede escribir con [I])")
    p.add_argument("--age", "--edad", dest="age", type=int, default=None,
                   help="Edad del paciente en anos")
    g3.add_argument("--protocols", default="protocolos.json",
                    help="Archivo JSON editable con los protocolos por edad y planes de pacientes")
    g3.add_argument("--day", default=None,
                    help="Simular dia de la semana (lunes..domingo)")
    g3.add_argument("--auto-close", action="store_true",
                    help="Cerrar automaticamente al completar todos los ejercicios del dia")
    g3.add_argument("--db-file", default="bd_pacientes.json1",
                    help="Archivo NoSQL JSONL donde se acumulan las sesiones de los pacientes")

    g4 = p.add_argument_group("Validacion experimental del Plan de Trabajo (OE1, OE2, OE5)")
    g4.add_argument("--subject", default="",
                    help="ID del voluntario (S01..S12): metadato del corpus y del conteo manual")
    g4.add_argument("--view", default="", choices=["", "frontal", "obl30", "obl45", "lateral"],
                    help="Vista de camara del experimento de geometria (OE1)")
    g4.add_argument("--light", default="", choices=["", "alta", "baja"],
                    help="Condicion de iluminacion (OE2)")
    g4.add_argument("--band", default="", choices=["", "ninguna", "baja", "media", "alta"],
                    help="Nivel de resistencia de la banda elastica (OE2)")
    g4.add_argument("--perf-log", default="",
                    help="CSV de rendimiento (FPS, latencia, CPU, RAM, temperatura, visibilidad, "
                         "cinematica) para la Tabla 2 / OE1. Ej: logs/perf_frontal.csv")

    p.add_argument("--sweep", action="store_true")
    p.add_argument("--image", default=None)
    p.add_argument("--delay", type=float, default=5.0)
    p.add_argument("--draw-always", action="store_true")
    p.add_argument("--show-roi", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args()
    if args.no_gst:
        args.cam = "v4l2"
    return args


def main():
    args = parse_args()

    model = TRTInference(args.engine)
    if len(model.input_shape) != 4 or len(model.output_shape) != 4:
        raise RuntimeError("Se esperaba input NCHW y output (1,K,h,w); input=%s output=%s"
                           % (str(model.input_shape), str(model.output_shape)))
    _, _, in_h, in_w = model.input_shape
    K, hm_h, hm_w = model.output_shape[1:]
    nk = num_kpts(model, args)
    print("[TRT] input %s (%s) | heatmaps %s (%s) | usando %d kpts"
          % (str(model.input_shape), model.input_dtype,
             str(model.output_shape), model.output_dtype, nk))
    use_feet = (not args.no_foot) and nk >= N_KPTS
    if not use_feet and not args.no_foot:
        print("[AVISO] El engine tiene %d canales (<%d): no hay puntos de pie." % (K, N_KPTS))

    pre = PosePreprocessor(in_w, in_h, to_rgb=not args.bgr_input, norm=args.norm)
    kfilter = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
    tracker = ROITracker(aspect=in_w / float(in_h), thr=args.kpt_thr,
                         enabled=not args.no_track)
    foot_mon = FootMonitor(args.flat_deg, args.kpt_thr, hold=args.flat_hold)

    for _ in range(5):
        model.infer(nk)

    if args.sweep or args.image:
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise RuntimeError("No pude leer la imagen %s" % args.image)
        else:
            frame = grab_frame(args)
            if frame is None:
                raise RuntimeError("No se pudo capturar un frame de la camara")
        if args.sweep:
            sweep(model, frame, args)
        else:
            image_test(model, frame, args)
        return

    cam = USBCameraStream(args.device, args.width, args.height, args.fps,
                          backend=args.cam).start()

    proto_mgr = ProtocolManager(path=args.protocols, override_day=args.day)
    cur_patient, _ = proto_mgr.ensure_patient(args.patient or "Paciente_01", age=args.age)
    init_status = proto_mgr.get_daily_status(cur_patient)
    args.abd_peak = float(init_status["abd_peak_deg"])
    print("[PLAN] Paciente: %s (%d anos, %s) | Dia: %s | Estado hoy: %s | Rango ABD>=%.0f deg"
          % (cur_patient, init_status["edad"], init_status["nivel"],
             init_status["dia_label"], init_status["estado_dia"], args.abd_peak))

    logger = None
    if not args.no_log:
        log_path = args.log_csv or os.path.join(
            "logs", "reps_%s.csv" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        logger = SessionLogger(log_path, meta={"subject": args.subject or cur_patient,
                                               "view": args.view, "light": args.light,
                                               "band": args.band})

    framedb = None
    if args.db != "none":
        sid = args.session_id or ("%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), cur_patient))
        framedb = FrameDB(sid, backend=args.db, url=args.db_url, maxlen=args.db_maxlen,
                          out_dir=args.db_dir, patient=cur_patient, only_active=not args.db_all,
                          unified_file=args.db_file, age=init_status["edad"],
                          day_name=init_status["dia_semana"], abd_peak=args.abd_peak)

    pending_post_rep = []

    def on_protocol_rep(title, ok, reasons, info):
        st_before = proto_mgr.get_daily_status(router.patient_name)
        ej_b = (st_before["ejercicios"].get(title.upper()) or {})
        s_before = int(ej_b.get("series_completadas", 0))
        s_meta = int(ej_b.get("series_meta", 2))
        r_before = int(ej_b.get("reps_en_serie_actual", 0))
        r_meta = int(ej_b.get("repeticiones_meta", 10))

        ev_res = proto_mgr.on_rep(router.patient_name, title, ok, reasons, info)
        if ev_res is None:
            return

        c_obj = router.counters.get(title.lower())
        if c_obj is not None and getattr(c_obj, "rep_eval", None) is not None:
            s_tag = "%d/%d" % (min(s_before + 1, s_meta), s_meta)
            r_num = (r_before + 1) if ev_res["contada"] else r_before
            r_tag = "%d/%d" % (min(r_num, r_meta), r_meta)
            c_obj.rep_eval["serie"] = s_tag
            c_obj.rep_eval["rep_en_serie"] = r_tag

        t_ev = time.perf_counter()
        if ev_res["plan_completado"]:
            msg = ("EJERCICIOS COMPLETOS! %s termino todas sus series de hoy. Sesion terminada."
                   % router.patient_name)
            print("[PLAN] " + msg)
            auto_exit = args.auto_close or bool(
                (proto_mgr.data.get("configuracion_general") or {}).get(
                    "auto_cerrar_al_completar_todo", False)
            )
            pending_post_rep.append(("close" if auto_exit else "menu", msg, (0, 255, 0)))
        elif ev_res["ejercicio_completado"]:
            nxt = proto_mgr.next_pending_exercise(router.patient_name)
            key_map = {"sts": "S", "abd": "A", "talones": "H", "puntas": "T"}
            nxt_txt = (" Siguiente: %s ([P] o [%s])."
                       % (nxt.upper(), key_map.get(nxt, "P"))) if nxt else ""
            msg = ("%s COMPLETADO (%d/%d series)!%s"
                   % (title.upper(), ev_res["series_completadas"], ev_res["series_meta"], nxt_txt))
            print("[PLAN] " + msg)
            pending_post_rep.append(("next" if nxt else "menu", msg, (0, 255, 128), nxt))
        elif ev_res["serie_completada"]:
            rest_s = int(ev_res["descanso_s"])
            router.rest_until = t_ev + float(rest_s)
            msg = ("Serie %d/%d de %s completada! Descanso: %d s ([Espacio] para saltar)."
                   % (ev_res["series_completadas"], ev_res["series_meta"], title.upper(), rest_s))
            print("[PLAN] " + msg)
            router.notify(msg, t_ev, duration=6.0, col=(0, 220, 255))

    def make_counter(kind):
        st_now = proto_mgr.get_daily_status(router.patient_name if router else cur_patient)
        abd_pk = float(st_now["abd_peak_deg"])
        k_low = kind.lower()
        if k_low == "sts":
            c = STSCounter(sit_enter=args.sit_deg, stand_enter=args.stand_deg,
                           trunk_max=args.trunk_flex_max, trunk_upright=args.trunk_upright,
                           omega_max=args.sts_omega_max, refractory_ms=args.refractory_ms,
                           kpt_thr=args.kpt_thr)
        elif k_low == "abd":
            c = AbdCounter(rest_enter=args.abd_rest, rest_exit=args.abd_rest + 5.0,
                           peak_min=abd_pk, peak_down=max(20.0, abd_pk - 5.0),
                           trunk_max=args.abd_trunk_max, knee_min=args.abd_knee_min,
                           knee_margin=args.abd_knee_margin, omega_max=args.abd_omega_max,
                           refractory_ms=args.refractory_ms, kpt_thr=args.kpt_thr)
        elif k_low in ("talones", "puntas"):
            ex_cfg = ExerciseConfig(
                kpt_thr=min(args.kpt_thr, 0.25),
                heel_up_deg=args.heel_up_deg,
                toe_up_deg=args.toe_up_deg,
            )
            c = HeelToeCounter(kind=k_low, cfg=ex_cfg, side=args.side,
                               refractory_ms=args.refractory_ms)
        else:
            raise ValueError("Ejercicio desconocido: %s" % kind)
        c.listeners.append(on_protocol_rep)
        if logger is not None:
            c.listeners.append(logger.on_rep)
        if framedb is not None:
            c.listeners.append(framedb.on_rep)
        return c

    router = None
    clf = collector = None
    if args.exercise in ("auto", "menu") or args.collect:
        clf = TemporalClassifier(args.model if args.exercise in ("auto", "menu") else None,
                                 hold=args.clf_hold, min_conf=args.min_conf)
        if args.collect or (args.exercise == "auto" and clf.model is None):
            collector = DataCollector(args.collect_out, stride=args.collect_stride,
                                      label=args.collect_label,
                                      meta={"subject": args.subject or cur_patient,
                                            "view": args.view, "light": args.light,
                                            "band": args.band})
    mode0 = args.exercise
    if mode0 == "menu" and args.no_display:
        mode0 = "auto" if (clf is not None and clf.model is not None) else "none"
        print("[MENU] Sin ventana no hay menu: usando modo '%s'." % mode0)
    forced = None if mode0 in ("auto", "menu") else mode0
    plane_est = None if args.no_plane else PlaneEstimator(args.kpt_thr, args.plane_ref)
    router = ExerciseRouter(make_counter, forced, clf, collector, plane_est, args.kpt_thr,
                            proto_mgr=proto_mgr, patient_name=cur_patient)
    router.menu = (mode0 == "menu")
    router.logger_ref = logger
    if router.menu and not (args.patient or "").strip():
        router.input_mode = "name"
        router.input_buf = ""

    def switch_active_patient(new_name, new_age=None, t_event=0.0):
        pname, _ = proto_mgr.ensure_patient(new_name, age=new_age)
        st_p = proto_mgr.get_daily_status(pname)
        router.patient_name = pname
        router.reset_counts()
        router.rest_until = 0.0
        args.abd_peak = float(st_p["abd_peak_deg"])
        router.update_abd_protocol(args.abd_peak)
        if framedb is not None:
            framedb.set_patient(pname, age=st_p["edad"], day_name=st_p["dia_semana"],
                                abd_peak=args.abd_peak)
        if st_p["estado_dia"] == "DIA_DESCANSO":
            router.notify("Paciente %s (%d anos): HOY ES DIA DE DESCANSO (%s)."
                          % (pname, st_p["edad"], st_p["dia_label"]),
                          t_event, 4.5, (0, 165, 255))
        elif st_p["estado_dia"] == "COMPLETO":
            router.notify("Paciente %s (%d anos): EJERCICIOS COMPLETOS POR HOY."
                          % (pname, st_p["edad"]),
                          t_event, 4.5, (0, 255, 0))
        else:
            router.notify("Paciente %s (%d anos, %s): Plan del dia listo."
                          % (pname, st_p["edad"], st_p["nivel"]),
                          t_event, 3.5, (0, 230, 255))
        return pname

    if not router.menu and mode0 in ("sts", "abd", "talones", "puntas", "auto"):
        if not router.can_start_exercise(mode0, time.perf_counter()):
            router.set_mode("menu", time.perf_counter(), check_plan=False)
        elif mode0 != "auto":
            router._activate(mode0, time.perf_counter())

    if not args.no_display:
        cv2.namedWindow("HRNet Pose Unificado", cv2.WINDOW_NORMAL)

    last_id, frame_idx = -1, 0
    perf = perf_fx = None
    if args.perf_log:
        perf = PerfLogger(args.perf_log, meta={"view": args.view, "light": args.light})
        perf_fx = FeatureExtractor(args.kpt_thr)
    kpts = conf = roi = feet = None
    infer_ms, loop_fps = 0.0, 0.0
    prev_t = time.perf_counter()
    last_print = last_dbg = prev_t

    try:
        while True:
            frame, last_id = cam.read(last_id)
            if frame is None:
                if not cam.alive:
                    print("[CAM] Camara detenida.")
                    break
                continue

            fh, fw = frame.shape[:2]
            if roi is None:
                roi = tracker.initial(fw, fh)

            run_inf = (not router.menu) and (router.input_mode is None) and (frame_idx % max(1, args.skip) == 0)
            if run_inf:
                pre.run(frame, roi, model.input_view)
                t0 = time.perf_counter()
                out = model.infer(nk)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                infer_ms = dt_ms if infer_ms == 0.0 else 0.8 * infer_ms + 0.2 * dt_ms

                hm = out.reshape(K, hm_h, hm_w)[:nk]
                kpts, conf = decode_heatmaps(hm, roi)
                kpts = kfilter(kpts, time.perf_counter())
                t_inf = time.perf_counter()

                if args.debug and (time.perf_counter() - last_dbg) >= 1.0:
                    hmf = hm[:nk].astype(np.float32)
                    msg = ("[DBG] mode=%s roi=(%.0f,%.0f,%.0f,%.0f) conf max=%.3f mean=%.3f "
                           "n>thr=%d | hm min=%.3f max=%.3f"
                           % ("TRACK" if tracker.tracking else "SEARCH",
                              roi[0], roi[1], roi[2], roi[3],
                              float(conf.max()), float(conf.mean()),
                              int((conf > args.kpt_thr).sum()),
                              float(np.nanmin(hmf)) if np.isfinite(hmf).any() else float("nan"),
                              float(np.nanmax(hmf)) if np.isfinite(hmf).any() else float("nan")))
                    print(msg)
                    last_dbg = time.perf_counter()

                roi = tracker.update(kpts, conf, fw, fh)
                if perf is not None:
                    perf.sample(kpts, conf, tracker.tracking, args.kpt_thr,
                                perf_fx.update(kpts, conf, t_inf) if tracker.tracking else None)
                if not tracker.tracking:
                    kfilter.reset()
                    foot_mon.reset()
                    feet = None
                    router.lost(t_inf)
                    if framedb is not None:
                        framedb.log_frame(router.snapshot(t_inf), frame_idx, infer_ms, False)
                else:
                    if use_feet:
                        feet = foot_mon.update(kpts, conf)
                    router.step(kpts, conf, t_inf)
                    if framedb is not None:
                        framedb.log_frame(router.snapshot(t_inf), frame_idx, infer_ms, True, kpts, conf)
                    if pending_post_rep:
                        item_post = pending_post_rep.pop(0)
                        act, amsg, acol = item_post[0], item_post[1], item_post[2]
                        router.notify(amsg, t_inf, duration=6.0, col=acol)
                        if act == "close":
                            router.session_finished_auto = True
                            break
                        elif act == "next" and len(item_post) > 3 and item_post[3]:
                            router.set_mode(item_post[3], t_inf, check_plan=False)
                        elif act == "menu":
                            router.set_mode("menu", t_inf, check_plan=False)
            frame_idx += 1

            now = time.perf_counter()
            inst = 1.0 / max(now - prev_t, 1e-6)
            loop_fps = inst if loop_fps == 0.0 else 0.9 * loop_fps + 0.1 * inst
            prev_t = now
            if perf is not None:
                perf.tick(cam.fps, loop_fps, infer_ms)

            if args.no_display:
                if now - last_print >= 1.0:
                    ctxt = ""
                    if router.counter is not None:
                        ctxt = " | %s [%s] OK=%d FALLIDAS=%d" % (
                            router.counter.TITLE, router.counter.state,
                            router.counter.ok, router.counter.fail)
                    print("Camera FPS: %.1f | Loop FPS: %.1f | infer %.1f ms | %s kpts=%d%s"
                          % (cam.fps, loop_fps, infer_ms,
                             "TRACK" if tracker.tracking else "SEARCH", tracker.n_good, ctxt))
                    last_print = now
                continue

            # ---- Visualizacion en pantalla ----
            if kpts is not None and (tracker.tracking or args.draw_always or args.no_track):
                draw_pose(frame, kpts, conf, args.kpt_thr)
                if feet is not None:
                    draw_foot_status(frame, feet, kpts)
            if args.show_roi and roi is not None:
                x0, y0, rw, rh = roi
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x0 + rw), int(y0 + rh)),
                              (255, 255, 0), 1)

            # 1) FPS + infer_ms + TRACK/SEARCH + kpts en la parte inferior izquierda
            put_hud(frame, cam.fps, loop_fps, infer_ms, tracker.tracking, tracker.n_good)
            # 2) Barra de controles en el borde inferior
            draw_controls(frame)

            tnow = time.perf_counter()
            if router.menu:
                draw_menu(frame, router, tnow, cur_patient)
            else:
                y_after_hdr = draw_plan_header(frame, router, tnow, cur_patient)
                if router.counter is not None:
                    if isinstance(router.counter, HeelToeCounter):
                        draw_exercise_status(
                            frame, router.counter.last_res, router.counter.TITLE,
                            org=(10, y_after_hdr + 14)
                        )
                    else:
                        draw_rep_hud(
                            frame, router.counter, tnow,
                            y0=y_after_hdr + 14, planes=router.planes
                        )
                draw_patient_input_modal(frame, router)

            cv2.imshow("HRNet Pose Unificado", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            # ---- Entrada interactiva de Nombre / Edad ([I] / [E]) tanto en Home como en Camara ----
            if router.input_mode is not None:
                if key == 27:   # ESC -> cancelar
                    router.input_mode = None
                    router.input_buf = ""
                    router.notify("Usando paciente actual: %s" % cur_patient, tnow, 2.5, (180, 180, 180))
                elif key == 9 and router.input_mode == "name":   # TAB -> autocompletar
                    pts = proto_mgr.patient_list()
                    if pts:
                        cur_b = proto_mgr.clean_name(router.input_buf) if router.input_buf else ""
                        idx_t = (pts.index(cur_b) + 1) % len(pts) if cur_b in pts else 0
                        router.input_buf = pts[idx_t]
                elif key in (8, 127):   # Backspace
                    router.input_buf = router.input_buf[:-1]
                elif key in (10, 13, 141):   # Enter
                    val = router.input_buf.strip()
                    if router.input_mode == "name":
                        if val:
                            cname = proto_mgr.clean_name(val)
                            if cname in proto_mgr.data.get("pacientes", {}):
                                router.input_mode = None
                                router.input_buf = ""
                                cur_patient = switch_active_patient(cname, None, tnow)
                            else:
                                router.pending_name = cname
                                router.input_mode = "age"
                                router.input_buf = ""
                        else:
                            router.input_mode = None
                    elif router.input_mode == "age":
                        try:
                            age_v = max(10, min(120, int(val))) if val else 65
                        except Exception:
                            age_v = 65
                        target_name = router.pending_name or cur_patient
                        router.input_mode = None
                        router.input_buf = ""
                        cur_patient = switch_active_patient(target_name, age_v, tnow)
                else:
                    ch_in = ""
                    if 32 <= key < 127:
                        ch_in = chr(key)
                    elif 176 <= key <= 185:
                        ch_in = chr(key - 128)
                    if ch_in:
                        if router.input_mode == "name":
                            if ch_in.isalnum() or ch_in in ("_", "-", " "):
                                router.input_buf += ("_" if ch_in == " " else ch_in)
                        elif router.input_mode == "age":
                            if ch_in.isdigit() and len(router.input_buf) < 3:
                                router.input_buf += ch_in
                continue

            if key in (ord("q"), ord("Q")):
                break

            ch = chr(key).lower() if key < 128 else ""
            if ch == "i":
                router.input_mode = "name"
                router.input_buf = ""
                router.pending_name = ""
            elif ch == "e":
                router.pending_name = cur_patient
                router.input_mode = "age"
                router.input_buf = ""
            elif ch == "d":
                new_d = proto_mgr.cycle_override_day()
                st_d = proto_mgr.get_daily_status(cur_patient)
                if framedb is not None:
                    framedb.day_name = st_d["dia_semana"]
                d_msg = ("Dia simulado: %s -> Estado: %s"
                         % (st_d["dia_label"].upper(), st_d["estado_dia"])) if new_d else (
                         "Dia real del sistema (%s) -> Estado: %s"
                         % (st_d["dia_label"].upper(), st_d["estado_dia"]))
                router.notify(d_msg, tnow, 4.0, (0, 220, 255))
            elif ch == "f":
                proto_mgr.force_today(cur_patient)
                router.notify("Sesion autorizada hoy para %s." % cur_patient,
                              tnow, 3.5, (0, 255, 128))
            elif ch == "n":
                nxt_p = proto_mgr.step_patient(cur_patient, 1)
                cur_patient = switch_active_patient(nxt_p, None, tnow)
            elif ch == "b":
                prv_p = proto_mgr.step_patient(cur_patient, -1)
                cur_patient = switch_active_patient(prv_p, None, tnow)
            elif ch == "s":
                router.set_mode("sts", tnow)
            elif ch == "a":
                router.set_mode("abd", tnow)
            elif ch == "h":
                router.set_mode("talones", tnow)
            elif ch == "t":
                router.set_mode("puntas", tnow)
            elif ch == "g":
                router.set_mode("auto", tnow)
            elif ch == "m":
                router.set_mode("menu", tnow, check_plan=False)
            elif ch == "p":
                if router.menu:
                    if router.can_start_exercise("plan", tnow):
                        nxt_ej = proto_mgr.next_pending_exercise(cur_patient)
                        if nxt_ej:
                            router.set_mode(nxt_ej, tnow)
                else:
                    router.paused = not router.paused
                    router.notify("EJERCICIO PAUSADO" if router.paused else "EJERCICIO REANUDADO",
                                  tnow, 2.5, (0, 200, 255))
            elif ch == "r":
                if router.menu:
                    proto_mgr.reset_today(cur_patient)
                    router.reset_counts()
                    router.notify("Progreso de hoy reiniciado para %s." % cur_patient,
                                  tnow, 3.5, (0, 220, 255))
                else:
                    if router.counter is not None:
                        router.counter.reset_counts()
                        router.notify("Contador de %s reiniciado." % router.counter.TITLE,
                                      tnow, 2.5, (0, 220, 255))
            elif key == 32 and router.rest_until > tnow:   # Espacio -> saltar descanso
                router.rest_until = 0.0
                router.notify("Descanso terminado. Continua con la siguiente serie!",
                              tnow, 2.5, (0, 255, 128))
            elif ch == "k":
                if logger is not None:
                    n_ser = logger.next_series()
                    router.notify("Nueva serie de validacion: #%d (cuenta manual aparte)" % n_ser,
                                  tnow, 3.0, (0, 255, 255))
            elif collector is not None and key in DataCollector.KEYS:
                collector.set_label(DataCollector.KEYS[key])
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if perf is not None:
            perf.close()
        router.close()
        if framedb is not None:
            framedb.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
