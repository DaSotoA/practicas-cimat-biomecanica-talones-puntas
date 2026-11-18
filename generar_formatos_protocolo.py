#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera el Protocolo de Adquisicion (A1.3) en .docx:
  Protocolo_Adquisicion_Consentimiento_Ficha.docx
    1. Guion de sesion y definiciones operativas (repeticion valida / compensacion)
    2. Matriz del experimento de geometria de camara (OE1)
    3. Carta de Consentimiento Informado
    4. Ficha de Registro de Sesion (con banda elastica e iluminacion)
    5. Hoja de conteo manual para 2 observadores
Uso: python generar_formatos_protocolo.py [--out archivo.docx]
"""
import argparse

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Protocolo_Adquisicion_Consentimiento_Ficha.docx")
    a = ap.parse_args()
    d = Document()
    d.styles["Normal"].font.name = "Calibri"
    d.styles["Normal"].font.size = Pt(11)

    h = d.add_heading("Protocolo de Adquisicion de Datos", 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = d.add_paragraph("Reconocimiento y Monitoreo Biomecanico de la Elevacion de Talones y Puntas de "
                        "Pie mediante Estimacion de Pose y Conteo Automatico de Repeticiones en una "
                        "Plataforma Embebida")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_paragraph("Practicante: David Guadalupe Soto Aldaba  |  Supervisor: Dr. Hugo Arnoldo Mitre "
                    "Hernandez  |  CIMAT Unidad Zacatecas / UAZ").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1 ------------------------------------------------------------------ #
    d.add_heading("1. Guion de sesion", 1)
    for s in [
        "Colocar la camara a 2.0-2.5 m del voluntario, a la altura de la cadera (~0.9-1.0 m), "
        "con el cuerpo completo (cabeza a pies) dentro del encuadre. Silla sin descansabrazos, "
        "altura de asiento 43-46 cm, contra una pared.",
        "Registrar la ficha de sesion (seccion 4) y verificar que el consentimiento esta firmado.",
        "Ejecutar el pipeline con los metadatos del voluntario, por ejemplo:\n"
        "python3 hrnet_jetson_pipeline_unificado.py --engine hrnet_wb_fp16.engine --cam nvv4l2 "
        "--width 1024 --height 768 --fps 30 --subject S01 --view frontal --light alta "
        "--patient S01 --age 68 --perf-log logs/perf_S01_frontal.csv",
        "Calentamiento: 3 repeticiones de practica sin registrar.",
        "Por cada serie: pulsar [K] (nueva serie de validacion), indicar 'inicio', el voluntario "
        "realiza la serie y dos observadores cuentan repeticiones en silencio y de forma independiente.",
        "Al terminar la serie anotar en la hoja de conteo manual (seccion 5) el numero de serie "
        "mostrado en pantalla/terminal y el conteo de cada observador.",
        "Descanso entre series segun protocolo.json del paciente (60-120 s). Detener ante dolor, "
        "mareo o fatiga excesiva.",
    ]:
        d.add_paragraph(s, style="List Number")

    d.add_heading("Definiciones operativas", 2)
    table(d, ["Ejercicio", "Repeticion VALIDA", "Compensaciones (repeticion FALLIDA)"], [
        ["Sentarse-levantarse (STS)",
         "Pasa de sedente (rodilla < 95 grados) a bipedo completo (rodilla > 165 grados) y regresa, "
         "con control al sentarse.",
         "Inclinacion de tronco > 30 grados, rodilla sin extension completa, caida brusca "
         "(velocidad angular excesiva), uso de las manos."],
        ["Abduccion de cadera (ABD)",
         "Separacion lateral de la pierna >= rango objetivo segun edad (ver protocolos.json) "
         "y regreso a reposo (< 15 grados).",
         "Inclinacion lateral del tronco > 15 grados, rodilla flexionada, movimiento balistico."],
        ["Elevacion de talones", "Talon elevado >= 10 grados respecto al piso y regreso a pie plano.",
         "Elevacion insuficiente, perdida de apoyo, impulso con el tronco."],
        ["Elevacion de puntas", "Antepie elevado >= 6 grados y regreso a pie plano.",
         "Elevacion insuficiente, flexion de rodillas, balanceo."],
    ])

    # 2 ------------------------------------------------------------------ #
    d.add_heading("2. Experimento de geometria de camara (OE1)", 1)
    d.add_paragraph("Con un voluntario en postura neutra (de pie, brazos a los lados, 15 s por vista) "
                    "se graba el rendimiento con --view y --perf-log. El error angular se mide contra "
                    "el valor anatomico esperado (flexion de rodilla = 0 grados, abduccion = 0 grados).")
    table(d, ["Vista", "Angulo de camara", "Comando --view", "Archivo de salida"], [
        ["Frontal", "0 grados", "frontal", "logs/perf_frontal.csv"],
        ["Oblicua 30", "30 grados", "obl30", "logs/perf_obl30.csv"],
        ["Oblicua 45", "45 grados", "obl45", "logs/perf_obl45.csv"],
        ["Lateral", "90 grados", "lateral", "logs/perf_lateral.csv"],
    ])

    # 3 ------------------------------------------------------------------ #
    d.add_page_break()
    d.add_heading("3. Carta de Consentimiento Informado", 1)
    for s in [
        "Titulo del estudio: Reconocimiento y Monitoreo Biomecanico del Ciclo Sentarse-Levantarse "
        "de una Silla y de la Abduccion de Cadera mediante vision por computadora en un dispositivo embebido.",
        "Responsables: Jorge Alamillo Murillo (practicante, UAZ) y Dr. Hugo Arnoldo Mitre Hernandez "
        "(CIMAT Unidad Zacatecas).",
        "Proposito: Evaluar un sistema que cuenta repeticiones y mide angulos de ejercicios de "
        "fortalecimiento de miembros inferiores (sentarse-levantarse, abduccion de cadera, "
        "elevacion de talones y de puntas) usando una camara.",
        "Procedimiento: Realizara series de 8-12 repeticiones de los ejercicios indicados, ajustando la "
        "intensidad mediante el rango de altura/elevacion del pie segun el nivel asignado por edad "
        "(sin bandas elasticas), mientras una camara graba. Duracion aproximada: 30-40 minutos.",
        "Datos que se guardan: Se almacenan unicamente coordenadas de articulaciones (puntos de "
        "esqueleto), angulos y conteos asociados a un codigo (p. ej. S01). NO se guardan imagenes ni "
        "video con su rostro, salvo que usted lo autorice expresamente abajo.",
        "Riesgos: Fatiga muscular leve o molestias transitorias. Puede detenerse en cualquier momento. "
        "Se contara con una silla estable y supervision durante toda la sesion.",
        "Beneficios: No hay beneficio directo; sus datos contribuyen a validar una herramienta para "
        "el fortalecimiento muscular orientado a la prevencion de caidas en personas mayores.",
        "Confidencialidad: Los datos se identifican solo con un codigo. Los resultados se presentaran "
        "de forma agregada. Su participacion es voluntaria y puede retirarse sin ninguna consecuencia, "
        "solicitando la eliminacion de sus datos.",
    ]:
        d.add_paragraph(s)
    d.add_paragraph("[  ] Autorizo ademas el registro de video para fines de anotacion interna "
                    "(se eliminara al terminar el proyecto).")
    d.add_paragraph("[  ] No autorizo el registro de video.")
    table(d, ["Dato", "Informacion"], [
        ["Nombre del participante", ""], ["Codigo asignado (S__)", ""], ["Edad / sexo", ""],
        ["Firma del participante", ""], ["Nombre y firma del investigador", ""], ["Fecha", ""],
    ])

    # 4 ------------------------------------------------------------------ #
    d.add_page_break()
    d.add_heading("4. Ficha de Registro de Sesion", 1)
    table(d, ["Campo", "Valor"], [
        ["Codigo del voluntario (--subject)", "S__"],
        ["Fecha / hora", ""], ["Edad (anos)", ""], ["Sexo (M/F)", ""],
        ["Estatura (cm) / peso (kg)", ""],
        ["Antecedentes (lesiones, cirugias, cirugia de rodilla/cadera, medicamentos)", ""],
        ["Altura de la silla (cm)", "43-46"], ["Distancia camara - voluntario (m)", "2.0-2.5"],
        ["Altura de la camara (m)", ""],
        ["Vista de camara (--view)", "frontal / obl30 / obl45 / lateral"],
        ["Iluminacion (--light)", "alta (>= 300 lux) / baja (~100 lux)"],
        ["Nivel de rango de altura en Abduccion (--age)",
         "Nivel 1 Bajo (75+ anos, 30 deg) / Nivel 2 Medio (60-74 anos, 35-40 deg) / Nivel 3 Alto (<60 anos, 45 deg)"],
        ["Tipo de calzado", ""], ["Comentarios del investigador", ""],
    ])
    d.add_paragraph()
    d.add_paragraph("Condiciones del corpus propio (OE2): >= 12 sesiones x 3 niveles de rango de altura "
                    "del pie en abduccion (bajo/medio/alto segun protocolo de edad) x 2 iluminaciones; "
                    "conteo y segmentacion a cargo de 2 anotadores independientes.")

    # 5 ------------------------------------------------------------------ #
    d.add_heading("5. Hoja de conteo manual (2 observadores)", 1)
    d.add_paragraph("Solo anota cuantas repeticiones conto el Observador 1 y el Observador 2 en cada serie "
                    "de 30 segundos. Despues pasalas a resultados/manual_conteo.csv o con el comando: "
                    "python evaluacion_final_plan.py capturar-manual")
    rows_sheet = []
    for sid in ("D13", "D14", "D15", "D16", "D17"):
        rows_sheet.append([sid, "TALONES (Serie 1)", "1", "", "", "30 s"])
        rows_sheet.append([sid, "PUNTAS (Serie 2)", "2", "", "", "30 s"])
    table(d, ["Sujeto", "Ejercicio", "Serie", "Repeticiones Obs 1", "Repeticiones Obs 2", "Duracion"],
          rows_sheet)
    d.save(a.out)
    print("Guardado:", a.out)


if __name__ == "__main__":
    main()
