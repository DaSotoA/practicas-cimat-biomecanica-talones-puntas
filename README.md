# Reconocimiento y Monitoreo Biomecánico de Elevación de Talones y Puntas en Plataforma Embebida

Proyecto de Prácticas Profesionales de Carrera — **Ingeniería en Computación (UAZ)**  
**Institución Receptora:** Centro de Investigación en Matemáticas, A.C. (CIMAT), Unidad Zacatecas  
**Practicante:** David Guadalupe Soto Aldaba  
**Supervisor Académico:** Dr. Hugo Arnoldo Mitre Hernández  

---

## 📌 Resumen del Proyecto

Este repositorio contiene la implementación completa, modelos, corpus de datos y herramientas de evaluación del sistema embebido para el reconocimiento y conteo automático de repeticiones en ejercicios de rehabilitación del tren inferior:
- **Elevación de talones** (*flexión plantar* — sóleo y gemelos)
- **Elevación de puntas de pie** (*dorsiflexión* — tibial anterior)

El sistema opera en tiempo real sobre una tarjeta **NVIDIA Jetson Nano (4 GB)** con aceleración **TensorRT FP16**, utilizando el estimador de pose **HRNet WholeBody**, que provee los puntos articulares específicos del pie (talón, dedo gordo y dedo pequeño). Además, se encuentra unificado con las pruebas de sentarse-levantarse (STS) y abducción de cadera (ABD).

---

## 🎯 Cumplimiento de Indicadores de Éxito (Tabla 2 del Plan)

| Indicador | Meta del Plan | Resultado Obtenido | Estado |
| :--- | :---: | :---: | :---: |
| **Error Absoluto Medio (MAE)** | $\le 0.50$ rep | **0.25 rep** | ✅ Cumple |
| **Series con error $\le 1$ rep** | $\ge 90.0\%$ | **100.0%** | ✅ Cumple |
| **F1 Macro de clasificación** | $\ge 0.900$ | **0.956** | ✅ Cumple |
| **Concordancia Temporal (ICC)** | $\ge 0.750$ | **0.922** | ✅ Cumple |
| **Tasa sostenida de procesamiento** | $\ge 8.00$ cuadros/s | **8.84 cuadros/s** | ✅ Cumple |
| **Piso de procesamiento (p5)** | $\ge 5.00$ cuadros/s | **6.53 cuadros/s** | ✅ Cumple |
| **Tamaño del clasificador** | $\le 300.0$ KB | **267.0 KB** | ✅ Cumple |
| **Degradación por cuantización/luz** | $\le 2.00$ puntos F1 | **0.00 puntos F1** | ✅ Cumple |

---

## 📂 Estructura del Repositorio

- `hrnet_jetson_pipeline_unificado.py`: Pipeline principal en tiempo real con decodificación de los puntos del pie, calibración de pie plano, máquina de estados `HeelToeCounter` y extractor de 6 características.
- `train_exercise_classifier.py`: Entrenamiento y validación Leave-One-Subject-Out (LOSO) de clasificadores (Random Forest, SVM-RBF, MLP, XGBoost).
- `evaluacion_final_plan.py`: Script para el cálculo riguroso de métricas del plan, gráficos Bland-Altman y matrices de confusión.
- `export_session.py`: Exportación y estructuración de sesiones a JSON Lines.
- `generar_reporte_david.py`: Generador automatizado del reporte técnico final en Word.
- `generar_formatos_protocolo.py`: Generador de formatos de protocolo, consentimiento y justificación clínica.
- `exercise_clf.joblib`: Clasificador desplegado (SVM con kernel RBF, 267 KB).
- `protocolos.json`: Configuración clínica de series, repeticiones y descansos por edad.
- `dashboard_pacientes.html`: Panel clínico web para visualización del progreso del paciente.
- `requirements_jetson.txt`: Dependencias y paquetes requeridos.
- `dataset/`: Corpus de 12 sesiones de prueba (`D01.csv` a `D12.csv`).
- `logs/`: Registros de hardware y validación en vivo fuera de muestra (`reps_D13.csv` a `D17.csv`, `perf_*.csv`, `bd_pacientes.json1`).
- `resultados/`: Gráficas PNG en alta resolución y tablas de resultados.
- Documentos oficiales:
  - `Plan_DavidSoto_Imprimir (1).pdf`
  - `Reporte_Tecnico_Final_Practicas_David_Soto.docx`
  - `Protocolo_Adquisicion_Consentimiento_Ficha.docx`
  - `Justificacion_Protocolos_por_Edad_STS_ABD_Talones_Puntas.docx`

---

## 🚀 Instalación y Uso

### 1. Requisitos
```bash
pip install -r requirements_jetson.txt
```

### 2. Ejecutar el Pipeline en Vivo
```bash
# Con cámara USB UVC en Jetson Nano
python hrnet_jetson_pipeline_unificado.py --engine hrnet_wb_fp16.engine --camera 0 --skip 2

# Con archivo de video
python hrnet_jetson_pipeline_unificado.py --engine hrnet_wb_fp16.engine --video mi_video.mp4
```

### 3. Re-entrenar y Evaluar
```bash
python train_exercise_classifier.py
python evaluacion_final_plan.py
```

### 4. Visualizador Web
Abrir `dashboard_pacientes.html` directamente en el navegador.
