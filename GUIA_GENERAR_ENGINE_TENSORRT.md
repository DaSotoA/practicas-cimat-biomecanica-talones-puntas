# Guía Paso a Paso: Cómo Generar el Motor TensorRT (`.engine`) en NVIDIA Jetson Nano

Esta guía documenta el procedimiento completo para compilar el archivo optimizado **`hrnet_wb_fp16.engine`** a partir del modelo ONNX de **HRNet WholeBody** utilizando la herramienta nativa **`trtexec`** de NVIDIA en una placa **NVIDIA Jetson Nano (4 GB)**.

---

## 📌 ¿Por qué se debe compilar en la propia Jetson?

TensorRT no utiliza un formato genérico de modelo. Al compilar un `.engine`, TensorRT perfila y optimiza los operadores matemáticos específicamente para los **128 núcleos de arquitectura NVIDIA Maxwell** y la versión de CUDA instalada en tu Jetson. Un archivo `.engine` generado en otra tarjeta gráfica o en una PC con GPU RTX/GTX **no funcionará en la Jetson Nano**.

---

## 1. Requisitos Previos

- **Hardware:** Tarjeta NVIDIA Jetson Nano Developer Kit (versión de 4 GB de RAM).
- **Sistema Operativo:** NVIDIA JetPack 4.6 / 4.6.1 (Ubuntu 18.04 LTS).
- **Software Preinstalado por JetPack:**
  - CUDA Toolkit 10.2
  - TensorRT 8.0 o 8.2 (incluye la utilidad `/usr/src/tensorrt/bin/trtexec`)
  - cuDNN 8.2+
- **Almacenamiento:** Tarjeta microSD (Clase 10 / U3) con al menos 8 GB de espacio libre.

---

## 2. Preparar el Entorno en la Jetson Nano

### Paso 2.1: Activar Memoria Swap (Imprescindible)
La fase de optimización de capas de TensorRT consume más de los 4 GB de memoria física de la placa. Sin memoria swap configurada, el sistema operativo cerrará el proceso con el mensaje `Killed` (Out of Memory).

Ejecuta en la terminal de la Jetson:

```bash
# Crear un archivo de swap de 4 GB si no lo tienes habilitado
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verificar que el swap esté activo
free -h
```

### Paso 2.2: Poner la Jetson en Modo de Máxima Potencia (10W)
Asegura que la GPU y CPU funcionen a su frecuencia de reloj más alta durante la compilación:

```bash
# Modo 10W (todos los núcleos activos)
sudo nvpmodel -m 0

# Fijar frecuencias máximas de reloj del ventilador y procesador
sudo jetson_clocks
```

---

## 3. Obtener el Modelo Base en Formato ONNX

El pipeline requiere el modelo **HRNet WholeBody** con los 133 puntos anatómicos (cuerpo, pies, manos y cara).

1. El checkpoint oficial pertenece al ecosistema **MMPose / OpenMMLab** (*Whole-Body Human Pose Estimation*):
   - Arquitectura: `HRNet-W48` (o `HRNet-W32`) entrenado en el conjunto COCO-WholeBody.
   - Resolución de entrada estándar: `192x256` (ancho $\times$ alto) o `256x192`.
2. Una vez exportado a ONNX (o transferido a la Jetson), renómbralo o colócalo en tu directorio de trabajo con el nombre:
   ```bash
   hrnet_w48_coco_wholebody.onnx
   ```

---

## 4. Compilación con `trtexec` (Paso Principal)

La herramienta `trtexec` viene incluida en el sistema operativo JetPack en la ruta `/usr/src/tensorrt/bin/trtexec`.

Ejecuta el siguiente comando en la terminal para iniciar la compilación y perfilado en precisión media (**FP16**):

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=hrnet_w48_coco_wholebody.onnx \
  --saveEngine=hrnet_wb_fp16.engine \
  --fp16 \
  --workspace=1024
```

### Parámetros explicados:
- **`--onnx=...`**: Ruta al modelo de entrada en formato ONNX.
- **`--saveEngine=...`**: Nombre del archivo binario serializado que generará (`hrnet_wb_fp16.engine`).
- **`--fp16`**: Habilita la ejecución en coma flotante de 16 bits. Esto reduce a la mitad el consumo de memoria de la GPU y duplica la velocidad de cálculo respecto a FP32.
- **`--workspace=1024`**: Asigna hasta 1024 MB de memoria de trabajo a TensorRT para encontrar las combinaciones de algoritmos más rápidas para cada capa.

> ⏱️ **Tiempo estimado:** La compilación en la Jetson Nano puede tardar entre **10 y 25 minutos**. Durante este tiempo el sistema probará internamente cientos de variantes matemáticas. No interrumpas el proceso hasta que veas en consola el mensaje `[I] Engine built and saved successfully`.

---

## 5. Verificación del Archivo `.engine` Generado

Una vez finalizado el comando anterior, comprueba que el archivo se haya creado correctamente:

```bash
ls -lh hrnet_wb_fp16.engine
```

Puedes realizar una prueba rápida de velocidad de inferencia directamente con `trtexec`:

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=hrnet_wb_fp16.engine
```
Verás un resumen con la latencia media por cuadro (aproximadamente entre 80 ms y 85 ms por inferencia).

---

## 6. Ejecución del Pipeline Unificado

Copia o coloca el archivo `hrnet_wb_fp16.engine` generado en la misma carpeta del repositorio donde se encuentra `hrnet_jetson_pipeline_unificado.py` y `exercise_clf.joblib`:

```bash
# Ejecución en tiempo real con cámara USB UVC
python3 hrnet_jetson_pipeline_unificado.py \
  --engine hrnet_wb_fp16.engine \
  --camera 0 \
  --skip 2

# Ejecución sobre un video de evaluación
python3 hrnet_jetson_pipeline_unificado.py \
  --engine hrnet_wb_fp16.engine \
  --video mi_evaluacion.mp4
```

---

## 7. Solución de Problemas Frecuentes (Troubleshooting)

### Error 1: El comando termina diciendo `Killed`
- **Causa:** La memoria RAM física (4 GB) se saturó durante la fase de optimización de capas.
- **Solución:** Revisa el Paso 2.1 y asegúrate de que el swapfile de al menos 4 GB esté activo antes de compilar.

### Error 2: `[E] [TRT] Engine built with different CUDA/TensorRT version`
- **Causa:** El archivo `.engine` fue copiado de otra computadora o generado con otra versión de JetPack.
- **Solución:** Vuelve a compilar el engine directamente en la Jetson siguiendo los pasos de esta guía.

### Error 3: La tasa de cuadros (FPS) es inferior a 8 FPS
- **Causa:** La Jetson se encuentra en modo bajo consumo (5W) o la cámara satura el hilo de inferencia.
- **Solución:** Ejecuta `sudo nvpmodel -m 0` y añade el flag `--skip 2` en el pipeline para procesar de forma alterna 1 de cada 2 cuadros manteniendo seguimiento cinemático continuo.
