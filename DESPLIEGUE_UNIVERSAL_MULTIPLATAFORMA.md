# Despliegue Universal y Compatibilidad Multiplataforma

Este documento describe cómo adaptar el sistema de monitoreo biomecánico y conteo de repeticiones (desarrollado originalmente para la tarjeta **NVIDIA Jetson Nano**) para ejecutarse en **cualquier computadora o dispositivo**, sin necesidad de contar con hardware embebido específico ni compilar motores `.engine`.

---

## 📌 Contexto: ¿Por qué TensorRT en Jetson y qué cambia en otros equipos?

En el proyecto original se utilizó **TensorRT FP16** debido a las restricciones de hardware de la **Jetson Nano** (arquitectura ARM Cortex-A57 de 2015 con 4 GB de memoria compartida). Sin esa optimización a bajo nivel, la tarjeta no podía alcanzar la meta de tiempo real ($\ge 8$ cuadros por segundo).

Sin embargo, las computadoras portátiles y de escritorio modernas cuentan con procesadores y tarjetas gráficas notablemente más potentes. En estos equipos no es necesario atarse a un archivo `.engine`: se puede utilizar el estándar abierto **ONNX Runtime**, logrando portabilidad inmediata y tasas de procesamiento de entre **15 y 60 FPS**.

---

## 💻 Compatibilidad por Tipo de Dispositivo

| Dispositivo / Plataforma | Motor Recomendado | Proveedor de Ejecución (*Execution Provider*) | Tasa Típica Estimada |
| :--- | :--- | :--- | :---: |
| **PC / Laptop con Windows (Cualquier CPU)** | ONNX Runtime | `CPUExecutionProvider` (Intel / AMD) | 12 – 22 FPS |
| **PC / Laptop con GPU NVIDIA (GTX / RTX)** | ONNX Runtime | `CUDAExecutionProvider` | 30 – 60+ FPS |
| **Computadoras Apple (MacBook M1 / M2 / M3)** | ONNX Runtime | `CoreMLExecutionProvider` | 25 – 45 FPS |
| **Linux (Ubuntu, Debian, Fedora)** | ONNX Runtime | `CPUExecutionProvider` / `CUDAExecutionProvider` | 15 – 60 FPS |
| **Raspberry Pi 4 / 5 (ARM64)** | ONNX Runtime / TFLite | `CPUExecutionProvider` (NEON) | 6 – 12 FPS |
| **NVIDIA Jetson Nano (4 GB)** *(Original)* | TensorRT | `TensorrtExecutionProvider` / `trtexec` | 8 – 10 FPS |

---

## 🚀 Cómo Implementar la Inferencia Universal con ONNX Runtime

El pipeline cinemático de tu proyecto está desacoplado:
1. **La estimación de pose** obtiene las coordenadas $(x, y)$ del cuerpo y los pies.
2. **La lógica biomecánica** (`HeelToeCounter`, `OneEuroFilter`, calibración de pie plano y el clasificador `exercise_clf.joblib`) opera sobre coordenadas numéricas y es **100% independiente del hardware**.

Por lo tanto, solo se necesita cambiar la clase que alimenta la imagen a la red neuronal.

### 1. Instalación de Dependencias Universales
En cualquier computadora con Windows, Linux o Mac:

```bash
# Para computadoras sin tarjeta gráfica dedicada (corre en cualquier procesador):
pip install onnxruntime opencv-python numpy scikit-learn joblib

# O para computadoras con tarjeta gráfica NVIDIA dedicada:
pip install onnxruntime-gpu opencv-python numpy scikit-learn joblib
```

### 2. Clase de Inferencia Universal (`ONNXInference`)
Se puede sustituir o alternar con la clase `TRTInference` mediante un bloque como el siguiente:

```python
import onnxruntime as ort
import numpy as np

class ONNXInference:
    """Motor de inferencia universal compatible con Windows, Mac y Linux."""
    def __init__(self, onnx_model_path):
        # ONNX Runtime selecciona automáticamente la GPU si está disponible; si no, usa el CPU
        available_providers = ort.get_available_providers()
        print(f"[INFO] Proveedores de aceleración detectados: {available_providers}")
        
        self.session = ort.InferenceSession(onnx_model_path, providers=available_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 256, 192]
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, preprocessed_image):
        """Ejecuta inferencia directa sin necesidad de PyCUDA ni asignación manual de memoria."""
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed_image})
        return outputs[0]  # Mapas de calor (heatmaps) de 133 puntos clave
```

---

## 🔄 Adaptación Híbrida del Pipeline (`--engine` vs `--onnx`)

Podemos configurar el argumento `--engine` para que acepte tanto archivos `.engine` como archivos `.onnx`:

```python
parser.add_argument('--model', type=str, required=True,
                    help='Ruta al modelo: .engine (para Jetson TensorRT) o .onnx (para cualquier PC)')

# Lógica de selección automática:
if args.model.endswith('.engine'):
    print("[MODO EMBEBIDO] Iniciando motor TensorRT optimizado para Jetson Nano...")
    model = TRTInference(args.model)
elif args.model.endswith('.onnx'):
    print("[MODO UNIVERSAL] Iniciando motor ONNX Runtime compatible con cualquier equipo...")
    model = ONNXInference(args.model)
```

---

## 📱 Extensiones Futuras: Aplicaciones Móviles y Web

Al disponer del modelo en formato ONNX o exportarlo a **TensorFlow Lite (TFLite)**:

1. **Aplicaciones Móviles (Android / iOS):**
   - El clasificador `exercise_clf.joblib` y la máquina de estados pueden integrarse en una app móvil con MediaPipe o TFLite para que los adultos mayores hagan sus ejercicios en casa usando la cámara de su teléfono inteligente o tableta.
2. **Plataforma Web en el Navegador (Webcam en Vivo):**
   - Utilizando **ONNX Runtime Web (WASM / WebGL)** o **MediaPipe JS**, todo el conteo y la retroalimentación auditiva pueden ejecutarse dentro de una página web en Google Chrome o Edge, sin necesidad de instalar Python en la computadora del paciente.
3. **Servidor Centralizado / Telemedicina:**
   - La cámara local envía únicamente los vectores numéricos de las articulaciones (preservando completamente la privacidad visual del paciente) hacia un servidor clínico central que actualiza el `dashboard_pacientes.html` en tiempo real para el fisioterapeuta.

---

## 🎯 Conclusión

El diseño modular del proyecto separa con éxito la **adquisición sensorial**, la **detección de pose** y la **lógica biomecánica**. Aunque la Jetson Nano fue el hardware de validación del Plan de Prácticas, el núcleo del algoritmo (calibración de línea base, máquina de cuatro estados e invariancia de escala) es universal y puede desplegarse en cualquier computadora convencional con una simple cámara web.
