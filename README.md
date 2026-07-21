# TinyML para envío inteligente de datos con BME280

Este repositorio contiene un proyecto de TinyML orientado a la lectura de un sensor BME280 en una placa ESP32, con el objetivo de monitorear temperatura, humedad y presión, y de decidir de forma inteligente si una medición debe transmitirse o no.

La idea central del sistema es reducir el consumo energético y el volumen de datos enviados, utilizando un modelo de inferencia en el microcontrolador para determinar si una lectura es relevante. El enfoque está pensado para alcanzar una reducción aproximada del 80% en la transmisión de datos en comparación con un esquema de envío continuo.

## Objetivo del proyecto

- Leer datos ambientales con un sensor BME280.
- Comparar cada nueva medición con la anterior mediante diferencias o deltas.
- Ejecutar un modelo TensorFlow Lite en el dispositivo embebido.
- Transmitir únicamente cuando la probabilidad de relevancia supera un umbral predefinido.
- Reducir consumo energético mediante el uso de deep sleep y memoria RTC.

## Descripción funcional

El sistema opera de la siguiente manera:

1. Se inicializa el sensor BME280 y el modelo TinyML.
2. Se leen temperatura, humedad y presión.
3. Se calculan los cambios respecto a la última medición guardada.
4. Se normalizan los datos y se ejecuta la inferencia del modelo.
5. Si la salida del modelo supera el umbral configurado, se considera que la lectura debe transmitirse.
6. En caso contrario, la lectura se descarta y el dispositivo entra en modo de bajo consumo.

## Componentes principales

### Firmware

El firmware principal se encuentra en la carpeta [code/tinyml](code/tinyml). Allí se implementa:

- Lectura del sensor BME280.
- Configuración de comunicación I2C.
- Inferencia con TensorFlow Lite para ESP32.
- Gestión de deep sleep.
- Uso de memoria RTC para conservar valores entre ciclos de wake-up.

### Datos y entrenamiento

La carpeta [datasets](datasets) contiene:

- Conjuntos de datos utilizados para entrenamiento y evaluación.

### Modelos entrenados

La carpeta [models](models) incluye:

- Archivo del modelo en formato TensorFlow Lite.
- Modelo en formato Keras.
- Archivo de escalado utilizado para normalizar las entradas.

## Estructura del repositorio

- [code/tinyml](code/tinyml): código fuente del firmware para ESP32.
- [datasets](datasets): datos de entrenamiento.
- [models](models): modelos entrenados y artefactos de preprocesamiento.
- [README.md](README.md): documentación general del proyecto.

## Tecnologías utilizadas

- ESP32
- Sensor BME280
- TensorFlow Lite para Microcontrollers
- Arduino / ESP32 Arduino Core
- Python para entrenamiento y preparación de datos

## Beneficios esperados

- Ahorro de energía en dispositivos embebidos.
- Menor consumo de ancho de banda y transmisión de datos.
- Implementación viable para monitoreo remoto o IoT con recursos limitados.
- Enfoque práctico para aplicaciones de detección inteligente basada en sensorismo.

## Notas

Este proyecto está enfocado en una solución de bajo consumo para monitoreo ambiental con inferencia en borde, priorizando eficiencia energética y transmisión selectiva de información.
