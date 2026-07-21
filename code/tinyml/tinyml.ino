/*
==========================================================
TFM
ESP32 + BME280 + TinyML + Deep Sleep
Parte 1
==========================================================
*/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "modelo.h"

//--------------------------------------------------
// Pines
//--------------------------------------------------

#define SDA_PIN 23
#define SCL_PIN 22

#define BME280_ADDRESS 0x76

//--------------------------------------------------
// Tiempo de Deep Sleep
//--------------------------------------------------

//#define TIEMPO_SLEEP (5ULL * 1000000ULL)   // 5 segundos
#define TIEMPO_SLEEP_MIN 30
#define TIEMPO_SLEEP (TIEMPO_SLEEP_MIN * 60ULL * 1000000ULL)
//--------------------------------------------------
// Umbral del modelo
//--------------------------------------------------

#define UMBRAL 0.312f

//--------------------------------------------------
// Sensor
//--------------------------------------------------

Adafruit_BME280 bme;

//--------------------------------------------------
// TensorFlow Lite
//--------------------------------------------------

namespace {

tflite::ErrorReporter* error_reporter = nullptr;

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;

TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;

uint8_t tensor_arena[kTensorArenaSize];

}

//--------------------------------------------------
// StandardScaler
//--------------------------------------------------

const float mean[6] = {

25.0624371585f,
71.2158360656f,
894.1497704918f,
-0.0035956284f,
0.0283934426f,
0.0045792350f

};

const float scale[6] = {

0.7515401055f,
4.9999482003f,
1.7863470837f,
0.2527887526f,
2.5119532248f,
0.4660055970f

};

//--------------------------------------------------
// Variables
//--------------------------------------------------

float temperatura;
float humedad;
float presion;

float deltaTemp;
float deltaHum;
float deltaPres;

//--------------------------------------------------
// Memoria RTC
//--------------------------------------------------

RTC_DATA_ATTR bool primeraMedicion = true;

RTC_DATA_ATTR float tempAnterior = 0;

RTC_DATA_ATTR float humAnterior = 0;

RTC_DATA_ATTR float presAnterior = 0;

//--------------------------------------------------
// Prototipos
//--------------------------------------------------

bool inicializarSensor();

bool inicializarTinyML();

void leerSensor();

void calcularDeltas();

float ejecutarTinyML();

void guardarLecturaRTC();

void dormir();

//--------------------------------------------------

void setup() {

Serial.begin(115200);

delay(500);

Serial.println();
Serial.println("==============================");
Serial.println("ESP32 TinyML");
Serial.println("==============================");

//--------------------------------
// Inicializar BME280
//--------------------------------

if (!inicializarSensor()) {

Serial.println("Error inicializando BME280");

while (1);

}

//--------------------------------
// Inicializar TinyML
//--------------------------------

if (!inicializarTinyML()) {

Serial.println("Error inicializando TinyML");

while (1);

}

//--------------------------------
// Leer sensor
//--------------------------------

leerSensor();

//--------------------------------
// Primer arranque
//--------------------------------

if (primeraMedicion) {

Serial.println();
Serial.println("Primera medicion.");

guardarLecturaRTC();

primeraMedicion = false;

dormir();

}

//--------------------------------
// Calcular deltas
//--------------------------------

calcularDeltas();

//--------------------------------
// Ejecutar modelo
//--------------------------------

float probabilidad = ejecutarTinyML();

Serial.println();

Serial.print("Probabilidad: ");

Serial.println(probabilidad,6);

//--------------------------------
// Decision
//--------------------------------

if(probabilidad>=UMBRAL){

Serial.println();
Serial.println(">>> TRANSMITIR <<<");

}
else{

Serial.println();
Serial.println(">>> NO TRANSMITIR <<<");

}

//--------------------------------
// Guardar ultima lectura
//--------------------------------

guardarLecturaRTC();

//--------------------------------

dormir();

}

void loop(){

}

//========================================================
// Inicializar BME280
//========================================================

bool inicializarSensor() {

    Wire.begin(SDA_PIN, SCL_PIN);

    if (!bme.begin(BME280_ADDRESS))
        return false;

    bme.setSampling(
        Adafruit_BME280::MODE_NORMAL,
        Adafruit_BME280::SAMPLING_X16,
        Adafruit_BME280::SAMPLING_X16,
        Adafruit_BME280::SAMPLING_X16,
        Adafruit_BME280::FILTER_X16,
        Adafruit_BME280::STANDBY_MS_500
    );

    return true;
}

//========================================================
// Inicializar TensorFlow Lite
//========================================================

bool inicializarTinyML() {

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(modelo_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION)
        return false;

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        error_reporter);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
        return false;

    input = interpreter->input(0);
    output = interpreter->output(0);

    return true;
}

//========================================================
// Leer sensor
//========================================================

void leerSensor() {

    temperatura = bme.readTemperature();

    humedad = bme.readHumidity();

    presion = bme.readPressure() / 100.0F;

    Serial.println();

    Serial.print("Temperatura : ");
    Serial.println(temperatura, 2);

    Serial.print("Humedad     : ");
    Serial.println(humedad, 2);

    Serial.print("Presion     : ");
    Serial.println(presion, 2);
}

//========================================================
// Calcular deltas
//========================================================

void calcularDeltas() {

    deltaTemp = temperatura - tempAnterior;

    deltaHum = humedad - humAnterior;

    deltaPres = presion - presAnterior;

    Serial.println();

    Serial.print("Delta Temp : ");
    Serial.println(deltaTemp, 3);

    Serial.print("Delta Hum  : ");
    Serial.println(deltaHum, 3);

    Serial.print("Delta Pres : ");
    Serial.println(deltaPres, 3);
}

//========================================================
// Ejecutar modelo
//========================================================

float ejecutarTinyML() {

    float entrada[6] = {

        temperatura,
        humedad,
        presion,
        deltaTemp,
        deltaHum,
        deltaPres

    };

    //----------------------------------------------------
    // Normalizar
    //----------------------------------------------------

    for (int i = 0; i < 6; i++) {

        input->data.f[i] =
            (entrada[i] - mean[i]) / scale[i];

    }

    //----------------------------------------------------
    // Inferencia
    //----------------------------------------------------

    if (interpreter->Invoke() != kTfLiteOk) {

        Serial.println("Error en la inferencia");

        return -1.0;

    }

    return output->data.f[0];
}

//========================================================
// Guardar lectura en RTC
//========================================================

void guardarLecturaRTC() {

    tempAnterior = temperatura;

    humAnterior = humedad;

    presAnterior = presion;

}

//========================================================
// Deep Sleep
//========================================================

void dormir() {

    Serial.println();

    Serial.println("Entrando en Deep Sleep...");

    Serial.flush();

    esp_sleep_enable_timer_wakeup(TIEMPO_SLEEP);

    esp_deep_sleep_start();

}