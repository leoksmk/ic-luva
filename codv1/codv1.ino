#define FLEX_PIN 34

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);  // 0 a 4095 no ESP32
}

void loop() {
  int valor = analogRead(FLEX_PIN);
  int nivel = 0;

  if (valor < 100) {        // 800 + 0.2 * (2700 - 800)
    nivel = 1;
  } else if (valor < 200) {
    nivel = 2;
  } else if (valor < 300) {
    nivel = 3;
  } else if (valor < 400) {
    nivel = 4;
  } else {
    nivel = 5;
  }

  Serial.print("Valor bruto: ");
  Serial.print(valor);
  Serial.print("  |  Nível de flexão: ");
  Serial.println(nivel);

  delay(500);
}
