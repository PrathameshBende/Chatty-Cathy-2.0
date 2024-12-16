#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create the PCA9685 object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo settings
#define SERVOMIN 150   // Minimum pulse length count (adjust based on your servo)
#define SERVOMAX 600   // Maximum pulse length count (adjust based on your servo)

// Servo ranges (adjust based on your servo and application)
#define X_MIN 0
#define X_MAX 640
#define X_ANGLE_MIN 0
#define X_ANGLE_MAX 120

#define Y_MIN 0
#define Y_MAX 480
#define Y_ANGLE_MIN 60
#define Y_ANGLE_MAX 125

void setup() {
  Serial.begin(115200);  // Start serial communication for debugging
  Wire.begin();  // Initialize I2C
  pwm.begin();
  pwm.setPWMFreq(50); // Set frequency to 50 Hz for servos
  Serial.println("Servo driver initialized!");
}

void loop() {
  // Check if serial data is available
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // Read until newline
    int commaIndex = data.indexOf(',');         // Find the comma separator

    // Parse x and y coordinates from the received string
    if (commaIndex != -1) {
      String x_str = data.substring(0, commaIndex);
      String y_str = data.substring(commaIndex + 1);
      int x = x_str.toInt();
      int y = y_str.toInt();

      // Map x and y to servo angles
      int x_angle = map(x, X_MIN, X_MAX, X_ANGLE_MIN, X_ANGLE_MAX);
      int y_angle = map(y, Y_MIN, Y_MAX, Y_ANGLE_MIN, Y_ANGLE_MAX);

      // Convert angles to pulse lengths
      int x_pulse = map(x_angle, 0, 180, SERVOMIN, SERVOMAX);
      int y_pulse = map(y_angle, 0, 180, SERVOMIN, SERVOMAX);

      // Move the servos
      pwm.setPWM(0, 0, x_pulse); // Servo for X-coordinate
      pwm.setPWM(1, 0, y_pulse); // Servo for Y-coordinate

      // Debug output
      Serial.print("X: ");
      Serial.print(x);
      Serial.print(" -> X Angle: ");
      Serial.println(x_angle);

      Serial.print("Y: ");
      Serial.print(y);
      Serial.print(" -> Y Angle: ");
      Serial.println(y_angle);
    }
  }
}
