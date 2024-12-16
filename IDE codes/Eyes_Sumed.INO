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

// Blink-specific settings for four servos
int servoAngles[4][2] = {
  {130, 90},  // Servo 0 (Channel 0)
  {90, 150},  // Servo 1 (Channel 1)
  {90, 120},  // Servo 2 (Channel 2)
  {150, 90}   // Servo 3 (Channel 3)
};
int servoChannels[4] = {0, 1, 2, 3}; // Servo channels
#define BLINK_MIN_INTERVAL 1000 // Minimum time between blinks (ms)
#define BLINK_MAX_INTERVAL 5000 // Maximum time between blinks (ms)
#define BLINK_DURATION 200      // Duration of a blink (ms)

// Timing variables for blinking
unsigned long lastBlinkTime = 0;     // Last time a blink happened
unsigned long nextBlinkTime = random(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL); // Randomly decide next blink time

// Map angle to PCA9685 pulse length
int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Move servos to a specific angle
void moveServosToAngle(bool toStartAngle) {
  for (int i = 0; i < 4; i++) {
    int angle = toStartAngle ? servoAngles[i][0] : servoAngles[i][1];
    pwm.setPWM(servoChannels[i], 0, angleToPulse(angle));
    Serial.print("Servo ");
    Serial.print(i);
    Serial.print(" (Channel ");
    Serial.print(servoChannels[i]);
    Serial.print(") moved to ");
    Serial.println(angle);
  }
}

void setup() {
  Serial.begin(115200);  // Start serial communication for debugging
  Wire.begin();          // Initialize I2C
  pwm.begin();
  pwm.setPWMFreq(50);    // Set frequency to 50 Hz for servos
  // Initialize servos to the "open" state
  moveServosToAngle(true);
}

void loop() {
  // Get current time
  unsigned long currentTime = millis();

  // Check if it's time for a blink
  if (currentTime - lastBlinkTime >= nextBlinkTime) {
    // Perform a blink: close then immediately open
    moveServosToAngle(false); // Close
    delay(BLINK_DURATION);    // Wait for blink duration
    moveServosToAngle(true);  // Open again

    // Update timing for next blink
    lastBlinkTime = currentTime;
    nextBlinkTime = random(BLINK_MIN_INTERVAL, BLINK_MAX_INTERVAL);
  }

  // Check for serial data to move X/Y servos
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // Read until newline
    int commaIndex = data.indexOf(',');         // Find the comma separator

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
      pwm.setPWM(4, 0, x_pulse); // Servo for X-coordinate
      pwm.setPWM(5, 0, y_pulse); // Servo for Y-coordinate
  }
}
}
