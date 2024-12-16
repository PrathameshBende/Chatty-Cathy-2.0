  #include <Wire.h>
  #include <Adafruit_PWMServoDriver.h>

  // Create the PCA9685 object
  Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

  // Servo settings
  #define SERVOMIN 150   // Minimum pulse length count (adjust based on your servo)
  #define SERVOMAX 600   // Maximum pulse length count (adjust based on your servo)

  // Servo angle ranges (in degrees)
  int servoAngles[4][2] = {
    {130, 90},  // Servo 0 (Channel 0)
    {90, 150}, // Servo 1 (Channel 4, remapped due to Channel 1 issue)
    {90, 120},  // Servo 2 (Channel 2)
    {150, 90}  // Servo 3 (Channel 3)
  };

  // Corresponding servo channels
  int servoChannels[4] = {2, 3, 4, 5};

  // Map angle to PCA9685 pulse length
  int angleToPulse(int angle) {
    return map(angle, 0, 180, SERVOMIN, SERVOMAX);
  }

  // Move all servos to a specific angle
  void moveAllServosToAngle(bool toStartAngle) {
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
    Serial.begin(9600);  // Start serial communication
    pwm.begin();
    pwm.setPWMFreq(50); // Set frequency to 50 Hz for servos
    Serial.println("Servo tester initialized!");
  }

  void loop() {
    // Move all servos to their start angles
    Serial.println("Moving all servos to start angles...");
    moveAllServosToAngle(true);
    delay(2000); // Wait 2 seconds at the start angles

    // Move all servos to their stop angles
    Serial.println("Moving all servos to stop angles...");
    moveAllServosToAngle(false);
    delay(2000); // Wait 2 seconds at the stop angles
  }
