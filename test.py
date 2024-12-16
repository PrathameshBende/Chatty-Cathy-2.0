import cv2
import mediapipe as mp
import serial
import time

# Initialize serial port
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def main():
    cap = cv2.VideoCapture(0)
    last_time_sent = time.time()
    send_interval = 0.1  # Send data every 0.5 seconds

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Assume using landmarks to find a point to send
                x = int(face_landmarks.landmark[1].x * 640)  # Example: landmark 1
                y = int(face_landmarks.landmark[1].y * 480)

                # Only send coordinates at intervals
                current_time = time.time()
                if current_time - last_time_sent > send_interval:
                    coord_str = f"{x},{y}\n"
                    ser.write(coord_str.encode())
                    last_time_sent = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
