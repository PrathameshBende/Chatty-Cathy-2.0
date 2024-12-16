import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Setup serial connection
ser = serial.Serial('COM6', 115200, timeout=1)  # Adjusted baud rate
time.sleep(2)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def send_coordinates(x, y):
    coord_str = f"{x},{y}\n"
    ser.write(coord_str.encode())

def get_eye_center(landmarks, eye_indices):
    eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    eye_center = np.mean(eye_points, axis=0)
    return int(eye_center[0] * 640), int(eye_center[1] * 480)

def mouth_aspect_ratio_3d(landmarks):
    # Calculate the mouth aspect ratio to determine if speaking or not
    upper_lip = np.array([landmarks[13], landmarks[14], landmarks[15]])
    lower_lip = np.array([landmarks[308], landmarks[317], landmarks[318]])
    A = np.linalg.norm(upper_lip[1] - lower_lip[1])
    B = np.linalg.norm(landmarks[61] - landmarks[291])
    return A/B

def create_trackbars():
    # Create trackbars for real-time parameter adjustments
    cv2.namedWindow('Parameters')
    cv2.createTrackbar('MAR_THRESHOLD', 'Parameters', 120, 200, lambda x: None)
    cv2.createTrackbar('DIFFERENCE_THRESHOLD', 'Parameters', 2, 100, lambda x: None)
    cv2.createTrackbar('CONSECUTIVE_FRAMES', 'Parameters', 6, 20, lambda x: None)
    cv2.createTrackbar('SMOOTHING_WINDOW', 'Parameters', 7, 20, lambda x: None)
    cv2.resizeWindow('Parameters', 500, 100)

def get_trackbar_values():
    # Read values from trackbars
    mar_threshold = cv2.getTrackbarPos('MAR_THRESHOLD', 'Parameters') / 1000.0
    difference_threshold = cv2.getTrackbarPos('DIFFERENCE_THRESHOLD', 'Parameters') / 10000.0
    consecutive_frames = cv2.getTrackbarPos('CONSECUTIVE_FRAMES', 'Parameters')
    smoothing_window = cv2.getTrackbarPos('SMOOTHING_WINDOW', 'Parameters')
    return mar_threshold, difference_threshold, consecutive_frames, smoothing_window

def main():
    cap = cv2.VideoCapture(0)
    create_trackbars()
    mar_history = []
    previous_mar = 0
    speaking_frames = 0
    not_speaking_frames = 0
    speaking_state = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        mar_threshold, difference_threshold, consecutive_frames, smoothing_window = get_trackbar_values()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark])
                current_mar = mouth_aspect_ratio_3d(landmarks)
                mar_history.append(current_mar)
                if len(mar_history) > smoothing_window:
                    mar_history.pop(0)
                smoothed_mar = np.mean(mar_history)
                mar_change = abs(current_mar - previous_mar)
                if smoothed_mar > mar_threshold and mar_change > difference_threshold:
                    speaking_frames += 1
                    not_speaking_frames = 0
                else:
                    not_speaking_frames += 1
                    speaking_frames = 0
                if speaking_frames > consecutive_frames:
                    speaking_state = True
                elif not_speaking_frames > consecutive_frames // 2:
                    speaking_state = False
                if speaking_state:
                    cv2.putText(frame, "Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Not Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                previous_mar = current_mar
                left_eye_indices = [33, 133, 159, 158, 157] 
                right_eye_indices = [362, 463, 387, 386, 385]
                left_eye_center = get_eye_center(face_landmarks.landmark, left_eye_indices)
                right_eye_center = get_eye_center(face_landmarks.landmark, right_eye_indices)
                eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                              (left_eye_center[1] + right_eye_center[1]) // 2)
                print(f"Eye Center: {eye_center}")
                send_coordinates(640-eye_center[0], 480-eye_center[1])
                time.sleep(0.01)  # Control the rate of sending data
                cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)
                cv2.circle(frame, eye_center, 5, (255, 0, 0), -1)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        cv2.putText(frame, f"MAR_THRESHOLD: {mar_threshold:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"DIFFERENCE_THRESHOLD: {difference_threshold:.4f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CONSECUTIVE_FRAMES: {consecutive_frames}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"SMOOTHING_WINDOW: {smoothing_window}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Mouth Movement Detection (Calibration)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
