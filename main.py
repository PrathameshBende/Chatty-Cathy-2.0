import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mouth_aspect_ratio_3d(landmarks):
    upper_lip = np.array([landmarks[13], landmarks[14], landmarks[15]])
    lower_lip = np.array([landmarks[308], landmarks[317], landmarks[318]])

    A = np.linalg.norm(upper_lip[1] - lower_lip[1])

    left_corner = landmarks[61]
    right_corner = landmarks[291]
    B = np.linalg.norm(left_corner - right_corner)

    mar = A / B
    return mar

def create_trackbars():
    cv2.namedWindow('Parameters')
    cv2.createTrackbar('MAR_THRESHOLD', 'Parameters', 120, 200, lambda x: None)
    cv2.createTrackbar('DIFFERENCE_THRESHOLD', 'Parameters', 2, 100, lambda x: None)
    cv2.createTrackbar('CONSECUTIVE_FRAMES', 'Parameters', 6, 20, lambda x: None)
    cv2.createTrackbar('SMOOTHING_WINDOW', 'Parameters', 7, 20, lambda x: None)

    cv2.resizeWindow('Parameters', 500, 100)

def get_trackbar_values():
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
