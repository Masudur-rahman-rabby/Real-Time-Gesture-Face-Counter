import cv2
import mediapipe as mp
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Initialize Mediapipe face and hand models
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class FingerFaceCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Finger & Face Counter")
        self.root.geometry("800x600")
        self.root.configure(bg="#282828")

        self.video_frame = Label(self.root)
        self.video_frame.pack(pady=20)

        self.counts_label = Label(self.root, text="Counts:\nFingers: 0\nFaces: 0", font=("Helvetica", 16), bg="#282828",
                                  fg="white")
        self.counts_label.pack(pady=10)

        self.mode_var = StringVar(value="combined")
        self.radio_combined = ttk.Radiobutton(self.root, text="Combined", variable=self.mode_var, value="combined",
                                              command=self.update_mode)
        self.radio_combined.pack(side=LEFT, padx=10, pady=10)
        self.radio_separate = ttk.Radiobutton(self.root, text="Separate", variable=self.mode_var, value="separate",
                                              command=self.update_mode)
        self.radio_separate.pack(side=LEFT, padx=10, pady=10)

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_video)
        self.start_button.pack(side=LEFT, padx=10, pady=10)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=RIGHT, padx=10, pady=10)

        self.running = False
        self.mode = "combined"  # Initialize mode attribute

    def update_mode(self):
        # Update the mode variable
        self.mode = self.mode_var.get()

    def start_video(self):
        self.running = True
        self.capture = cv2.VideoCapture(0)
        self.update_video()

    def stop_video(self):
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.release()
        self.video_frame.config(image='')

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        results = face_detection.process(rgb_frame)
        if results.detections:
            faces_count = len(results.detections)
        else:
            faces_count = 0

        # Hand detection and counting
        results_hands = hands.process(rgb_frame)
        left_fingers_count = 0
        right_fingers_count = 0

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Determine handedness
                handedness = results_hands.multi_handedness[
                    results_hands.multi_hand_landmarks.index(hand_landmarks)
                ]

                # Count fingers based on handedness
                if handedness.classification[0].label == 'Left':
                    left_fingers_count += self.count_fingers(hand_landmarks.landmark, handedness)
                elif handedness.classification[0].label == 'Right':
                    right_fingers_count += self.count_fingers(hand_landmarks.landmark, handedness)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update counts label based on mode
        if self.mode == "combined":
            total_fingers_count = left_fingers_count + right_fingers_count
            self.counts_label.config(text=f"Counts:\nFingers: {total_fingers_count}\nFaces: {faces_count}")
        else:
            self.counts_label.config(
                text=f"Counts:\nLeft Hand Fingers: {left_fingers_count}\nRight Hand Fingers: {right_fingers_count}\nFaces: {faces_count}")

        # Display video frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

        # Schedule next update
        self.root.after(10, self.update_video)

    def count_fingers(self, hand_landmarks, handedness):
        count = 0
        tips = [4, 8, 12, 16, 20]

        # Thumb detection based on handedness
        if handedness.classification[0].label == 'Right':
            wrist_x = hand_landmarks[0].x
            thumb_tip_x = hand_landmarks[tips[0]].x
            thumb_ip_x = hand_landmarks[tips[0] - 2].x

            if thumb_tip_x < wrist_x and thumb_tip_x < thumb_ip_x:
                count += 1
        elif handedness.classification[0].label == 'Left':
            wrist_x = hand_landmarks[0].x
            thumb_tip_x = hand_landmarks[tips[0]].x
            thumb_ip_x = hand_landmarks[tips[0] - 2].x

            if thumb_tip_x > wrist_x and thumb_tip_x > thumb_ip_x:
                count += 1

        # Other fingers detection
        for tip in tips[1:]:
            if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                count += 1

        return count


def main():
    root = Tk()
    app = FingerFaceCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
