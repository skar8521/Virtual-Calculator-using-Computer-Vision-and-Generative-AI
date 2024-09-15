import os
import cv2
import numpy as np
import PIL
import speech_recognition as sr
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings(action='ignore')


class Calculator:
    def streamlit_config(self):
        # Page configuration
        st.set_page_config(page_title='Calculator', layout="wide")
        
        # Style adjustments
        page_background_color = """
        <style>
        [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        }
        .block-container {
            padding-top: 0rem;
        }
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)
        st.markdown(f'<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
        add_vertical_space(1)

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Set up webcam capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

        # Initialize canvas and hand detector
        self.imgCanvas = np.zeros((550, 950, 3), np.uint8)
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Drawing points and previous time
        self.p1, self.p2 = 0, 0
        self.fingers = []

    def listen_for_commands(self):
        # Initialize the recognizer
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening for commands...")
            audio_data = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio_data)
                st.write("Recognized command: " + command)
                return command
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Error; {e}"

    def process_frame(self):
        success, img = self.cap.read()
        img = cv2.resize(img, (950, 550))
        self.img = cv2.flip(img, 1)
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        result = self.mphands.process(self.imgRGB)
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(self.img, hand_lms, hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = self.img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list:
            for id in [4, 8, 12, 16, 20]:
                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id - 2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id - 2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
            for i in range(5):
                if self.fingers[i]:
                    cx, cy = self.landmark_list[(i + 1) * 4][1], self.landmark_list[(i + 1) * 4][2]
                    cv2.circle(self.img, (cx, cy), 5, (255, 0, 255), 1)

    def handle_drawing_mode(self):
        # Both Thumb and Index Fingers Up in Drawing Mode
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]

            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (255, 0, 255), 5)
            self.p1, self.p2 = cx, cy

        # Thumb, Index & Middle Fingers UP ---> Disable the Points Connection
        elif sum(self.fingers) == 3 and self.fingers[0] == self.fingers[1] == self.fingers[2] == 1:
            self.p1, self.p2 = 0, 0

        # Both Thumb and Middle Fingers Up ---> Erase the Drawing Lines
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]

            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 0, 0), 15)
            self.p1, self.p2 = cx, cy

        # Both Thumb and Pinky Fingers Up ---> Erase the Whole Thing (Reset)
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            self.imgCanvas = np.zeros((550, 950, 3), np.uint8)

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(self.img, 0.7, self.imgCanvas, 1, 0)
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        self.img = cv2.bitwise_or(img, self.imgCanvas)

    def analyze_image_with_genai(self):
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
        imgCanvas = PIL.Image.fromarray(imgCanvas)
        return "Sample analysis of the image."

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            stframe = st.empty()

        with col3:
            st.markdown(f'<h5 style="text-position:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()
            voice_command_button = st.button("Listen for Command")

        while True:
            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown(f'<h4 style="text-position:center; color:orange;">Error: Could not open webcam.</h4>', unsafe_allow_html=True)
                break

            self.process_frame()
            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(self.img, channels="RGB")

            if voice_command_button:
                command = self.listen_for_commands()
                result_placeholder.write(f"Voice Command: {command}")

            if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
                result = self.analyze_image_with_genai()
                result_placeholder.write(f"Result: {result}")

        self.cap.release()
        cv2.destroyAllWindows()


try:
    calc = Calculator()
    calc.streamlit_config()
    calc.main()
except Exception as e:
    add_vertical_space(5)
    st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
