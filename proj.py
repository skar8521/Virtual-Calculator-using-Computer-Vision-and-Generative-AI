import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
import pyttsx3
import firebase_admin
from firebase_admin import credentials, firestore
import asyncio
import pytesseract
import matplotlib.pyplot as plt
import io

filterwarnings(action='ignore')

class Calculator:
    def __init__(self):
        load_dotenv()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)
        self.imgCanvas = np.zeros((550, 950, 3), dtype=np.uint8)
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
        self.p1, self.p2 = 0, 0
        self.p_time = 0
        self.fingers = []
        self.current_color = (255, 0, 255)
        self.brush_size = 5
        self.history = []
        self.engine = pyttsx3.init()
        self.setup_firebase()
    
    def setup_firebase(self):
        cred = credentials.Certificate("E:\Project-1\virtual-calc-firebase-adminsdk-we75j-7d8e9c1432.json")
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def streamlit_config(self):
        st.set_page_config(page_title='Virtual Calculator', layout="wide")
        page_background_color = """
        <style>
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        .block-container {
            padding-top: 0rem;
        }
        body {
            background-color: #f0f2f6;
        }
        h1 {
            color: #4CAF50;
            font-family: 'Arial';
        }
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
        add_vertical_space(1)
        with st.sidebar:
            st.markdown("### Calculation History")
            for calc in self.history[-10:]:
                st.write(calc)
            if st.button("Export History"):
                self.export_calculation()

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
                    h, w, _ = self.img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list:
            for id in [4,8,12,16,20]:
                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
            for i in range(5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[i*4][1], self.landmark_list[i*4][2]
                    cv2.circle(self.img, (cx, cy), 5, (255,0,255), 1)

    def handle_drawing_mode(self):
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), self.current_color, self.brush_size)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 3 and all(f == 1 for f in self.fingers[:3]):
            self.p1, self.p2 = 0, 0
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0,0,0), 15)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            self.imgCanvas = np.zeros((550, 950, 3), dtype=np.uint8)

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(self.img, 0.7, self.imgCanvas, 1, 0)
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        self.img = cv2.bitwise_or(img, self.imgCanvas)

    def analyze_image_with_genai(self):
        imgCanvas_rgb = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(imgCanvas_rgb)
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = ("Analyze the image and provide the following:\n"
                  "* The mathematical equation represented in the image.\n"
                  "* The solution to the equation.\n"
                  "* A short and sweet explanation of the steps taken to arrive at the solution.")
        response = model.generate_content([prompt, img_pil])
        return response.text

    def export_calculation(self):
        if self.history:
            history_str = "\n".join(self.history)
            with open("calculation_history.txt", "w") as file:
                file.write(history_str)
            with open("calculation_history.txt", "rb") as file:
                st.download_button('Download History', file, 'calculation_history.txt')

    async def analyze_image_async(self):
        response = await asyncio.to_thread(self.analyze_image_with_genai)
        self.result_placeholder.write(f"Result: {response}")
        self.speak("Analysis complete.")

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])
        with col1:
            stframe = st.empty()
        with col3:
            st.markdown('<h5 style="text-position:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            self.result_placeholder = st.empty()

        frame_count = 0
        while True:
            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown('<h4 style="text-position:center; color:orange;">Error: Could not open webcam. \
                            Please ensure your webcam is connected and try again</h4>', unsafe_allow_html=True)
                break

            self.process_frame()
            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()

            # Display the Output Frame in the Streamlit App
            img_display = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(img_display, channels="RGB")

            frame_count += 1
            if frame_count % 60 == 0:  # Analyze every 60 frames (~2 seconds)
                asyncio.run(self.analyze_image_async())
            
            # Handle user input for exiting the loop (e.g., pressing 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

try:
    calc = Calculator()
    calc.streamlit_config()
    calc.main()
except Exception as e:
    add_vertical_space(5)
    st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
    calc.speak("An error occurred. Please try again.")
