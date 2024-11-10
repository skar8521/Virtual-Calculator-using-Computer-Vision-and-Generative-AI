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
filterwarnings(action='ignore')

# Additional imports for plotting
import plotly.graph_objs as go
import sympy as sp
from sympy import sympify, symbols, lambdify

class calculator:

    def streamlit_config(self):

        # Page configuration
        st.set_page_config(page_title='Calculator', layout="wide")

        # Page header transparent color and removes top padding 
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

        # Title and position
        st.markdown(f'<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
        add_vertical_space(1)

    def __init__(self):

        # Load the Env File for Secret API Key
        load_dotenv()

        # Initialize a Webcam to Capture Video and Set Width, Height and Brightness
        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=950)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=550)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=130)

        # Initialize Canvas Image
        self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)

        # Initializes a MediaPipe Hand object
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Set Drawing Origin to Zero
        self.p1, self.p2 = 0, 0

        # Set Previous Time is Zero for FPS
        self.p_time = 0

        # Create Fingers Open/Close Position List
        self.fingers = []

    def process_frame(self):

        # Reading the Video Capture to return the Success and Image Frame
        success, img = self.cap.read()

        # Resize the Image
        img = cv2.resize(src=img, dsize=(950,550))

        # Flip the Image Horizontally for a Later Selfie View Display
        self.img = cv2.flip(src=img, flipCode=1)

        # BGR Image Convert to RGB Image
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):

        # Processes an RGB Image and Returns the Hand Landmarks
        result = self.mphands.process(image=self.imgRGB)

        # Draws the landmarks and the connections on the image
        self.landmark_list = []

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(image=self.img, landmark_list=hand_lms, 
                                             connections=hands.HAND_CONNECTIONS)
                
                # Extract ID and Origin for Each Landmark
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = self.img.shape
                    x, y = lm.x, lm.y
                    cx, cy = int(x * w), int(y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):

        # Identify Each Finger's Open/Close Position
        self.fingers = []

        # Verify the Hands Detection in Web Camera
        if self.landmark_list != []:
            for id in [4,8,12,16,20]:

                # Index Finger, Middle Finger, Ring Finger and Pinky Finger
                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                        
                # Thumb Finger
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)

            # Identify Finger Open Position 
            for i in range(0, 5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1], self.landmark_list[(i+1)*4][2]
                    cv2.circle(img=self.img, center=(cx,cy), radius=5, color=(255,0,255), thickness=1)

    def handle_drawing_mode(self):

        # Both Thumb and Index Fingers Up in Drawing Mode
        if sum(self.fingers) == 2 and self.fingers[0]==self.fingers[1]==1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(255,0,255), thickness=5)

            self.p1,self.p2 = cx,cy
        
        # Thumb, Index & Middle Fingers UP ---> Disable the Points Connection
        elif sum(self.fingers) == 3 and self.fingers[0]==self.fingers[1]==self.fingers[2]==1:
            self.p1, self.p2 = 0, 0
        
        # Both Thumb and Middle Fingers Up ---> Erase the Drawing Lines
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[2]==1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
        
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(0,0,0), thickness=15)

            self.p1,self.p2 = cx,cy
        
        # Both Thumb and Pinky Fingers Up ---> Erase the Whole Thing (Reset)
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[4]==1:
            self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)
            # Clear the plot as well
            self.clear_plot = True

    def blend_canvas_with_feed(self):

        # Blend the Live Camera Feed and Canvas Images
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)

        # Canvas_BGR Image Convert to Gray Scale Image
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)

        # Gray Image Convert to Binary_Inverse Image
        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)

        # Binary_Inverse Image Convert into BGR Image
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        # Blending both Images
        img = cv2.bitwise_and(src1=img, src2=imgInv)

        # Canvas Color added on the Top on Original Image
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)

    def analyze_image_with_genai(self):

        # Canvas_BGR Image Convert to RGB Image 
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)

        # Numpy Array Convert to PIL Image
        imgCanvas = PIL.Image.fromarray(imgCanvas)

        # Configures the genai Library
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        # Initializes a Flash Generative Model
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Input Prompt
        prompt = "Analyze the image and provide the following:\n" \
                 "Equation: (provide the mathematical equation represented in the image)\n" \
                 "Solution: (provide the solution to the equation)\n" \
                 "Explanation: (provide a short and sweet explanation of the steps taken to arrive at the solution)"

        # Sends Request to Model to Generate Content using a Text Prompt and Image
        response = model.generate_content([prompt, imgCanvas])

        # Extract the Text Content of the Model’s Response.
        response_text = response.text

        # Parse the response to extract equation, solution, and explanation
        equation = ''
        solution = ''
        explanation = ''

        lines = response_text.split('\n')
        for line in lines:
            if 'Equation:' in line:
                equation = line.replace('Equation:', '').strip()
            elif 'Solution:' in line:
                solution = line.replace('Solution:', '').strip()
            elif 'Explanation:' in line:
                explanation = line.replace('Explanation:', '').strip()

        return {'equation': equation, 'solution': solution, 'explanation': explanation}
    
    def replace_unicode_superscripts(self, equation):
        superscripts = {
            '⁰': '0',
            '¹': '1',
            '²': '2',
            '³': '3',
            '⁴': '4',
            '⁵': '5',
            '⁶': '6',
            '⁷': '7',
            '⁸': '8',
            '⁹': '9',
        }
        for uni_sup, digit in superscripts.items():
            equation = equation.replace(uni_sup, '**' + digit)
        return equation

    def main(self):

        # Initialize history in session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'last_equation' not in st.session_state:
            st.session_state['last_equation'] = None

        # Layout configuration
        col1, col2 = st.columns([3,2])

        with col1:
            # Stream the webcam video
            stframe = st.empty()

        with col2:
            # Placeholder for result output
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()
            plot_placeholder = st.empty()

        # Sidebar for History
        with st.sidebar:
            st.title("History")
            history_placeholder = st.empty()

        # Variable to track if plot needs to be cleared
        self.clear_plot = False

        while True:

            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown(body=f'<h4 style="text-position:center; color:orange;">Error: Could not open webcam. \
                                    Please ensure your webcam is connected and try again</h4>', 
                            unsafe_allow_html=True)
                break

            self.process_frame()

            self.process_hands()

            self.identify_fingers()

            self.handle_drawing_mode()

            self.blend_canvas_with_feed()
            
            # Display the Output Frame in the Streamlit App
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(self.img, channels="RGB")

            # Gesture: Index and Middle fingers up to trigger AI analysis
            if sum(self.fingers) == 2 and self.fingers[1]==self.fingers[2]==1:
                result = self.analyze_image_with_genai()
                # Display the result
                result_text = f"**Equation:** {result['equation']}\n\n" \
                              f"**Solution:** {result['solution']}\n\n" \
                              f"**Explanation:** {result['explanation']}\n\n"
                result_placeholder.markdown(result_text)
                # Store the equation in session state for later use
                st.session_state['last_equation'] = result['equation']
                st.session_state['history'].append({
                    'equation': result['equation'],
                    'solution': result['solution'],
                    'explanation': result['explanation']
                })
                # Update the history panel
                history_markdown = ""
                for idx, item in enumerate(reversed(st.session_state['history'])):
                    history_markdown += f"**{idx+1}. Equation:** {item['equation']}\n\n" \
                                        f"**Solution:** {item['solution']}\n\n" \
                                        f"**Explanation:** {item['explanation']}\n\n---\n\n"
                history_placeholder.markdown(history_markdown)

            # Gesture: Index and Pinky fingers up to trigger graph plotting
            elif sum(self.fingers) == 2 and self.fingers[1]==self.fingers[4]==1:
                # Retrieve the last analyzed equation
                if st.session_state['last_equation'] is not None:
                    equation = st.session_state['last_equation']
                    equation = self.replace_unicode_superscripts(equation)

                    # Proceed to plot the graph
                    try:
                        # Prepare the equation for parsing
                        equation_to_plot = equation.replace('^', '**')

                        # Ensure the equation is in the form 'lhs = rhs'
                        if '=' in equation_to_plot:
                            lhs, rhs = equation_to_plot.split('=')
                            equation_to_parse = f'({lhs}) - ({rhs})'
                        else:
                            equation_to_parse = equation_to_plot

                        # Convert the equation string into a SymPy expression
                        eq = sp.sympify(equation_to_parse)

                        # Define symbols
                        symbols_in_eq = list(eq.free_symbols)
                        if len(symbols_in_eq) == 0:
                            plot_placeholder.write("No variables to plot.")
                            continue  # Skip plotting

                        # For simplicity, consider equations with 'x' and 'y'
                        x, y = sp.symbols('x y')

                        # Try to solve for y
                        solutions = sp.solve(eq, y)
                        if solutions:
                            # Plot each solution
                            fig = go.Figure()
                            x_vals = np.linspace(-10, 10, 400)
                            for sol in solutions:
                                y_func = sp.lambdify(x, sol, 'numpy')
                                y_vals = y_func(x_vals)
                                # Handle imaginary and NaN values
                                y_vals = np.real(y_vals)
                                y_vals[np.isinf(y_vals)] = np.nan
                                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=str(sol)))
                            fig.update_layout(title='Plot of the Equation', xaxis_title='x', yaxis_title='y')
                            # Display in Streamlit
                            plot_placeholder.plotly_chart(fig)
                        else:
                            # Try to solve for x
                            solutions = sp.solve(eq, x)
                            if solutions:
                                fig = go.Figure()
                                y_vals = np.linspace(-10, 10, 400)
                                for sol in solutions:
                                    x_func = sp.lambdify(y, sol, 'numpy')
                                    x_vals = x_func(y_vals)
                                    # Handle imaginary and NaN values
                                    x_vals = np.real(x_vals)
                                    x_vals[np.isinf(x_vals)] = np.nan
                                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=str(sol)))
                                fig.update_layout(title='Plot of the Equation', xaxis_title='x', yaxis_title='y')
                                # Display in Streamlit
                                plot_placeholder.plotly_chart(fig)
                            else:
                                plot_placeholder.write("Could not solve the equation for plotting.")
                    except Exception as e:
                        plot_placeholder.write(f"Could not plot the equation: {e}")
                else:
                    plot_placeholder.write("No equation to plot. Please perform AI analysis first.")

            # Gesture: Thumb and Pinky fingers up to clear plot and reset canvas
            if sum(self.fingers) == 2 and self.fingers[0]==self.fingers[4]==1:
                # Clear the plot
                plot_placeholder.empty()
                # Reset the last equation
                st.session_state['last_equation'] = None
                # Reset the result placeholder
                result_placeholder.empty()
                # Set clear_plot back to False
                self.clear_plot = False

            # Limit the frame rate
            cv2.waitKey(1)

        # Release the camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

try:

    # Creating an instance of the class
    calc = calculator() 

    # Streamlit Configuration Setup
    calc.streamlit_config()

    # Calling the main method
    calc.main()             

except Exception as e:

    add_vertical_space(5)

    # Displays the Error Message
    st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
