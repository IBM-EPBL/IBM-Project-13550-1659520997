from flask import Flask,render_template
import cv2 
from tensorflow.keras.models import load_model
import numpy as np
import pyttsx3 #to convert text to speech
from skimage.transform import resize

app=Flask(__name__)
model=load_model("Model/keras_model.h5")

 
background = None
accumulated_weight = 0.5
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

word_dict = { 0:'A', 1:'B',  2:'C', 3: 'D', 4:'E', 5:'F', 6:'G',7:'H', 8:'I' }
vals = ['A', 'B','C','D','E','F','G','H','I']

def predict():
   
    def cal_accum_avg(frame, accumulated_weight):

        global background
        
        if background is None:
            background = frame.copy().astype("float")
            return None

        cv2.accumulateWeighted(frame, background, accumulated_weight)

    def segment_hand(frame, threshold=25):
        global background
        diff = cv2.absdiff(background.astype("uint8"), frame)
        _ , thresholded = cv2.threshold(diff, threshold,255,cv2.THRESH_BINARY)
        contours, hierarchy =cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            
            return (thresholded, hand_segment_max_cont)

    cam = cv2.VideoCapture(0)
    num_frames =0
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        if num_frames < 70:
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else: 
            hand = segment_hand(gray_frame)
            # Checking if we are able to detect the hand...
            if hand is not None:
                thresholded, hand_segment = hand
                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,ROI_top)],
                                 -1, (255, 0, 0),1)
                cv2.imshow("Thesholded Hand Image", thresholded)
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
                pred = model.predict(thresholded)
                result =word_dict[np.argmax(pred)]
                cv2.putText(frame_copy, result,(170, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                text_speech =  pyttsx3.init()
                rate=100
                text_speech.setProperty("rate", rate)
                text_speech.say (result)
                text_speech.runAndWait()
        
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), 
                      (ROI_right,ROI_bottom), (255,128,0), 3)
        num_frames += 1
        cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _",
        (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        cv2.imshow("Sign Detection", frame_copy)
        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF
        if k == 'q':
            break
    cam.release()
    cv2.destroyAllWindows()  
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=img')


if __name__ == '__main__':
      app.run(debug=False)