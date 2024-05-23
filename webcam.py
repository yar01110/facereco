import cv2
from threading import Thread
import requests 
import json 
import base64
from SYSTEM import System
s=System()

  
class Stream:
  
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture
      
    # create thread for capturing images
    def frameResponse(self,frame,host="localhost",port=5000):
      headers = {
          'content-type': 'image/jpeg',
          # 'Accept':'text/plain'
      }
      try:
          response = requests.post(url="http://localhost:5000/",data=frame,headers=headers)
      except KeyError:
          print("Connection Error")
    def start(self):
        
        Thread(target=self._update_frame, args=()).start()
  
    def _update_frame(self):

        while(True):
            try:

                self.current_frame = s(self.video_capture.read()[1])
                
              
                encodedFrame =cv2.imencode('.jpg',self.current_frame)[1].tobytes()

                self.frameResponse(encodedFrame)

                cv2.imshow('VIDEO',self.current_frame)
                print(f"${self.current_frame.shape}")
                cv2.waitKey(1)
            except KeyError:
                print("Connection Error")
