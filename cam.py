import cv2
from threading import Thread
  
class Webcam:
  
    def __init__(self):
        # using video stream from IP Webcam for Android
        url = "rtsp://192.168.1.115/test.mp4&t=unicast&p=rtsp&ve=H264&w=1280&h=720&ae=PCMU&sr=8000"
        self.video_capture = cv2.VideoCapture(url)
        self.current_frame = self.video_capture.read()[1]
          
    # create thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()
  
    def _update_frame(self):
        while(True):
            try:
                self.current_frame = self.video_capture.read()[1]
                cv2.imshow('VIDEO',self.current_frame)
                cv2.waitKey(1)
            except:
                pass
                  
    # get the current frame
    def get_current_frame(self):
        return self.current_frame



# Create an instance of the Webcam class
webcam = Webcam()

# Start capturing frames
webcam.start()

# Get the current frame
image = webcam.get_current_frame()
