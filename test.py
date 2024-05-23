import cv2
from database_connection import connectionToDb ,get_names_from_embeddings
from data import Data
from data_stream import DataStream
from faceEmbedder import FaceOnnxEmbedder
from Face_livenes import FaceLivenes
from yoloProcessing import YoloDetection
from image_preprocessing import TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes 
import numpy as np

class System:
    def __init__(self,video_path):
        self.curr_img=None
        self.prev_img=None
        self.datast=DataStream()a
        self.person_detection=YoloDetection("./yolov8m.onnx",tracking=True)
        self.face_detection=YoloDetection("./yolov8m-face.onnx")
        self.facelive=FaceLivenes()
        
        self.embedder=FaceOnnxEmbedder()
        
        self.T=TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes((128,128))
        
        self.client=connectionToDb()  
        
        self.cap = cv2.VideoCapture(video_path)
          
    def detect_changes(self,prev_frame, frame):
        
        """Detect changes between previous frame and current frame."""
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(prev_gray, gray)
        
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        non_zero_pixels = cv2.countNonZero(thresh)
        
        return non_zero_pixels
    
    def interpret_result(self,img):

        silence_counter=0
        
        self.curr_img=img
        
        if self.prev_img is not None:
            
            silence_counter+=1
            
            if silence_counter==30:
            
                self.datast.clear_buffer()
            
            if 100>self.detect_changes(self.curr_img,self.prev_img):
            
                silence_counter=0
                
                return img
            
            else : self.prev_img =None

        detected_people=self.person_detection(img)
       
        if detected_people is None:
            self.prev_img=self.curr_img
            return img
        self.datast.updateWithDetection(detected_people,img)
        
        if len(self.datast.curr_state.get("faceNotApeared"))==0:
            img=self.datast()
            self.datast.clear_curr_data()
            return img

       
        faces_images=self.datast.getNotDetectedFacePersonsImages()
        detected_faces=self.face_detection(faces_images)
        
        if  detected_faces is None:
            
            img=self.datast()

            return img
        
        if isinstance(detected_faces[0],int) :
                
                img=self.datast()

                return img
        
        self.datast.updateWithFaceDetection(detected_faces)

        if len(self.datast.curr_state.get('faceApeared'))<0:

            img=self.datast()
            self.datast.clear_curr_data()
            
            return
        
        appeared_faces,shapes=self.datast.get_face_images_withPad()

        proccedfaces=self.T(appeared_faces,shapes)
        
        proccedfaces=np.array(proccedfaces)

        faceliveresult=self.facelive(proccedfaces)
        
        self.datast.updateWithFaceLivenes(faceliveresult)
        
        valid_faces=self.datast.filterSpoofedFaces(proccedfaces)
        
        if len(valid_faces)==0:
        
            img=self.datast()
            self.datast.clear_curr_data()
        
            return img

        face_embs=self.embedder(valid_faces)

        identifier=get_names_from_embeddings(self.client,face_embs[0],threshold=1)
        
        self.datast.updateWithFaceRecognition(identifier)
        
        img=self.datast()
        
        self.datast.clear_curr_data()
        
        return img
    
    def __call__(self):
        
        while True:
    
            ret, frame = self.cap.read()
            
            output_frame=self.interpret_result(frame)
        
            cv2.imshow('Frame', output_frame)
        
            if cv2.waitKey(25) & 0xFF == ord('q'):
        
                break


        self.cap.release()
        
        cv2.destroyAllWindows()

s=System(0)

s()
