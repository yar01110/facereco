from typing import Any , Union ,List
import supervision as sv
from ultralytics import YOLO
import numpy as np

class YoloDetection:
    def __init__(self 
                 ,model_path : str
                 ,tracking :bool =False
                 ,classes :list[int]=[0] ) -> None:
        
        self.classes=classes
        self.tracking=tracking
        if self.tracking:
            self.tracker = sv.ByteTrack()
        
        self.model = YOLO(model_path,task='detect',verbose=False)
    
    def __call__(self, img : Union[np.ndarray, list[np.ndarray]]) -> Union[sv.Detections,list]:
        #numpy array should be in shape of (C,H,W)
        
        try : 
            
            if isinstance(img,np.ndarray) :

                detection=self.infer(img)
            else :
                detection=self.multi_infer(img)
                
            return detection
        except :
                
            print(f"invalid input type expected a numpy.ndarray or a list of nmpy.ndarry got {type(img)} object tyope ")
       
        
        
        
    def infer(self,img : np.ndarray ) -> sv.Detections:
        
        result = self.model.predict(img, classes=self.classes,conf=0.5, verbose=False)
        #for a single image always take the first element cuz the obb is not supported in yolov8
        detections = sv.Detections.from_ultralytics(result[0])
        
        if self.tracking :
            detections = self.tracker.update_with_detections(detections)
       
        
        return detections
    
    def multi_infer(self,img :list[np.ndarray]) -> list :
        
        """ multi_infer is used to run iference of the yolo api in multiple images
            it is not consistent to use tracking on multiple images and further implementation
            should be done
        """
        result = self.model.predict(img, classes=self.classes,conf=0.50, verbose=False)
        
        
        detections=[
             
             self.reduceToSingleBb(detection)
                     for detection in result
            ]
        
        return detections
    def reduceToSingleBb(self,detection: list) -> sv.Detections:
        if len(detection.boxes) == 0 :
            return 0
        np.argmax(detection.boxes.conf.cpu().numpy())
        MAX_CONFIDENCE = np.argmax(detection.boxes.conf.cpu().numpy())
        box=detection.boxes.xyxy.cpu().numpy()[MAX_CONFIDENCE]
        
        return box

