from __future__ import annotations
import cv2
from typing import Optional , Dict , Set
from data import Data
from dataclasses import dataclass , field 
import numpy as np
import supervision as sv

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class DataStream(metaclass=Singleton):
    
    data_buffer: Dict[int,Data]= field(default_factory=dict)
    
    curr_state: Dict[str, Set[int]] = field(default_factory=lambda: {"faceNotApeared": set(), "faceApeared": set(), "stranger":set()})
    
    curr_data: Dict[int,Data]= field(default_factory=dict)

    img: Optional[np.ndarray] = None
    
    
    def updateWithDetection(cls,detection: sv.Detections,img: np.ndarray) -> DataStream : 
        cls.img=img
        
        for xyxy, trackId in zip(detection.xyxy,detection.tracker_id):
            
            if trackId in cls.data_buffer:

                cls.curr_data[trackId]=cls.data_buffer[trackId]

                cls.curr_data[trackId].personxyxy=xyxy
                
                cls.curr_data[trackId].personImg=cls.bbtoimg(xyxy,img)

                if cls.data_buffer[trackId].state in cls.curr_state.keys():

                    cls.curr_state[cls.data_buffer[trackId].state].add(trackId)

            else:

                data=Data(xyxy,cls.bbtoimg(xyxy,img),trackId)

                cls.curr_data[trackId]=cls.data_buffer[trackId]=data

                cls.curr_state["faceNotApeared"].add(trackId)
    
    def updateWithFaceDetection(cls,detection: list) :
        
        def adjust_relative_to_image(person_bb,face_bb):

            offset_x = person_bb[0]  # left
            offset_y = person_bb[1]  # top

            # Adjust face bounding box coordinates relative to the original image
            return (
                face_bb[0] + offset_x,                      # adjusted left
                face_bb[1] + offset_y,                      # adjusted top
                face_bb[2] + offset_x,  # adjusted right
                face_bb[3] + offset_y   # adjusted bottom
            
            )
        
        for person ,result in zip(cls.curr_state["faceNotApeared"],detection):
            
            if isinstance(result,int):
                continue
            
            faceBB=adjust_relative_to_image(cls.data_buffer[person].personxyxy,result)
            
            cls.data_buffer[person].facexyxy,cls.data_buffer[person].faceImg,cls.curr_data[person].state=faceBB,cls.bbtoimg(faceBB,cls.img),"faceApeared"
            
            cls.curr_state["faceApeared"].add(person)

    def get_face_images_withPad(cls) -> list[np.ndarray] :
        print(cls.curr_state["faceApeared"])
        
        max_height = 0
        max_width = 0
        
        persons_img_index=[]
        for index in cls.curr_state["faceApeared"]:
            height, width, _ = cls.data_buffer[index].faceImg.shape
            persons_img_index.append(cls.data_buffer[index].faceImg.shape)
            max_height = max(max_height, height)
            max_width = max(max_width, width)
            
        def padding(image,desired_shape):
            pad_width = [(0, desired_shape[i] - image.shape[i]) for i in range(len(desired_shape))]
            padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
            return padded_image
        
        face_imgs=np.stack([padding(cls.data_buffer[index].faceImg,(max_height,max_width,3)) for index in cls.curr_state["faceApeared"]])
        persons_img_index = np.array(persons_img_index, dtype=np.int64)
        return face_imgs ,persons_img_index
    
    def updateWithFaceLivenes(cls,validFace :list[int]) -> None:
        
        for index,result in zip(cls.curr_state["faceApeared"],validFace):
            
            if result==0:

                cls.data_buffer[index].state="stranger"
                cls.curr_state["stranger"].add(index)

            else:
                cls.data_buffer[index].state="spoofer"


    def filterSpoofedFaces(cls,images: np.array) -> np.array:
        
        if len(cls.curr_state['stranger'])==0:
            return images

        return np.stack([img for img, state in zip(images, cls.curr_state['faceApeared']) if state in cls.curr_state['stranger']])
        
    def updateWithFaceRecognition(cls,identity) -> None:
        if identity==0:
            return None
    
        for index,ident in zip(cls.curr_state["stranger"],identity):
            
            cls.data_buffer[index].identification,cls.data_buffer[index].state=ident,"recognated"
            
    
    def clear_curr_data(cls):
        #every frame
        cls.curr_data.clear()
        
        for key in cls.curr_state:
            cls.curr_state[key].clear()
    
    def clear_buffer(cls):
        #after 30 frame from no person detection
        cls.data_buffer.clear()
    

    def getNotDetectedFacePersonsImages(cls) -> list[np.ndarray] :
          
        persons_imgs=[cls.data_buffer[person].personImg for person in cls.curr_state["faceNotApeared"]]

        return persons_imgs

    @staticmethod
    def bbtoimg(xyxy: np.ndarray,img: np.ndarray  ) ->np.ndarray: #img: np.ndarray=img
        x1, y1, x2, y2 = xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[y1:y2, x1:x2]
    
    def draw_bounding_boxes(cls,bboxes, names):
       
        cv2.putText(cls.img, str(len(cls.curr_data)), (cls.img.shape[1] - cv2.getTextSize(str(len(cls.curr_data)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0] - 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for bbox, name in zip(bboxes, names):
            x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
            cv2.rectangle(cls.img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(cls.img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return cls.img
    
    def get_names_and_boxes(cls):
        boxes=[]
        names=[]
        for data  in cls.curr_data.values():
            boxes.append(data.personxyxy)
            if data.state=="faceNotApeared" or data.state=="stranger":
                names.append("stranger")
            elif data.state=="spoofer":
                names.append("spoofer")
            else :
                if isinstance(data.identification,str):
                    names.append("stranger")
                    continue
                names.append(f"name: {data.identification.get('name')} role: {data.identification.get('role')}")

                
        
        return boxes,names
    
    
    def __call__(cls):

        boxes ,names=cls.get_names_and_boxes()
        img=cls.draw_bounding_boxes(boxes,names)
        return img
