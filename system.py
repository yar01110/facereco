from database_connection import connectionToDb ,get_names_from_embeddings
from data_stream import DataStream
from faceEmbedder import FaceOnnxEmbedder
from Face_livenes import FaceLivenes
from yoloProcessing import YoloDetection
from image_preprocessing import TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes 
import numpy as np


class System:
    def __init__(self):
        self.self.current_img=None
        self.previous_image=None
        self.self.datastreaming=DataStream()
        self.self.person_detection=YoloDetection("./yolov8m.onnx",tracking=True)
        self.face_detection=YoloDetection("./yolov8m-face.onnx")
        self.facelive=FaceLivenes()
        self.embedder=FaceOnnxEmbedder()
        self.T=TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes((128,128))
        self.client=connectionToDb()

        if self.person_detection(self.img) is None:
            return
        else :
            self.datastreaming.updateWithDetection(self.person_detection(self.img))
        
        if len(self.datastreaming.curr_state.get("faceNotApeared"))==0:
            return