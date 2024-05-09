from skimage import io
from database_connection import connectionToDb ,get_names_from_embeddings
from data import Data
from data_stream import DataStream
from faceEmbedder import FaceOnnxEmbedder
from Face_livenes import FaceLivenes
from yoloProcessing import YoloDetection
from image_preprocessing import TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes 
import numpy as np
import cv2

#init
datastreaming=DataStream()
person_detection=YoloDetection("./yolov8m.onnx",tracking=True)
face_detection=YoloDetection("./yolov8m-face.onnx")
facelive=FaceLivenes()
embedder=FaceOnnxEmbedder()
t=TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes((128,128))
client=connectionToDb()


#img=io.imread('3people.jpg')
img=np.random.randint(10,size=(3,128,128))
detected_people=person_detection(img)
datastreaming.updateWithDetection(detected_people,img)
faces_images=datastreaming.getNotDetectedFacePersonsImages()
detected_faces=face_detection(faces_images)
datastreaming.updateWithFaceDetection(detected_faces)

appeared_faces,shapes=datastreaming.get_face_images_withPad()

proccedfaces=t(appeared_faces,shapes)
proccedfaces=np.array(proccedfaces)

faceliveresult=facelive(proccedfaces)
datastreaming.updateWithFaceLivenes(faceliveresult)
valid_faces=datastreaming.filterSpoofedFaces(proccedfaces)
face_embs=embedder(valid_faces)

identifier=get_names_from_embeddings(client,face_embs[0])
datastreaming.updateWithFaceRecognition(identifier)
datastreaming.clear_curr_data()




