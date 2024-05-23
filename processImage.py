from ultralytics import YOLO
from image_preproc import TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes as Trans
model=YOLO("yolov8m-face.onnx","predict",verbose=False)
from skimage import io
from database_connection import connectionToDb
client=connectionToDb()
from faceEmbedder import FaceOnnxEmbedder
import numpy as np
from image_preproc import TransposeResizeNormalizeBatchingWithDifferentDimesnsionShapes as Trans
import argparse
model=YOLO("yolov8m-face.onnx","detect",verbose=False)
def imagetoemb(image:str):

    image=io.imread(image)

    result=model.predict(image,conf=0.5, verbose=False)
    xyxy=result[0].boxes.xyxy.cpu().numpy()
    def bbtoimg(xyxy ,img ) : #img: np.ndarray=img
        x1, y1, x2, y2 = xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[y1:y2, x1:x2]
    embedder=FaceOnnxEmbedder("face_resnet-sim.onnx")
    img=bbtoimg(xyxy,image)
    T=Trans((128,128))
    img=T(img)
    img=np.array(img)
    img=np.stack([img])
    emb=embedder(img)
    return emb



def parser():
    parser = argparse.ArgumentParser(description="Process some user information.")
    parser.add_argument('name', type=str, help='The name of the user')
    parser.add_argument('role', type=str, help='The role of the user')
    parser.add_argument('image', type=str, help='The path to the user\'s image')

    args = parser.parse_args()
    data={"name":args.name,"role":args.role,"embedding":imagetoemb(args.image)[0][0]}
    res=client.insert("employees",data=data)
    print("INSERTING OPERATION DONE SUCCEFULLY : ",res)
    

parser()

