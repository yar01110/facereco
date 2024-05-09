import onnxruntime as ort
class FaceOnnxEmbedder:
    
    def __init__(self,path : str="face_resnet-sim.onnx"):
        self.path=path
        self.ort_session = ort.InferenceSession(self.path)
    
    def __call__(self,prediction):
        prediction = self.ort_session.run(None, {'input': prediction})
        return prediction
