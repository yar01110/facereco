
import onnxruntime as ort
import numpy as np



class FaceLivenes:
    
    def __init__(self,path : str="AntiSpoofing_bin_2_128.onnx"):
        self.path=path
        self.ort_session = ort.InferenceSession(self.path)
    def __call__(self,prediction):
        prediction = self.ort_session.run(None, {'input': prediction})

        softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
        l=[np.argmax(softmax(np.array([pred],dtype=np.float32))) for pred in prediction[0]]
        #0:is a real face , 1 is a fake face
        
        return l
