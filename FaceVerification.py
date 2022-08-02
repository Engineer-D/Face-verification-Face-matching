from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import numpy as np

def get_embeddings(faces):
    face = np.asarray(faces, 'float32')

    face = preprocess_input(face, version=2)

    model = VGGFace(model='resnet50', include_top = False,\
        input_shape=(224,224,3), pooling='avg')

    return model.predict(face)

def get_similarity(faces):
    embeddings = get_embeddings(faces)

    score = cosine(embeddings[0], embeddings[1])

    if score <= 0.5:
        return "Face Matched", score
    
    return "Face Not Matched", score