import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine


detector = MTCNN()

def create_bbox(image):
    faces = detector.detect_faces(image)
    bounding_box = faces[0]['box']

    cv2.rectangle(image, 
                (bounding_box[0], bounding_box[1]), 
                (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), 
                (155,155,155),
                2)
    return image

def extract_face(image, resize=(224,224)):
    image = cv2.imread(image)

    faces = detector.detect_faces(image)
    #print(f'face: {faces}')
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height

    # the numbers added and subtracted is jsut 
    # so we can get the entire face region
    face_boundary = image[y1-30:y2+30, x1-30:x2+30]

    face_image = cv2.resize(face_boundary, resize)

    return face_image

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


faces = [extract_face(image) \
    for image in ['images\\15.jpg', 'images\\12.jpg']]
#faces = extract_face("images\dara.jpg")
cv2.imshow("image",faces[0])
cv2.imshow("images",faces[1])
print(get_similarity(faces))
cv2.waitKey(0)