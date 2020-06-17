import tensorflow as tf

#from tensorflow.keras import VGGFace
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from scipy.spatial.distance import cosine

# khởi tạo mtcnn và vggface
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
# cái này để sửa lỗi ko hiểu nổi khi dùng Keras với Flask, dùng trong hàm get_embeddings bên dưới
graph = tf.get_default_graph()

def resize_image(filename, required_size=(224, 224)):
    image = Image.open(filename)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def extract_face(filename, required_size=(224, 224)):
    # load ảnh
    pixels = pyplot.imread(filename)
    # ko cần kênh alpha, xử lý trường hợp ảnh PNG
    pixels = pixels[:,:,:3]
    if pixels.dtype == np.float64 or pixels.dtype == np.float32:
        pixels *= 255
        pixels = pixels.astype(np.uint8)
    results = detector.detect_faces(pixels)
    # lấy vị trí khuôn mặt
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    # resize ảnh
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# lấy vector đặc trưng của 1 list ảnh
def get_embeddings(filenames, need_to_extract=False, model=model):
    global graph
    with graph.as_default():
        faces = []
        if need_to_extract is False:
            faces = [resize_image(f) for f in filenames]
        else:
            faces = [extract_face(f) for f in filenames]  
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = model.predict(samples)
        return yhat