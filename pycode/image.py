import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tqdm import tqdm
import tensorflow_addons as tfa



actors = os.listdir('cropped')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('cropped', actor)):
        filenames.append(os.path.join('cropped', actor, file))

pickle.dump(filenames, open('filenames.pkl', 'wb'))

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resent50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print(model.summary())


def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


features = []
for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

pickle.dump(features, open('embedding.pkl', 'wb'))
