import os
import pickle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

actors = os.listdir(r'C:\Users\Chetan\Desktop\olympicweb\pycode\sport image')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join(r'C:\Users\Chetan\Desktop\olympicweb\pycode\sport image', actor)):
        filenames.append(os.path.join(r'C:\Users\Chetan\Desktop\olympicweb\pycode\sport image', actor, file))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
# Your code for loading the images and feature extraction
features = []
for file in filenames:
    img = Image.open(file).convert('RGB')
    img_array = np.array(img.resize((224, 224)))
    img_array = img_array / 255.0  # Normalize image
    img_feature = img_array.reshape(1, -1)  # Flatten the image array
    features.append(img_feature)

# Apply PCA for dimensionality reduction
scaler = StandardScaler()
features_scaled = scaler.fit_transform(np.concatenate(features, axis=0))
pca = PCA(n_components=100)
features_pca = pca.fit_transform(features_scaled)


pickle.dump(features_pca, open('embedding.pkl', 'wb'))
