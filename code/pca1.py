import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the .mat file
mat_data = scipy.io.loadmat('/Users/ritvikwarrier/Desktop/HW8-FinalProject/DeepNetFeature/feature_vgg_f.mat')
image_feat = mat_data['image_feat']

image_names = []
for name in image_feat['name'][0]:
    image_names.append(name[0])
image_names = np.array(image_names)

features_list = []
for feat in image_feat['feat'][0]:
    features_list.append(feat.flatten())
features = np.vstack(features_list)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.title('PCA Feature Embedding')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

for i, name in enumerate(image_names):
    plt.annotate(name, (principal_components[i, 0], principal_components[i, 1]), fontsize=8)

plt.show()
