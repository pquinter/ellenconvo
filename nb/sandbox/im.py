from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
import sklearn.cluster
import sklearn.metrics
import sklearn.utils

im = io.imread('../data/IMG_2086.jpeg')

fig, axes = plt.subplots(1,3, figsize=(24,14))
[axes[i].imshow(im[:,:,i]**2) for i in range(3)]
plt.tight_layout()

im1 = im[:,:,0]
# compute 2D fourier transform
ft = np.fft.fftn(im1)
ft = np.log(np.abs(np.fft.fftshift(ft))**2)
fig, ax = plt.subplots(1, figsize=(24,14))
ax.imshow(ft, cmap='gray')

def transform_count_colors(im):
    # normalize image to range [0-1] for technical purposes 
    im = im / np.max(im.ravel())
    # unravel the image pixels
    w, h, d = im.shape
    trans_im = np.reshape(im, (w * h, d))
    # count number of colors
    color_df = pd.DataFrame(trans_im)
    n_colors = color_df.drop_duplicates().shape[0]
    return trans_im, n_colors
def recreate_image(centroids, labels, width, height):
    """
    Create a new image with centroids as color palette
    """
    d = centroids.shape[1]
    image = np.zeros((width, height, d))
    ind = 0
    for i in range(width):
        for j in range(height):
            image[i][j] = centroids[labels[ind]]
            ind += 1
    return image

im_, n_colors = transform_count_colors(im)
n_colors = 7
# Instantiate the mini-batch k-means class
kmeans_im = sklearn.cluster.MiniBatchKMeans(n_clusters=n_colors, random_state=12)
# cluster and get attributes
im_sample = sklearn.utils.shuffle(im_, random_state=0)[:1000]
kmeans_im.fit(im_sample)
labels = kmeans_im.predict(im_)
centroids = kmeans_im.cluster_centers_
imwidth, imheight = im.shape[:2]
new_im = recreate_image(centroids, labels, imwidth, imheight)
ims = [im, new_im]
fig, axes = plt.subplots(1,2, figsize=(24,14), sharex=True, sharey=True)
[axes[i].imshow(ims[i]**3, cmap='gray') for i in range(2)]
plt.tight_layout()
