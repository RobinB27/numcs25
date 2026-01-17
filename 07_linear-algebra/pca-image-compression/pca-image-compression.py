import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
    PCA can be used for Image compression.
    Treating the image as a matrix, we can find the most relevant
    singular vectors and reconstruct the image using only these.
    
    Note: Large images (like sample) take a few seconds to compute
"""

def truncated_SVD(A: np.ndarray, k: int):
    """ Return truncated SVD with k largest singular vectors """
    U, S, Vh = np.linalg.svd(A)
    return U[:, :k], S[:k], Vh[:k, :]


def compress_pca(img: np.ndarray, k: int):
    """ Reconstruct image with k largest singular vectors only """
    U, S, Vh = truncated_SVD(img, k)
    img_comp = U @ np.diag(S) @ Vh 
    return img_comp


def grayscale(img: Image): return np.asarray( img.convert("L") )

# Parameters to set

img_path = "./07_linear-algebra/pca-image-compression/sample-image.jpg"
k = 100     # Amount of singular vectors to keep

# Experiment runner

img = Image.open(img_path)
img = grayscale(img)
img_comp = compress_pca(img, k)
n = len( np.linalg.svd(img)[1] )    # Amount of singular vectors available
comp = k / n * 100                  # %information that is kept in the compressed image

# Plotting

fig, axs = plt.subplots(1, 2)

axs[0].imshow(img, cmap="gray")
axs[0].set_title(f"Original, n={n}")

axs[1].imshow(img_comp, cmap="gray")
axs[1].set_title(f"Compressed, k={k} ({round(comp, 2)}%)")

fig.tight_layout()
fig.suptitle("Image Compression via PCA")

plt.show()