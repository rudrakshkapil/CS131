import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    u,s,v = np.linalg.svd(image)
    
    u_cols = u[:,:num_values]
    s_diag = np.diag(s[:num_values])
    US = np.dot(u_cols, s_diag)
    
    v_rows = v[:num_values,:]
    USV = np.dot(US,v_rows)
    compressed_image = USV
    
    compressed_size = num_values * (u.shape[0] + 1 + v.shape[1])
    
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
