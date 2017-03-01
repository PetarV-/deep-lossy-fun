from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imsave
import numpy as np

# Just perform the preprocessing
def preprocess_batch(x):
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x

# Just perform the deprocessing
def deprocess_batch(x):
    x[:, :, :, 0] += 103.939
    x[:, :, :, 1] += 116.779
    x[:, :, :, 2] += 123.68
    x = np.clip(x, 0.0, 255.0)
    return x

def load_and_process(img_path, target_size=None):
    # Feed in the image, convert to array
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    
    # Add the batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Perform the usual ImageNet preprocessing
    img = preprocess_input(img)
    
    return img

def deprocess_and_save(x, img_path):
    # Remove the batch dimension
    x = np.squeeze(x)

    # Restore the mean values on each channel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR --> RGB
    x = x[:, :, ::-1]

    # Clip unprintable colours
    x = np.clip(x, 0, 255).astype('uint8')

    # Save the image
    imsave(img_path, x)

