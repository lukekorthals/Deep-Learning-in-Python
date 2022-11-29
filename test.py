import tensorflow as tf
import tensorflow_addons as tfa 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Image Quality Attributes
# Sharpness
# Noise
# Dynamic Range
# Tone reproduction
# Contrast
# Color
# Distortion
# Vignetting
# Exposure accuracy
# Lateral chromatic aberration
# Lens flare
# color moire
# artifacts

# Input Nell things can happen when people take pictures
# lightning, focus, film grain (noise)
# lightning - to bright, to dark, shot against light


def gausian_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.5
    sigma = var**0.01
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

# Read Image
image_path = './diving.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=3)
image_raw = image
image = tf.expand_dims(image, axis=0)

# Transform Image
image_condense = tf.nn.pool(input=image, window_shape=(20,20), pooling_type="MAX", strides=(1, 1), padding='SAME')
image_blur =  tfa.image.gaussian_filter2d(image, filter_shape = (100, 100), sigma = 10.0)
image_noise = gausian_noise(image_raw)

# Save Trasnformed Images
tf.keras.utils.save_img("diving_condensed.png", image_condense[0])
tf.keras.utils.save_img("diving_blurred.png", image_blur[0])
tf.keras.utils.save_img("diving_noisy.png", image_noise)

# Plot Results
f, axarr = plt.subplots(1,4)
axarr[0].imshow(tf.squeeze(image))
axarr[0].set_title("Raw Image")
axarr[1].imshow(tf.squeeze(image_condense))
axarr[1].set_title("Condensed Image")
axarr[2].imshow(tf.squeeze(image_blur))
axarr[2].set_title("Blurred Image")
axarr[3].imshow(image_noise)
axarr[3].set_title("Noisy Image")
plt.show()




"""plt.figure(figsize=(12, 6))
plt.subplot(111)
plt.imshow(tf.squeeze(image))
plt.axis('off')
plt.title('Input')
plt.subplot(122)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pool')
plt.subplot(133)
plt.imshow(tf.squeeze(image_blur))
plt.axis('off')
plt.title('Blur')
plt.show()"""