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
image_path = './random_tests/diving.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)
image = image / 255
image_raw = image
image = tf.expand_dims(image, axis=0)
image_path2 = './random_tests/diving2.jpg'
image2 = tf.io.read_file(image_path2)
image2 = tf.io.decode_jpeg(image2)
image2 = image2 / 255
image2 = tf.expand_dims(image2, axis=0)
image_path3 = './random_tests/grb.png'
image3 = tf.io.read_file(image_path3)
image3 = tf.io.decode_jpeg(image3, channels=3)
image3 = image3 / 255
print(image3.shape)
image3 = tf.expand_dims(image3, axis=0)
# Transform Image
Wk=3
Hk=3
Dk=3
w=np.zeros(shape=(1,Wk,Hk,Dk)).astype(int) #initialize 3x3x3 kernel
w[:,:,0]=np.array(([3,3,3],[-6,-6,-6],[-3,-3,-3]),dtype='float32')
w[:,:,1]=np.array(([3,-6,3],[3,-6,3],[3,-6,3]),dtype='float32')
w[:,:,2]=np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),dtype='float32')
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
kernel = tf.cast(kernel, "float32")
kernel = tf.expand_dims(kernel, axis=0)
kernel = tf.expand_dims(kernel, axis=0)

image_conv = tf.nn.conv2d(input=image,filters=w,strides=1,padding="SAME")
image_conv2 = tf.nn.conv2d(input=image2,filters=w,strides=1,padding="SAME")
image_conv3 = tf.nn.conv2d(input=image3,filters=w,strides=1,padding="SAME")
image_conv = tf.nn.pool(input=image_conv, window_shape=(20,20), pooling_type="MAX", strides=(1, 1), padding='SAME')
image_conv2 = tf.nn.pool(input=image_conv2, window_shape=(20,20), pooling_type="MAX", strides=(1, 1), padding='SAME')
image_conv3 = tf.nn.pool(input=image_conv3, window_shape=(20,20), pooling_type="MAX", strides=(1, 1), padding='SAME')

f, axarr = plt.subplots(2,3)
axarr[0,0].imshow(tf.squeeze(image))
axarr[1,0].imshow(tf.squeeze(image_conv))
axarr[0,1].imshow(tf.squeeze(image2))
axarr[1,1].imshow(tf.squeeze(image_conv2))
axarr[0,2].imshow(tf.squeeze(image3))
axarr[1,2].imshow(tf.squeeze(image_conv3))
plt.show()
quit()


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