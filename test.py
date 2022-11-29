import tensorflow as tf
import tensorflow_addons as tfa 
import matplotlib.pyplot as plt

# Read image
image_path = './diving.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=3)
image = tf.expand_dims(image, axis=0)

image_condense = tf.nn.pool(
    input=image,
    window_shape=(20,20),
    pooling_type="MAX",
    strides=(2, 2),
    padding='SAME',
)

image_blur =  tfa.image.gaussian_filter2d(
    image,
    filter_shape = (100, 100),
    sigma = 10.0
)

f, axarr = plt.subplots(1,3)
axarr[0].imshow(tf.squeeze(image))
axarr[1].imshow(tf.squeeze(image_condense))
axarr[2].imshow(tf.squeeze(image_blur))
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