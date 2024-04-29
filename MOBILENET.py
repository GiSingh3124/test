import socket
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import datetime  
mobileNet = tf.keras.applications.MobileNet()
mobileNet.summary()
IMAGE_SHAPE = (224, 224)
img = 'img.jpeg'
input_data = Image.open(img).resize(IMAGE_SHAPE)

input_data = np.array(input_data)/255.0
input_data.shape

input_data = input_data[np.newaxis, ...]
input_data.shape

ct = datetime.datetime.now()
print("current time:-", ct)
server_output = mobileNet.predict(input_data)
ct = datetime.datetime.now()
print("current time:-", ct)

print(np.max(server_output))



predicted_class = tf.math.argmax(server_output[0], axis=-1)
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
predicted_class_name = imagenet_labels[predicted_class]
print(predicted_class_name)
