import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pylab as plt
import cv2
import scipy

save_dir = './data/image2'
print("resizing images")
print("current directory:",save_dir)
image_folder = '/home/mao/Downloads/dataset/hanzi/test2'
image_names = os.listdir(image_folder)
all_image_paths = []
pixth = 64

for i in range(len(image_names)):
    all_image_paths.append(os.path.join(image_folder, image_names[i]))

def modify_image(image):
    resized = tf.image.resize_images(image, [pixth, pixth])
    resized.set_shape([pixth,pixth,1])
    flipped_images = resized
    return flipped_images

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return image

def inputs():
    filenames = all_image_paths
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=None)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return reshaped_image

with tf.Graph().as_default():
    image = inputs()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for i in range(200):
        img = sess.run(image)
        img = np.reshape(img, [pixth,pixth])
        # plt.imshow(img)
        # plt.show()
        # img.astype('uint8')
        scipy.misc.imsave(name=os.path.join(save_dir,"hanzi"+str(i)+".jpeg"), arr=img)
        # img = Image.fromarray(img, mode='1')
        # img.save(os.path.join(save_dir,"hanzi"+str(i)+".jpeg"))