import os
import cv2
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from caffe_classes import class_names


class AlexNet_model():
    def __init__(self):
        print("AlexNet model init...")
        self.imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

        # placeholder for input and dropout rate
        self.x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        # create model with default config ( == no skip_layer and 1000 units in the last layer)
        self.model = AlexNet(self.x, self.keep_prob, 1000, [])

        # define activation of last layer as score
        score = self.model.fc8
        # create op to calculate softmax
        self.softmax = tf.nn.softmax(score)

        with tf.Session().as_default() as self.sess:
            # Initialize all variables
            self.sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the model
            self.model.load_initial_weights(self.sess)
        print("init finished\n")
    
    def fast_fun(self,img):
        try:
            # Run the session and calculate the class probability
            probs = self.sess.run(self.softmax, feed_dict={
                                    self.x: img, self.keep_prob: 1})[0]
            index = np.argmax(probs)
            return True
        except:
            return False

    def run(self, image_path):
        try:
            if not os.path.exists(image_path):
                print("file not exists\n")
                return False
            img = cv2.imread(image_path)
            img = cv2.resize(img.astype(np.float32), (227, 227))
            # Subtract the ImageNet mean
            img -= self.imagenet_mean
            # Reshape as needed to feed into model
            img = img.reshape((1, 227, 227, 3))
            # Run the session and calculate the class probability
            probs = self.sess.run(self.softmax, feed_dict={
                                  self.x: img, self.keep_prob: 1})[0]
            # Get the class name of the class with the highest probability
            # indexs = np.argsort(probs)
            # for i in range(10):
            #     print("{}\t{:.6e}\t{}".format(
            #         i+1, probs[indexs[i]], class_names[indexs[i]]))
            index = np.argmax(probs)
            return  True
        except:
            print("error\n")
            return False
