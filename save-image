from PIL import Image,ImageDraw,ImageFont
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# logger = logging.Logger(name='gen verification')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import cifar10, mnist
import concurrent.futures
import numpy as np
from PIL import Image
import argparse
import math
import matplotlib.pyplot as plt
import keras.backend as K
from keras import constraints, initializers
from Chinese_inputs import CommonChar, ImageChar
import os
from tqdm import tqdm

def save_image(cc_chars):
    for i, c in enumerate(cc.chars):
        ic.drawText(c)
        image = (ic.toArray() - 127.5) / 127.5
        # print(image)
        Image.fromarray(image.astype(np.uint8)).save(
            "./processed-image/" + str(i) + ".png")

if __name__ == "__main__":
    cc = CommonChar(path='./data')
    ic = ImageChar()
    if not os.exists('./processed-image'):
        os.mkdir('./processed-image')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(save_image, cc.chars)

