import os
import sys
import random

from glob import glob
from run_model import *

_ENABLE_PRINT = False

# 取消所有print
if not _ENABLE_PRINT:
    f = open(os.devnull, 'w')
    sys.stdout = f

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_OUT_FILE = os.path.join(os.getcwd(), "images", "out.txt")


def batch_test(image_count, image_path="images", out_file=_OUT_FILE):
    images_path = glob(os.path.join(os.getcwd(), image_path, "*.JPEG"))
    if len(images_path) < image_count:
        print("images not enough : ", len(images_path))
        return
    random.shuffle(images_path)
    model = AlexNet_model()
    result = []
    for image in images_path:
        prob, class_name = model.run(image)
        result.append("prob : {:.5e} \tclass : {}\n".format(prob, class_name))
    if out_file != None:
        with open(out_file, "w") as f:
            for i in range(len(images_path)):
                f.writelines([image+"\n", result[i]])


batch_test(image_count=10)
