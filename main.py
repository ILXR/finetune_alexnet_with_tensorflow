import os
import sys
import time
import random

from glob import glob
from run_model import *

_IMAGE_COUNT = 201
_ENABLE_PRINT = False
_IMAGE_PATH = "images"
_TIME_OUT_FILE = "time.txt"
_CLASS_OUT_FILE = os.path.join(os.getcwd(), "images", "out.txt")
_INIT_FILE = "images/zebra.jpeg"

# 取消所有print
if not _ENABLE_PRINT:
    f = open(os.devnull, 'w')
    sys.stdout = f

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = None


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def batch_test(image_count, image_path=_IMAGE_PATH, out_file=_CLASS_OUT_FILE):
    images_path = glob(os.path.join(os.getcwd(), image_path, "*"))
    images_path = [image for image in images_path if is_image_file(image)]
    if len(images_path) < image_count:
        print("images not enough : ", len(images_path))
        return False
    random.shuffle(images_path)

    result = []
    for image in images_path:
        if image_count == 0:
            break
        prob, class_name = model.run(image)
        result.append("prob : {:.5e} \t class : {}\n".format(prob, class_name))
        image_count -= 1
    if out_file != None:
        with open(out_file, "w") as f:
            for i in range(len(result)):
                f.write(images_path[i]+"\n"+result[i])
    return True


def init():
    global model
    model = AlexNet_model()
    model.run(_INIT_FILE)


if __name__ == "__main__":
    result = []
    start = time.clock()
    init()
    end = time.clock()
    result.append("AlexNet Model Init : {:.05f}s\n".format(end - start))
    for i in range(0, _IMAGE_COUNT, 5):
        if i == 0:
            i += 1
        start = time.clock()
        success = batch_test(i)
        if success:
            end = time.clock()
            result.append("batch size : {:<10d} time : {:.05f}s\n".format(
                i, end - start))
    with open(_TIME_OUT_FILE, "w") as f:
        f.writelines(result)
