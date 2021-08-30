import os
import sys
import time
import random

import progressbar
from glob import glob
from run_model import *

_BAR = progressbar.ProgressBar()
_IMAGE_COUNT = 1000
_DIVID_COUNT = 50
_ENABLE_PRINT = True
_IMAGE_PATH = "images"
_TIME_OUT_FILE = "time.txt"
_CLASS_OUT_FILE = os.path.join(os.getcwd(), "images", "out.txt")
_INIT_FILE = "images/zebra.jpeg"

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

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


def init():
    global model
    model = AlexNet_model()
    model.run(_INIT_FILE)


def pre_precess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img.astype(np.float32), (227, 227))
    # Subtract the ImageNet mean
    img -= imagenet_mean
    # Reshape as needed to feed into model
    img = img.reshape((1, 227, 227, 3))
    return img


if __name__ == "__main__":
    result = []
    start = time.clock()
    init()
    end = time.clock()
    result.append("AlexNet Model Init : {:.05f}s\n".format(end - start))
    images_path = glob(os.path.join(os.getcwd(), _IMAGE_PATH, "*"))
    images_path = [image for image in images_path if is_image_file(image)]
    if len(images_path) < _IMAGE_COUNT:
        print("images not enough : ", len(images_path))
        exit()
    random.shuffle(images_path)
    count, index, all_time = 0, 0, 0.0
    print("Start run batch test")
    _BAR.start()
    while count < _IMAGE_COUNT:
        _BAR.update(count*100/_IMAGE_COUNT)
        image = images_path[count]
        img = pre_precess(image)
        start = time.clock()
        success = model.fast_fun(img)
        if success:
            count += 1
            end = time.clock()
            all_time += end-start
        if count % _DIVID_COUNT == 0 and count > 0 and success:
            result.append("batch size : {:<10d} time : {:.05f}s\n".format(
                count, all_time))
        index += 1
    with open(_TIME_OUT_FILE, "w") as f:
        f.writelines(result)
    _BAR.finish()
    print("See result in ", _TIME_OUT_FILE)
