import os
import os.path
# path
ROOT = os.path.expanduser("~/dataset/cifar/")
CKPT = os.path.expanduser("./ckpt.d/")
EVAL = os.path.expanduser("./eval.d/")

# canny
HIGHTH = 200.0
COELOWTH = 0.5

# gpu
GPUORDINAL = 1

# tensorboard
TFLOGDIR = os.path.expanduser("~/tfboardlog/")
