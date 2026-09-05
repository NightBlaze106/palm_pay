import cv2
import numpy as np

img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
win_size = (128, 128)
block_size = (32, 32)
block_stride = (16, 16)
cell_size = (16, 16)
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
feat = hog.compute(img)
print(feat.shape, feat.dtype)
