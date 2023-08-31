import cv2
from matplotlib import pyplot as plt
import os
src_filepath = './download_image/1.png'
img = cv2.imread(src_filepath)
dst = cv2.fastNlMeansDenoisingColored(img,None,30,10,7,21)
cv2.imwrite(os.path.join('./download_image', 'your_image.png'), dst)

# plt.subplot(121),plt.imshow(img)
# plt.show()
# plt.savefig(dst)