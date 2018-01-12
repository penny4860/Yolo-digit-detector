


import os
import matplotlib.pyplot as plt
FOLDER = "detected"

imgs = os.listdir(FOLDER)

for i, fname in enumerate(imgs):
    path = os.path.join(FOLDER, fname)
    img = plt.imread(path)
    plt.subplot(4, 5, i+1)
    plt.title(fname)
    plt.imshow(img)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


# IMG_FILE = os.path.join(FOLDER, "1.png")
# 
# 
# # Just a figure and one subplot
# 
# 
# # plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background


