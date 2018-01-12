


import os
import matplotlib.pyplot as plt
FOLDER = "detected"

imgs = os.listdir(FOLDER)

for i, fname in enumerate(imgs):
    path = os.path.join(FOLDER, fname)
    img = plt.imread(path)
    plt.subplot(4, 6, i+1)
#     plt.title(fname)
    plt.axis('off')
    plt.imshow(img)
plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, hspace=0.05)
plt.show()

# IMG_FILE = os.path.join(FOLDER, "1.png")
# 
# 
# # Just a figure and one subplot
# 
# 
# # plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background


