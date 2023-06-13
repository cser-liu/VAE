import torch
import matplotlib.pyplot as plt


def showImage(img, tag=''):
    #img: 64x64x3
    plt.imshow(img)
    plt.axis('off')  
    plt.show()
    plt.savefig("output/visualize/image" + tag + ".jpg")