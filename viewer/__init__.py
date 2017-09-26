import DicomReader, DicomViewer

import numpy as np
import matplotlib.pyplot as plt
from skimage.viewer import CollectionViewer
from skimage.data import coins, astronaut

def viewSequence(image_batch):
    data_importer = numpy2vtk(image_batch)
    DicomViewer.DicomViewer(data=data_importer, isSingleView=True).viewSlice()

def numpy2vtk(image_batch):
    dims = image_batch.shape
    image_batch = image_batch.reshape((dims[0], dims[1], dims[2]))
    data_importer = DicomReader.DicomReader.numpy2vtk(image_batch)
    return data_importer




def set_mask_clr(img_clr, mask, value, color):
    index = mask == value
    index = index[:, :, 0]
    img_clr[index, 0] = color[0]
    img_clr[index, 1] = color[1]
    img_clr[index, 2] = color[2]
def clr_image(image, mask):
    img_clr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float)
    img_clr[:, :, 0], img_clr[:, :, 1], img_clr[:, :, 2] = image[:, :, 0], image[:, :, 0], image[:, :, 0]
    set_mask_clr(img_clr, mask, 1, (0, 0, 0.5))
    set_mask_clr(img_clr, mask, 2, (0.5, 0, 0))
    return img_clr

def clr_imgs(images, masks):
    dims = images.shape
    imgs = np.zeros((dims[0], dims[1], dims[2], 3), dtype=np.float)
    for i, (image, mask) in enumerate(zip(images, masks)):
        img_clr = clr_image(image, mask)
        imgs[i, :, :, :] = img_clr
    return imgs


def show_image_mask(images, masks):
    # imgs = clr_imgs(images, masks)
    # CollectionViewer(imgs).show()
    for i, (image, mask) in enumerate(zip(images, masks)):
        if i == 109:
            plt.figure()
            plt.imshow(np.reshape(image, tuple(image.shape[:-1])), cmap='gray')
            plt.figure()
            plt.imshow(np.reshape(mask, tuple(image.shape[:-1])), cmap='gray')
            plt.figure()
            plt.imshow(clr_image(image, mask))
            plt.show()
            tmp=0


if __name__ == '__main__':
    CollectionViewer([coins(), astronaut()]).show()