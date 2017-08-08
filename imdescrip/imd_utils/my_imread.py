from skimage import io
from skimage import color

def imread(fname):

    img = io.imread(fname)
    nchans = len(img.shape)
    if(nchans>2):
        return img
    else:
        return color.gray2rgb(img)
