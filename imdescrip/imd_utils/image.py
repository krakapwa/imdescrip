# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
#
# This file is part of Imdescrip.
#
# Imdescrip is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# Imdescrip is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Imdescrip. If not, see <http://www.gnu.org/licenses/>.

""" Some useful and generic commonly performed image operations. """

import numpy as np
from skimage.transform import resize
import skimage.color as color
from skimage import io
from skimage import color
import cv2

def imread(fname):

    img = cv2.imread(fname)
    nchans = len(img.shape)
    if(nchans>2):
        img = img[:,:,0:3]
    else:
        img = color.gray2rgb(img)[:,:,0:3]
    return np.asfortranarray(img)

def im_resize (im, maxdim=None):
    """ resize the and image to a maximum dimension (preserving aspect)

    Arguments:
        im: image as numpy array
        maxdim: int of the maximum dimension the image should take (in pixels).
            None if no resize is to take place (same as imread).

    Returns:
        image: (height, width, channels) np.array of the image, if maxdim is not
            None, then {height, width} <= maxdim.

    Note: either gray scale or RGB images are returned depending on the original
            image's type.
    """

    # Resize image if necessary
    imgdim = max(im.shape[0], im.shape[1])
    if (imgdim > maxdim) and (maxdim is not None):
        scaler = float(maxdim)/imgdim
        #imout = cv.CreateMat(int(round(scaler*image.rows)),
        #                     int(round(scaler*image.cols)), image.type)
        #cv.Resize(image, imout)
        im= np.asfortranarray(cv2.resize(image,(int(round(scaler*image.shape[0])),int(round(scaler*image.shape[1]))) ))
        return im
    else:
        return im

def imread_resize (imname, maxdim=None):
    """ Read and resize the and image to a maximum dimension (preserving aspect)

    Arguments:
        imname: string of the full name and path to the image to be read
        maxdim: int of the maximum dimension the image should take (in pixels).
            None if no resize is to take place (same as imread).

    Returns:
        image: (height, width, channels) np.array of the image, if maxdim is not
            None, then {height, width} <= maxdim.

    Note: either gray scale or RGB images are returned depending on the original
            image's type.
    """

    # read in the image
    image = cv2.imread(imname)

    # Resize image if necessary
    imgdim = max(image.shape[0], image.shape[1])
    if (imgdim > maxdim) and (maxdim is not None):
        scaler = float(maxdim)/imgdim
        #imout = cv.CreateMat(int(round(scaler*image.rows)),
        #                     int(round(scaler*image.cols)), image.type)
        #cv.Resize(image, imout)
        image = np.asfortranarray(cv2.resize(image,(int(round(scaler*image.shape[0])),int(round(scaler*image.shape[1]))) ))
        return image
    else:
        return image

    # BGR -> RGB colour conversion
    #if image.type == cv.CV_8UC3:
    #    imcol = cv.CreateMat(imout.rows, imout.cols, cv.CV_8UC3)
    #    cv.CvtColor(imout, imcol, cv.CV_BGR2RGB)
    #    return np.asarray(imcol)
    #else:


def rgb2gray (rgbim):
    """ Convert an RGB image to a gray-scale image.

    Arguments:
        rgbim: an array (height, width, 3) which is the image. If this image is
            already a gray-scale image, this function returns it directly.

    Returns:
        image: an array (height, width, 1) which is a gray-scale version of the
            image.
    """

    # Already gray
    if rgbim.ndim == 2:
        return rgbim
    elif rgbim.ndim != 3:
        raise ValueError("Need a three channel image!")

    #grayim = cv.CreateMat(rgbim.shape[0], rgbim.shape[1], cv.CV_8UC1)
    #cv.CvtColor(cv.fromarray(rgbim), grayim, cv.CV_RGB2GRAY)
    grayim = color.rgb2gray(rgbim)
    return grayim

