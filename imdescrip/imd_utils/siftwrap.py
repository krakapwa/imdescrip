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

""" Functions for extracting dense SIFT patches from images.

    This file has a few useful functions for extracting and processing scale
    invariant feature transform (SIFT) patches from images. This module
    essentially wraps the vlfeat DSIFT python routines.

    NOTE:   The SIFT descriptors output by vlfeat are [0, 255] integers!

    TODO:   Cut out the middle man, make this interface with the c++ vl_feat
            direct OR write better python wrappers.

"""

import math
import numpy as np
#from vlfeat import vl_dsift
import cv2
from imdescrip.imd_utils.image import imread_resize, rgb2gray
from imdescrip.imd_utils.progress import Progress
from skimage import (color,io,segmentation)


def training_patches (imnames, npatches, psize, maxdim=None, verbose=False):
    """ Extract SIFT patches from images for dictionary training

    Arguments:
        imnames: A list of image names from which to extract training patches.
        npatches: The number (int) of patches to extract from the images
        maxdim: The maximum dimension of the image in pixels. The image is
            rescaled if it is larger than this. By default there is no scaling.
        psize: A int of the size of the square patches to extract
        verbose: bool, print progress bar

    Returns:
        An np.array (npatches, 128) of SIFT descriptors. NOTE, the actual
        npatches found may be slightly more or less than that requested.

    Note:
        The SIFT descriptors output by vlfeat are [0, 255] integers!

    """

    nimg = len(imnames)
    ppeimg = int(round(float(npatches)/nimg))
    plist = []
    bsize = __patch2bin(psize)

    sift = cv2.xfeatures2d.SIFT_create()
    if verbose == True:
        print('Extracting SIFT patches from images...')

    # Set up progess updates
    progbar = Progress(nimg, title='Extracting patches', verbose=verbose)

    # Get patches
    for i, ims in enumerate(imnames):

        # Read in and resize the image -- convert to gray if needed
        img = imread_resize(ims, maxdim)
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = img[...,np.newaxis]

        # Extract the patches
        spaceing = max(1, int(math.floor(math.sqrt( \
                        float(np.prod(img.shape))/ppeimg))))
        kpDense = [cv2.KeyPoint(x, y, psize) for y in np.arange(0, img.shape[0], spaceing)  for x in np.arange(0, img.shape[1], spaceing)]
        #kpDense = [cv2.KeyPoint(x, y, 1) for y in range(0, img.shape[0], 1)  for x in range(0, img.shape[1], 1)]
        kps,des = sift.compute(img,kpDense)
        plist.append(np.asfortranarray(des.T))

        progbar.update(i)

    progbar.finished()
    return np.asarray(plist).transpose(0,1,2).reshape(-1,128)


def DSIFT_patches (image, psize, pstride):
    """ Extract a grid of (overlapping) SIFT patches from an image

    This function extracts SIFT descriptors from square patches in an
    overlapping, dense grid from an image.

    Arguments:
        image: np.array (rows, cols, channels) of an image (in memory)
        psize: int the size of the square patches to extract, in pixels.
        pstride: int the stride (in pixels) between successive patches.

    Returns:
        patches: np.array (npatches, 128) SIFT descriptors for each patch
        centresx: np.array (npatches) the centres (column coords) of the patches
        centresy: np.array (npatches) the centres (row coords) of the patches

    Note:
        The SIFT descriptors output by vlfeat are [0, 255] integers!

    """

    if pstride < 1:
        raise ValueError('pstride needs to be 1 pixel or more!')

    if image.ndim > 2:
        image = cv2.cvtColor((image*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kpDense = [cv2.KeyPoint(x, y,psize) for y in np.arange(0, image.shape[0], pstride)  for x in np.arange(0, image.shape[1], pstride)]
    kps,des = sift.compute(image,kpDense)
    norms = np.linalg.norm(des,axis=1).reshape(-1,1)
    norms[norms == 0] = 1 #For all zeros descriptors
    des = des/norms
    des[np.where(des >= 0.2)] = 0.2
    norms = np.linalg.norm(des,axis=1).reshape(-1,1)
    norms[norms == 0] = 1 #For all zeros descriptors
    des = des/norms
    #xy, desc = vl_dsift(np.float32(image), step=pstride,
    #                size=__patch2bin(psize))

    x = np.asarray([kpDense[i].pt[0] for i in range(len(kpDense))])
    y = np.asarray([kpDense[i].pt[1] for i in range(len(kpDense))])
    xy = np.vstack((x,y))

    return des, xy[0,:], xy[1,:]


def __patch2bin (psize):
    """ Convert image patch size to SIFT bin size as expected by VLFeat. """

    return int(round(float(psize)/4)) + 1
