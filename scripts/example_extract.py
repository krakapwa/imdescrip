#! /usr/bin/env python

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

import glob, sys
import pickle as pk
from imdescrip.extractor import extract_smp
from imdescrip.descriptors.ScSPM import ScSPM
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Make a list of images
imgdir   = '/home/laurent.lejeune/medical-labeling/data/Dataset2/input-frames/'
outdir   = '/home/laurent.lejeune/medical-labeling/data/Dataset2/precomp_descriptors/'
filelist = sorted(glob.glob(imgdir + '*.png'))

start = 140
end = 260

filelist = filelist[start:end]

# Train a dictionary
desc = ScSPM(dsize=512, compress_dim=None)

#D1
desc.learn_dictionary(filelist, npatches=200000, niter=1000)

# Save the dictionary
with open(os.path.join(outdir,'ScSPM.p'), 'wb') as f:
    pk.dump(desc, f, protocol=2)

n = 100
arr = np.logspace(-1,5,10)
bins = np.linspace(np.min(arr),np.max(arr),n)
arr_qt = np.digitize(arr,bins)
