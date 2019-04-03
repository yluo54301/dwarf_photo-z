import os
import glob
from astropy.io import fits
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import geometry # local module

images_dir = "../data/galaxy_images_training/quarry_files/"

def get_image(HSC_id, band, images_dir=images_dir):
    """Reads a single image from its fits file
    Inputs
    ------
        HSC_id : int
        band : str
            valid options: 'g', 'r', 'i' , 'z', 'y'
            case-insensitive
    """
    
    image_pattern = os.path.join(images_dir,"{}-cutout-HSC-{}*.fits".format(
        str(HSC_id),
        band.upper()))
    image_matches = glob.glob(image_pattern)
    
    if len(image_matches) == 0:
        raise FileNotFoundError("No files match: {}".format(image_pattern))
    elif len(image_matches) > 1:
        raise RuntimeError("Found too many files matching: {}".format(image_pattern))
        
    image_filename = image_matches[0]
    
    hdulist = fits.open(image_filename)
    image = hdulist[1].data
    flux_mag_0 = hdulist[0].header['FLUXMAG0']
    return image, flux_mag_0

def image_plotter(image, reverse_cmap=False, no_ticks=False):
    cmap = "gray_r" if reverse_cmap else "gray"
    
    plt.imshow(image,
               cmap=plt.get_cmap(cmap)
               )
    
    if no_ticks:
        ax = plt.gca()
        ax.get_xaxis().set_major_locator(plt.NullLocator())
        ax.get_yaxis().set_major_locator(plt.NullLocator())

def scale(x, fluxMag0):
    """adapted from https://hsc-gitlab.mtk.nao.ac.jp/snippets/23

    To see more about asinh magnitude system, see : [Lupton, Gunn and Szalay (1999)](http://iopscience.iop.org/article/10.1086/301004/meta) used for SDSS. (It's expliticly given in the [SDSS Algorithms](http://classic.sdss.org/dr7/algorithms/fluxcal.html) documentation as well as [this overview page](https://ned.ipac.caltech.edu/help/sdss/dr6/photometry.html#asinh)).
    """
    mag0 = 19
    scale = 10 ** (0.4 * mag0) / fluxMag0
    x *= scale

    u_min = -0.05
    u_max = 2. / 3.
    u_a = np.exp(10.)

    x = np.arcsinh(u_a*x) / np.arcsinh(u_a)
    x = (x - u_min) / (u_max - u_min)

    return x

def get_cutout(image, cutout_size):
    """Takes an image, and cuts it down to `cutout_size x cutout_size`
    It only affects the final two dimensions of the array, so
    you can easily deal with multiple images / multiple channels
    simply by setting up the array with shape (n_channels, height, width)
    
    Inputs
    ------
    image : np.ndarray (ndim >= 2)
    cutout_size : int
        the [maximum] number of pixels you want in each dimension
        of the final image.



    Notes
    -----
    If `cutout_size` is large than an image dimension, it'll silently
    keep *the entire* range of that dimension. This won't have any 
    side-effects on the other dimension of non-square images.
"""
    image_shape = image.shape[-2:]

    
    center_index = (image_shape[0]//2, image_shape[1]//2)

    # Check that these are actually x and y ordered
    min_x = center_index[0] - (cutout_size//2)
    max_x = center_index[0] + (cutout_size//2)
    min_y = center_index[1] - (cutout_size//2)
    max_y = center_index[1] + (cutout_size//2)
    
    if cutout_size % 2 == 1:
        # handle odd number of pixels
        max_x += 1
        max_y += 1

    cutout = image[..., min_x:max_x, min_y:max_y]
    
    return cutout

def transform_0_1(X):
    return (X - X.min())/(X.max() - X.min())

def transform_plotter(
    X,
    reflect_x = False,
    rotation_degrees=45, 
    dx_after = 0,
    dy_after = 0,
    shear_degrees=0,
    zoom_x = 1,
    crop=False,
    color=True,
):
    x_tmp = X[0].copy()
    h, w = x_tmp.shape[1], x_tmp.shape[2]
    
    transform_matrix = geometry.get_full_transform(h, w,
                                          reflect_x=reflect_x,
                                          rotation_degrees=rotation_degrees,
                                          translate_dx=dx_after,
                                          translate_dy=dy_after,
                                          shear_degrees=shear_degrees,
                                          zoom_x = zoom_x,
                                         )
    

    x_tmp = x_tmp[0:3]

    result = geometry.apply_transform_new(x_tmp, transform_matrix, 
                            channel_axis=0, fill_mode="constant", cval=np.max(x_tmp),
                                order=2)

    if crop:
        result = np.array([get_cutout(channel, channel.shape[0]//2)
                           for channel in result])

    with mpl.rc_context(rc={"figure.figsize": (10,6)}):
        if color:
            plt.imshow(transform_0_1(result.transpose(1,2,0)))
        else:
            plt.imshow(result[0], 
                       cmap=plt.get_cmap("viridis"),
                       vmin=0, vmax=np.max(x_tmp[0]),
                       )


