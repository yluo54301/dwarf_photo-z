import numpy as np
import keras
import scipy.ndimage as ndi


def get_translation_matrix(dx, dy):
    """Get the affine transformation matrix for a translation (shift)
    
    Inputs
    ------
    dx : numeric
        how many pixels to shift the image in x (can be fractional)
    dy : numeric
        how many pixels to shift the image in y (can be fractional) 

    Returns
    -------
    translation_matrix : np.ndarray (shape 3x3, dtype=float)
        a transformed version of `matrix` account for the centering shifts
        
    Notes
    -----
    To apply *after* another matrix, you should do:
        `np.dot(other_matrix, translation_matrix)`
    """
    translation_matrix = np.array([[1, 0, -dy], [0, 1, -dx], [0, 0, 1]])
    return translation_matrix


def transform_matrix_offset_center(matrix, x, y):
    """Takes a transformation matrix (e.g. rotation) and adjusts it
    so that the image center is at the origin before the transformation,
    and then shifts it back after the center.
    
    Inputs
    ------
    matrix : np.ndarray (shape 3x3, dtype=float)
        a matrix defining an affine transformation
    x : numeric
        size of image in axis 0 (width)
    y : numeric
        size of image in axis 1 (height)

    Returns
    -------
    transform_matrix : np.ndarray (shape 3x3, dtype=float)
        a transformed version of `matrix` account for the centering shifts

    """
    o_x = float(x) / 2 - .5
    o_y = float(y) / 2 - .5
    offset_matrix = get_translation_matrix(-o_x, -o_y)
    reset_matrix = get_translation_matrix(o_x, o_y)
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
    
def get_reflection_matrix(axis):
    """Get the affine transform matrix for a reflection about the origin
    for a specific axis
    
    Inputs
    ------
    axis : Optional(integer)
        the axis that you want flipped
        
    Returns
    -------
    reflection_matrix : np.ndarray (shape 3x3, dtype=int)
    
    Notes
    -----
    see `keras.preprocessing.image.flip_axis()` for alternative
    """
    if axis == 0:
        reflection_matrix = np.diag((-1, 1, 1))
    elif axis==1:
        reflection_matrix = np.diag((1, -1, 1))
    else:
        raise ValueError("`axis` must be 0 or 1, not {}".format(axis))
                
    return reflection_matrix

def get_centered_reflection_matrix(axis, h, w):
    """Get the affine transform matrix for a reflection about the image center
    for a specific axis
    
    Inputs
    ------
    axis : Optional(integer)
        the axis that you want flipped
    x : numeric
        size of image in axis 0 (width)
    y : numeric
        size of image in axis 1 (height)
        
    Returns
    -------
    centered_reflection_matrix : np.ndarray (shape 3x3, dtype=int)
    
    Notes
    -----
    see `keras.preprocessing.image.flip_axis()` for alternative
    """
    reflection_matrix = get_reflection_matrix(axis)
        
    centered_reflection_matrix = transform_matrix_offset_center(reflection_matrix, w, h)
        
    return centered_reflection_matrix


def get_rotation_matrix(rotation_degrees):
    """Get the affine transform matrix for a rotation about the origin
    
    Inputs
    ------
    rotation_degrees: numeric
        Rotation angle, in degrees, clockwise
    
    Returns
    -------
    rotation_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """
    theta = (np.pi / 180) * rotation_degrees # convert to radians
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
    
    return rotation_matrix

def get_centered_rotation_matrix(rotation_degrees, h, w):
    """Get the affine transform matrix for a rotation about the image center
    
    Inputs
    ------
    rotation_degrees: numeric
        Rotation angle, in degrees, clockwise
    h : int
        height of image, in pixels (axis=0)
    w : int
        width of image, in pixels (axis=1)
    
    Returns
    -------
    centered_rotation_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """
    rotation_matrix = get_rotation_matrix(rotation_degrees)
    centered_rotation_matrix = transform_matrix_offset_center(rotation_matrix, w, h)
    
    return centered_rotation_matrix

def get_shear_matrix(shear_degrees):
    """Get the affine transform matrix for a shear w.r.t the origin
    
    Inputs
    ------
    shear_degrees: numeric
        shear angle, in degrees, clockwise
    
    Returns
    -------
    shear_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """
    theta = (np.pi / 180) * shear_degrees # convert to radians
    
    shear_matrix = np.array([[1, -np.sin(theta), 0],
                            [0,   np.cos(theta), 0],
                            [0,   0,             1]])
     
    return shear_matrix


def get_centered_shear_matrix(shear_degrees, h, w):
    """Get the affine transform matrix for a shear w.r.t the image center
    
    Inputs
    ------
    shear_degrees: numeric
        shear angle, in degrees, clockwise
    h : int
        height of image, in pixels (axis=0)
    w : int
        width of image, in pixels (axis=1)
    
    Returns
    -------
    centered_shear_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """    
    shear_matrix = get_shear_matrix(shear_degrees)
    centered_shear_matrix = transform_matrix_offset_center(shear_matrix, w, h)

    return centered_shear_matrix


def get_zoom_matrix(zoom_x, zoom_y):
    """Get the affine transform matrix for a zoom centered on the origin
    
    Inputs
    ------
    zoom_x: numeric
        zoom in on the x axis by a factor of zoom_x
    zoom_y: numeric
        zoom in on the x axis by a factor of zoom_x
    
    Returns
    -------
    zoom_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """    
    zoom_matrix = np.array([[zoom_y, 0,      0],
                            [0,      zoom_x, 0],
                            [0,      0,      1]])
    
    return zoom_matrix

def get_centered_zoom_matrix(zoom_x, zoom_y, h, w):
    """Get the affine transform matrix for a zoom centered on image center
    
    Inputs
    ------
    zoom_x: numeric
        zoom in on the x axis by a factor of zoom_x
    zoom_y: numeric
        zoom in on the x axis by a factor of zoom_x
    h : int
        height of image, in pixels (axis=0)
    w : int
        width of image, in pixels (axis=1)
    
    Returns
    -------
    centered_zoom_matrix : np.ndarray (shape 3x3, dtype=float)
        affine transform matrix
    """    
    zoom_matrix = get_zoom_matrix(zoom_x, zoom_y)
    centered_zoom_matrix = transform_matrix_offset_center(zoom_matrix, w, h)

    return centered_zoom_matrix

def apply_transform_new(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    order=0):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def get_full_transform(
    h, w,
    reflect_x = False,
    reflect_y = False,
    rotation_degrees = 0,
    translate_dx = 0,
    translate_dy = 0,
    shear_degrees = 0,
    zoom_x = 1,
    zoom_y = 1,
    ):
    """Get the combined affine transform matrix for reflection, rotation and translation
    
    Inputs
    ------
    h : int
        height of the image (axis=0)
    w : int
        width of the image (axis=1)
    reflect_x : Optional(bool)
    reflect_y : Optional(bool)
    rotation_degrees : Optional(numeric)
        rotation angle (clockwise) after reflection
    translate_dx : Optional(numeric)
        number of pixels to shift in x after reflection and rotation 
        (can be fractional)
    translate_dy : Optional(numeric)
        number of pixels to shift in y after reflection and rotation 
        (can be fractional)
    shear_degrees : Optional(numeric)
    zoom_x : Optional(numeric)
    zoom_y : Optional(numeric)

        
        
    Returns
    -------
    centered_transform_matrix : np.ndarray (shape 3x3, numeric)
        the matrix defining the affine transform
    """

    transform_matrix = np.eye(3, dtype=int)
    
    if reflect_x:
        reflection_matrix_x = get_reflection_matrix(1)
        transform_matrix = np.dot(transform_matrix, reflection_matrix_x)

    if reflect_y:
        reflection_matrix_y = get_reflection_matrix(0)
        transform_matrix = np.dot(transform_matrix, reflection_matrix_y)
        
    if rotation_degrees not in (0, 360):
        rotation_matrix = get_rotation_matrix(rotation_degrees)
        transform_matrix = np.dot(transform_matrix, rotation_matrix)
    
    if (translate_dx != 0) or (translate_dy != 0):
        # should this be at this point, or after I apply the shear + zoom?
        translation_matrix = get_translation_matrix(translate_dx, translate_dy)
        transform_matrix = np.dot(transform_matrix, translation_matrix)
    
    if shear_degrees != 0:
        shear_matrix = get_shear_matrix(shear_degrees)
        transform_matrix = np.dot(transform_matrix, shear_matrix)
    
    if (zoom_x != 1) or (zoom_y != 1):
        zoom_matrix = get_zoom_matrix(zoom_x, zoom_y)
        transform_matrix = np.dot(transform_matrix, zoom_matrix)
        
    if not (transform_matrix == np.eye(3)).all():
        centered_transform_matrix = transform_matrix_offset_center(transform_matrix, w, h)
    else:
        centered_transform_matrix = np.eye(3)
    
    return centered_transform_matrix
    
def create_random_transform_matrix(h, w,
                                   include_reflection=True,
                                   include_rotation=True,
                                   translation_size=0.0,
                                   seed=None,
                                   verbose=False,
                                  ):
    """Create a random transformation matrix
    Inputs
    ------
    h : int
        height of the image (axis=0)
    w : int
        width of the image (axis=1)
    include_reflection : Optional(bool)
        only includes an initial flip of the x-axis
        Assumes flip ~ Bernoulli(p=0.5)
    include_rotation : Optional(bool)
        Assumes rotation angle ~ Uniform(0, 360) degrees
    translation_size : Optional(float)
        Fraction of the image that the image can be translated
        Assumes translation[axis=i] ~ Uniform[-translation_size*shape[i], +translation_size*shape[i]]
    seed : Optional(int or None)
        random seed, or None if unseeded
    verbose : Optional(bool)
        
    Returns
    -------
    transfrom_matrix : np.ndarray (shape 3x3, numeric)
        the matrix defining the affine transform
    """
    np.random.seed(seed)
    
    if include_reflection:
        reflect_x = np.random.choice((True, False))
        reflect_y = np.random.choice((True, False))
    else:
        reflect_x = False
        reflect_y = False
    
    if include_rotation:
        rotation_degrees = np.random.random()*360
    else:
        rotation_degrees = 0
        
        
    translation_dx = (np.random.random() - 0.5) * 2 * translation_size * w
    translation_dy = (np.random.random() - 0.5) * 2 * translation_size * h

    
    if verbose:
        print("reflection axis = {}".format(reflection_axis))
        print("rotation angle = {:.1f}°".format(rotation_degrees))
        print("translation = ({:.1f}, {:.1f}) px".format(translation_dx,
                                                         translation_dy,))
        
    transform_matrix = get_full_transform(h, w,
                                          reflect_x = reflect_x,
                                          reflect_y = reflect_y,
                                          rotation_degrees = rotation_degrees,
                                          translate_dx = translation_dx,
                                          translate_dy = translation_dy,
                                         )
    
    return transform_matrix
    

