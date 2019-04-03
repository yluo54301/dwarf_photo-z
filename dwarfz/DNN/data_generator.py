import numpy as np
from keras import backend as K
K.set_image_data_format('channels_first')

import geometry # local module

from keras.preprocessing.image import Iterator

class ArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    Builds on keras.preprocessing.image.NumpyArrayIterator
    but does not directly subclass
    
    Inputs
    ------
    x: Numpy array of input data 
        if `channels_first`, then shape N_img x N_channels x height x width
        if `channels_last`,  then shape N_img x height x width x N_channels

    y: Numpy array of targets data
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
    batch_size: Integer, size of a batch.
    shuffle: Boolean, whether to shuffle the data between epochs.
    seed: Random seed for data shuffling.
    output_image_shape: None, or list/tuple
        if list/tuple, should have ndim=2, of (height x width)
        if None, defaults to input image shape
    data_format: String, one of `channels_first`, `channels_last`.
    save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 output_image_shape=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        self.channels_axis = channels_axis

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.output_image_shape = output_image_shape
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_shape = [None]*4
        batch_shape[0] = len(index_array)
        batch_shape[self.channels_axis] = self.x.shape[self.channels_axis]
        image_axes = (2,3) if self.channels_axis==1 else (3,4)
        if self.output_image_shape is None:
            for image_axis in image_axes:
                batch_shape[image_axis] = self.x.shape[image_axis]
        else:
            for i, image_axis in enumerate(image_axes):
                batch_shape[image_axis] = self.output_image_shape[i]
                        
        batch_x = np.zeros(batch_shape,
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    This is adapted from `keras.preprocessing.image.ImageDataGenerator`
    but is not a direct subclass.
    
    Major differences:
        - accepts images with arbitrary number of channels, not just 1, 3 or 4
        - uses my custom data augmentation. In particular I start with a large image,
          apply desired transformations, then crop the image to a smaller size.
          This allows me to better deal with out-of-bounds issues with transformations,
          without needing to rely on things like `fill_mode`

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        with_reflection_x: boolean
            whether to randomly flip images horizontally before other transformations.
        with_reflection_y: boolean
            whether to randomly flip images vertically before other transformations.
        with_rotation: boolean
            whether to rotate the image randomly (after reflection, before translation)
        width_shift_range: fraction of total width. Shift will be randomly 
            chosen within +/- this range. (translation applied after rotation)
        height_shift_range: fraction of total height. Shift will be randomly 
            chosen within +/- this range. (translation applied after rotation)
        shear_range: shear intensity (shear angle in degrees).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [z**-1, z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
            (shifts all values up by a constant, while clipping to the original min/max)
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'constant'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        rescale: rescaling factor. If None or 0, no rescaling is applied to
            the pixel values, otherwise we multiply the data by the value 
            provided. This is applied after the `preprocessing_function` 
            (if any provided) but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run *before* any pixel-wise modification on it,
            but after affine transformations/channel-wise shifts.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        postprocessing_function: function that will be implied on each input.
            The function will run *after* all other modifications and transformations.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and *can* output a Numpy tensor with *a different shape*.
        output_image_shape: None, or list/tuple
            if list/tuple, should have ndim=2, of (height x width)
            if None, defaults to input image shape
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 with_reflection_x=False,
                 with_reflection_y=False,
                 with_rotation=False,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0,
                 zoom_range=1,
                 channel_shift_range=0,
                 fill_mode='constant',
                 cval=0.,
                 rescale=None,
                 preprocessing_function=None,
                 postprocessing_function=None,
                 output_image_shape=None,
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.with_reflection_x = with_reflection_x
        self.with_reflection_y = with_reflection_y
        self.with_rotation = with_rotation
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.postprocessing_function = postprocessing_function
        self.output_image_shape = output_image_shape

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1/zoom_range, zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return ArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            output_image_shape=self.output_image_shape,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= np.std(x, keepdims=True) + 1e-7

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')

        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
                
        if self.postprocessing_function:
            x = self.postprocessing_function(x)
        return x

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.with_reflection_x:
            reflect_x = np.random.choice((True, False))
        else:
            reflect_x = False
            
        if self.with_reflection_y:
            reflect_y = np.random.choice((True, False))
        else:
            reflect_y = False
        
        if self.with_rotation:
            rotation_degrees = np.random.uniform(0, 360)
        else:
            rotation_degrees = 0

        if self.height_shift_range:
            translate_dx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * h
        else:
            translate_dx = 0

        if self.width_shift_range:
            translate_dy = np.random.uniform(-self.width_shift_range, self.width_shift_range) * w
        else:
            translate_dy = 0

        if self.shear_range:
            shear_degrees = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear_degrees = 0

        if (self.zoom_range[0] != 1) or (self.zoom_range[1] != 1):
            # log uniform distributed (since it's a scale-like parameter)
            zoom_x, zoom_y = np.exp(np.random.uniform(np.log(self.zoom_range[0]), 
                                                      np.log(self.zoom_range[1]),
                                                      2)) 
        else:
            zoom_x, zoom_y = 1, 1

            
        transform_matrix = geometry.get_full_transform(h, w,
                                              reflect_x=reflect_x,
                                              reflect_y=reflect_y,
                                              rotation_degrees=rotation_degrees,
                                              translate_dx = translate_dx,
                                              translate_dy = translate_dy,
                                              shear_degrees = shear_degrees,
                                              zoom_x = zoom_x,
                                              zoom_y = zoom_y,
                                             )

        transform_matrix = geometry.transform_matrix_offset_center(transform_matrix, h, w)
        x = geometry.apply_transform_new(x, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval,
                               order=0,
                               )

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)

        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)
