import geometry  # local module
import data_generator  # local module

import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

K.set_image_data_format('channels_first')


class DirectoryIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    Builds on keras.preprocessing.image.NumpyArrayIterator
    but does not directly subclass

    Inputs
    ------
    load_fn: takes in an index (correspond to the y index)
             and returns an image. This is designed to allow
             you to read in images from disk, but it isn't required

    y: Numpy array of targets data
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
    num_channels: the number of channels of the image `load_fn`
        will return
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

    def __init__(self, load_fn, y, image_data_generator,
                 num_channels,
                 batch_size=32, shuffle=False, seed=None,
                 output_image_shape=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.load_fn = load_fn
        self.epoch_size = len(y)
        self.num_channels = num_channels

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
        super(DirectoryIterator, self).__init__(self.epoch_size, batch_size,
                                                shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_shape = [None]*4
        batch_shape[0] = len(index_array)
        batch_shape[self.channels_axis] = self.num_channels
        image_axes = (2, 3) if self.channels_axis == 1 else (3, 4)
        if self.output_image_shape is None:
            for image_axis in image_axes:
                batch_shape[image_axis] = self.x.shape[image_axis]
        else:
            for i, image_axis in enumerate(image_axes):
                batch_shape[image_axis] = self.output_image_shape[i]
        batch_x = np.zeros(batch_shape,
                           dtype=K.floatx())
        for i, j in enumerate(index_array):

            x = self.load_fn(j)
            x = self.image_data_generator.random_transform(
                x.astype(
                    K.floatx()
                    )
                )
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format
                    )
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class ImageDataGeneratorLoadable(data_generator.ImageDataGenerator):
    """Like data_generatory.ImageDataGenerator, except this loads in images
    using an arbitrary load function (passed to `flow`) rather than assuming
    images are preloaded into memory.
    """

    def flow(self, load_fn, num_channels, y=None, batch_size=32, shuffle=True,
             seed=None, save_to_dir=None, save_prefix='', save_format='png'):
        return DirectoryIterator(
            load_fn, y, self,
            num_channels,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            output_image_shape=self.output_image_shape,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Look at data_generator.ImageDataGenerator to adapt this")
