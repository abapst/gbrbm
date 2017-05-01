import numpy as np

try:
    import PIL.Image as Image
except ImportError:
    import Image

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def avg(x):
    return sum(x)/len(x)


def _expand_line(line, k=5):
    expanded = [0] * (len(line) * k)
    for i, el in enumerate(line):
        if float(el) != 0.:
            el = float(el)
            expanded[(i*k) + int(round(el)) - 1] = 1
    return expanded


def expand(data, k=5):
    new = []
    for m in data:
        new.extend(_expand_line(m.tolist()))

    return np.array(new).reshape(data.shape[0], data.shape[1] * k)


def revert_expected_value(m, k=5, do_round=True):
    mask = list(range(1, k+1))
    vround = np.vectorize(round)

    if do_round:
        users = vround((m.reshape(-1, k) * mask).sum(axis=1))
    else:
        users = (m.reshape(-1, k) * mask).sum(axis=1)

    return np.array(users).reshape(m.shape[0], m.shape[1] / k)


def scale_to_unit_interval(ndar, mask, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0/(ndar.max() + eps)
    ndar *= mask
    return ndar


def scale_rows_to_unit_interval(ndar, eps=1e-8):
    """ Scales all rows in the ndarray ndar to be between 0 and 1 """
    for i in xrange(ndar.shape[0]):
        row = ndar[i,:]
        row -= row.min()
        row *= 1.0/(row.max() + eps)
        ndar[i,:] = row
    return ndar
	
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       mask=None,
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4

      if mask is None:
          mask = [np.ones(c.shape) for c in X[:3]]
          mask.append(None)
          mask=tuple(mask)

      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in range(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, mask[i], scale_rows_to_unit_interval, output_pixel_vals)
      return out_array
  else:
      if mask is None:
          mask = np.ones(X.shape)

      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

      for tile_row in range(tile_shape[0]):
          for tile_col in range(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape),mask[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array

def standardize_images(X):
    mu = np.mean(X,axis=0)
    std = np.std(X,axis=0) 
    X = (X - mu)/std

    return mu,std

class TileRows(object):
    def __init__(
        self,
        img_shape,
        tile_shape,
        tile_spacing=(1,1),
        gray=True
    ):   
        self.gray = gray
        self.img_shape = img_shape
        self.tile_shape = tile_shape
        self.tile_spacing = tile_spacing

    def imsave(self,X,filename,mask=None):

        if self.gray == False:
            if len(X.shape) > 1:
                X = np.split(X,3,axis=1)
            else:
                X = np.split(X,3)
            X.append(None)
            X = tuple(X)
            if mask is not None:
                mask = np.split(mask,3,axis=1)
                mask.append(None)
                mask = tuple(mask)

        image_data = tile_raster_images(
            X,
            self.img_shape,
            self.tile_shape,
            self.tile_spacing,
            mask
        )
        image = Image.fromarray(image_data)
        image.save(filename)
