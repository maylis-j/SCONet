import numpy as np

def split_long_rle_lengths(values, lengths, dtype=np.int64):
  max_length = np.iinfo(dtype).max
  lengths = np.asarray(lengths)
  repeats = lengths // max_length
  if np.any(repeats):
    repeats += 1
    remainder = lengths % max_length
    values = np.repeat(values, repeats)
    lengths = np.empty(len(repeats), dtype=dtype)
    lengths.fill(max_length)
    lengths = np.repeat(lengths, repeats)
    lengths[np.cumsum(repeats)-1] = remainder
  elif lengths.dtype != dtype:
    lengths = lengths.astype(dtype)
  return values, lengths

def dense_to_rle(dense_data, dtype=np.uint8):
  dense_data = dense_data.flatten()
  n = len(dense_data)
  starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
  lengths = np.diff(np.r_[starts, n])
  values = dense_data[starts]
  values, lengths = split_long_rle_lengths(values, lengths, dtype=dtype)
  out = np.stack((values, lengths), axis=1)
  return out.flatten()

def rle_to_dense(rle_data):
  values, counts = np.reshape(rle_data, (-1, 2)).T
  return np.repeat(values, counts)
