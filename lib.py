import numpy as np

# !/usr/bin/env python
import numpy as np

class TileCoder:
  def __init__(self, tilings, tiles, value_limits,
               offset=lambda n: 2 * np.arange(n) + 1):
    tiles_per_dim = [tiles / len(value_limits)] * len(value_limits)
    self.tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=int)+1
    self.tilings = tilings
    self.tiles_per_tiling = np.prod(self.tiling_dims)
    self.offsets = offset(len(tiles_per_dim)) * \
                   np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
    self.limits = np.array(value_limits)
    self.norm_dims = np.array(tiles_per_dim) / (self.limits[:, 1] - self.limits[:, 0])
    self.tile_base_ind = np.prod(self.tiling_dims) * np.arange(tilings)
    self.hash_vec = np.array(
      [np.prod(self.tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
    self.n_tiles = tilings * np.prod(self.tiling_dims)

  def encode(self, state):
    off_coords = ((state - self.limits[:, 0]) * self.norm_dims + self.offsets).astype(int)
    active_tiles = self.tile_base_ind + np.dot(off_coords, self.hash_vec)
    # Create a binary vector for the active tiles
    binary_vector = np.zeros(self.n_tiles, dtype=int)
    binary_vector[active_tiles] = 1
    return binary_vector

  @property
  def total_tiles(self):
    return self.n_tiles


if __name__ == '__main__':
  num_tilings = 2
  tiles_per_dim = 5
  limits = [(-1.2, 0.6), (-0.07, 0.07)]
  state = [-0.6, 0]

  tile_coder = TileCoder(num_tilings, tiles_per_dim, limits)
  encoded_state = tile_coder.encode(state)
  print(encoded_state, encoded_state.shape)


