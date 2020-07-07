"""
Script helpers
"""
import os
from enum import Enum

def read_labeled_tiles(file):
    tiles = []
    with open(file) as tiles_list:
        tiles.extend([line[:-1] for line in tiles_list])
    unique_tilename = set([tile.split('_')[1][1:] for tile in tiles])
    tiles_by_tilename = {tilename: [tile for tile in tiles if tilename in tile] for tilename in unique_tilename}
    return tiles_by_tilename



def path_exists_or_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Bands(Enum):
    TCI = 'TCI'
    B04 = 'B04'
    B08 = 'B08'
    B8A = 'B8A'
    B11 = 'B11'
    B12 = 'B12'

def date_limit(date, shp):
    return date >= shp['img_date'].min() and date <= shp['img_date'].max()

def image_non_zero(image_array):
    return np.count_nonzero(image_array) > image_array.size * (1 - MAXIMUM_EMPTY_PIXEL_PERCENTAGE)