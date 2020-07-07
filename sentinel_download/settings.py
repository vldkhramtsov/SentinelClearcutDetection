from utils import path_exists_or_create

MAXIMUM_DATES_REVIEWED_FOR_TILE = 220
MAXIMUM_DATES_STORE_FOR_TILE = 2
MAXIMUM_EMPTY_PIXEL_PERCENTAGE = 0.05
MAXIMUM_CLOUD_PERCENTAGE_ALLOWED = 50

PIECE_WIDTH = 56
PIECE_HEIGHT = 56

DOWNLOADED_IMAGES_DIR = path_exists_or_create('data/source_images/')
MODEL_TIFFS_DIR = path_exists_or_create('data/model_tiffs')
PIECES_DIR = path_exists_or_create('data/output')
POLYS_PATH = '../data/time-dependent'