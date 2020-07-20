import os
import cv2
import csv
import random
import imageio
import datetime

import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp

from tqdm import tqdm
from random import random
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
from skimage.transform import match_histograms
from rasterio.plot import reshape_as_image as rsimg

from utils import date_limit, path_exists_or_create
import settings


def date(filename):
    # S2B_MSIL2A_20190830T083609_N0213_R064_T36UYA_20190830T123940
    dt_name = filename.split('_')[-1][:8]
    date_part = dt_name[:4] + '-' + dt_name[4:6] + '-' + dt_name[6:8]
    return datetime.datetime.strptime(date_part, '%Y-%m-%d')


def getdates(data_path):
    tiles_description = [[name, date(name)] for name in os.listdir(data_path)]
    return tiles_description


def readtiff(filename):
    src = rs.open(filename)
    return rsimg(src.read()), src.meta


class ImageDifference:
    """
    ImageDifference class to build a set of differences between pairs of images divided into pieces.
    """
    def __init__(self):
        # path to pieces
        self.data_path = settings.PIECES_DIR
        # path to store the differences
        self.save_path = settings.DIFF_PATH
        # path to polygons
        self.polys_path = settings.POLYS_PATH
        # standard directory name for pieces of images
        self.images_path = 'images'
        # standard directory name for pieces of masks
        self.masks_path = 'masks'
        # standard directory name for pieces of clouds
        self.clouds_path = 'clouds'
        self.width = settings.PIECE_WIDTH
        self.height = settings.PIECE_HEIGHT
        # the maximum number of previous images to plot the difference
        # e.g. days_limit = 3 means that for a current image we build 
        # the differences at 15 days at maximum (3 * settings.SENTINEL_DELTA_DAYS)
        self.days_limit = settings.NEIGHBOURS
        # fractions of training, testing, and validation pieces, 
        # split by their positions at tile (X, Y)
        self.train_size = settings.TRAIN_SIZE
        self.test_size = settings.TEST_SIZE
        self.valid_size = settings.VALID_SIZE
    
    def diff(self, images, width, height):
        img_current, img_previous = images['current'], images['previous']
        dim = (width, height)
        I_current = np.clip(cv2.resize(img_current.astype(np.float32),
                            dim, interpolation = cv2.INTER_CUBIC),
                            0, 255)
        I_previous = np.clip(cv2.resize(img_current.astype(np.float32),
                             dim, interpolation = cv2.INTER_CUBIC),
                             0, 255)
        difference = ( (I_current - I_previous) / (I_current + I_previous) )
        difference = ((difference + 1) * 127).astype(np.uint8)
        return np.concatenate((difference,
                               I_current.astype(np.uint8),
                               I_previous.astype(np.uint8)),
                               axis=-1)

    def imgdiff(self, tile_current, tile_previous, diff_path, writer):
        
        def path_to_image(tile, path, x, y, ext='.png'):
            return os.path.join(self.data_path, tile, path, tile + '_' + x + '_' + y + ext)
        
        pieces = os.listdir(f"{self.data_path}/{tile_current}/{self.images_path}")
        xs = [piece.split('_')[-2:][0] for piece in pieces]
        ys = [piece.split('_')[-2:][1].split('.')[0] for piece in pieces]

        is_path_tile = {}
        for idx in range(len(xs)):
            images = {}
            masks = {}
            clouds = {}

            is_path_tile['current'] = os.path.exists(path_to_image(tile_current, self.images_path, xs[idx], ys[idx], ext='.tiff'))
            is_path_tile['previous'] = os.path.exists(path_to_image(tile_previous, self.images_path, xs[idx], ys[idx], ext='.tiff'))
            
            if is_path_tile['current'] and is_path_tile['previous']:
                images['current'], meta = readtiff(path_to_image(tile_current, 
                                             self.images_path,
                                             xs[idx], ys[idx],
                                             ext='.tiff')) 
                images['previous'], _ = readtiff(path_to_image(tile_current, 
                                                 self.images_path,
                                                 xs[idx], ys[idx],
                                                 ext='.tiff')) 
                
                mask_path = path_to_image(tile_current, self.masks_path, xs[idx], ys[idx])
                masks['current'] = imageio.imread(mask_path)
                mask_path = path_to_image(tile_previous, self.masks_path, xs[idx], ys[idx])
                masks['previous'] = imageio.imread(mask_path)

                cloud_path = path_to_image(tile_current, self.clouds_path, xs[idx], ys[idx])
                clouds['current'] = imageio.imread(cloud_path) / 255
                cloud_path = path_to_image(tile_previous, self.clouds_path, xs[idx], ys[idx])
                clouds['previous'] = imageio.imread(cloud_path) / 255
            else:
                continue

            is_clouds_current = np.sum(clouds['current']) / clouds['current'].size < 0.2
            is_clouds_previous = np.sum(clouds['previous']) / clouds['previous'].size < 0.2
            if is_clouds_current and is_clouds_previous:
                images['previous'] = match_histograms(images['current'],
                                                      images['previous'],
                                                      multichannel=True)

                diff_img = self.diff(images, self.width, self.height)

                diff_msk = (np.abs(masks['current'] - masks['previous']) > 0) * 255
                
                diff_msk = (gaussian_filter(diff_msk, 0.5) > 0) * 255
                diff_msk = diff_msk.astype(np.uint8)
                diff_msk = cv2.resize(diff_msk, (self.height, self.width),
                                      interpolation = cv2.INTER_NEAREST)
            
                meta['width'] = self.width
                meta['height'] = self.height
                meta['count'] = diff_img.shape[2]

                result_images = os.path.join(diff_path, self.images_path, diff_path.split('/')[-1]+'_'+xs[idx]+'_'+ys[idx]+'.tiff')
                with rs.open(result_images, 'w', **meta) as dst:
                    for ix in range(diff_img.shape[2]):
                        dst.write(diff_img[:, :, ix], ix + 1)
                dst.close()

                result_masks = os.path.join(diff_path, self.masks_path, diff_path.split('/')[-1]+'_'+xs[idx]+'_'+ys[idx]+'.png')
                imageio.imwrite(result_masks, diff_msk)
                writer.writerow([
                    diff_path.split('/')[-1], diff_path.split('/')[-1], xs[idx]+'_'+ys[idx], int(diff_msk.sum()/255)
                ])

    def get_diff_and_split(self):
        
        tiles_description = getdates(self.data_path)

        tiles = pd.DataFrame(tiles_description, columns=['tileID', 'img_date'])

        tiles = tiles.sort_values(['img_date'], ascending=False)

        infofile = os.path.join(self.save_path, 'data_info.csv')
        """
        markups = [gp.read_file(os.path.join(self.polys_path, shp)) for shp in os.listdir(self.polys_path)]
        for shp in markups:
            shp['img_date'] = shp['img_date'].apply(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
            )

        with open(infofile, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'dataset_folder', 'name', 'position', 'mask_pxl'
            ])

            number_of_dates = len(tiles)
            for index_current in range(number_of_dates - 1):
                index_next = index_current + 1
                for index_previous in range(index_next, index_next + self.days_limit):
                    if index_previous < number_of_dates:
                        date_current = str(tiles['img_date'].iloc[index_current].date())
                        date_previous = str(tiles['img_date'].iloc[index_previous].date())

                        diff_path = f"{self.save_path}/{date_current}_{date_previous}"
                        markup_number_current, markup_number_previous = 0, 0

                        for shp_num in range(len(markups)):
                            if date_limit(tiles['img_date'].iloc[index_current], markups[shp_num]): 
                                markup_number_current = shp_num
                            if date_limit(tiles['img_date'].iloc[index_previous], markups[shp_num]): 
                                markup_number_previous = shp_num
                        
                        dt = tiles['img_date'].iloc[index_current] - tiles['img_date'].iloc[index_previous]
                        if dt.days > (self.days_limit + 1) * settings.SENTINEL_DELTA_DAYS:
                            pass
                        elif markup_number_current != markup_number_previous:
                            pass
                        else:
                            path_exists_or_create(diff_path)
                            path_exists_or_create(os.path.join(diff_path, self.images_path))
                            path_exists_or_create(os.path.join(diff_path, self.masks_path))
                            
                            self.imgdiff(tiles['tileID'].iloc[index_current],
                                         tiles['tileID'].iloc[index_previous],
                                         diff_path, writer)
        """     
        df = pd.read_csv(infofile)
        xy = df['position'].unique()
        
        np.random.seed(seed=59)
        rand = np.random.random(size=len(xy))
        
        train = []
        test = []
        valid = []

        for i in range(len(xy)):
            if rand[i] <= self.train_size:
                train.append(xy[i])
            elif rand[i] > self.train_size and rand[i] < self.train_size + self.test_size:
                test.append(xy[i])
            else:
                valid.append(xy[i])
        
        path_exists_or_create(f'{self.save_path}/onlymasksplit')
        for data_type, name_type in zip([train, test, valid],
                                        ['train', 'test', 'valid']):
            markups = 0
            position_save = os.path.join(self.save_path, 'onlymasksplit', f'{name_type}_df.csv')
            output_file = os.path.join(self.save_path, f'{name_type}_df.csv')
            os.system(f'head -n1 {infofile} > {output_file}')
            os.system(f'head -n1 {infofile} > {position_save}')

            for position in data_type:
                df[df['position'] == position].to_csv(output_file,
                                                    mode='a',
                                                    header=False,
                                                    index=False,
                                                    sep=',')
                df[(df['position'] == position) & (df['mask_pxl'] > 0)].to_csv(position_save,
                                                                            mode='a',
                                                                            header=False,
                                                                            index=False,
                                                                            sep=',')
                markups += df[df['position'] == position].shape[0]
            print(f"{name_type} markups: {markups}")
        
        print('Train split: %d'%len(train))
        print('Test  split: %d'%len(test))
        print('Valid split: %d'%len(valid))
