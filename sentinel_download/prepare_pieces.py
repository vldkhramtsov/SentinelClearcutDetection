import os
import re
import csv
import cv2
import imageio
import datetime
import argparse
import rasterio

import numpy as np
import pandas as pd
import geopandas as gp

from tqdm import tqdm
from rasterio import features
from geopandas import GeoSeries
from skimage import img_as_ubyte
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.plot import reshape_as_image

from utils import path_exists_or_create, date_limit, image_non_zero
from settings import MODEL_TIFFS_DIR, PIECES_DIR, POLYS_PATH, MAXIMUM_CLOUD_PERCENTAGE_ALLOWED
from settings import PIECE_WIDTH, PIECE_HEIGHT

class PreparePieces:
    def __init__(self):
        self.tiff_path = MODEL_TIFFS_DIR
        self.polys_path = POLYS_PATH
        self.width = PIECE_WIDTH
        self.height = PIECE_HEIGHT
        self.image_width = None
        self.image_height = None

    def poly2mask(self, filename, image_path, data_path, filter_by_date=True):
        markups = [gp.read_file(os.path.join(self.polys_path, shp)) for shp in os.listdir(self.polys_path)]

        date = filename.split('_')[-1][:8]
        date = datetime.datetime.strptime(date, '%Y%m%d')
        
        for shp in markups:
            shp['img_date'] = shp['img_date'].apply(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
                                                    )
        markup = pd.concat([shp for shp in markups if date_limit(date, shp)])
        if filter_by_date: 
            print(f"Markup interval: \
                  {markup['img_date'].min()} - {markup['img_date'].max()}")
            print("Number of polygons:", markup.size)
        if filter_by_date:
            date += datetime.timedelta(days=1)
            polys = markup[markup['img_date'] <= date].loc[:, 'geometry']
        else:
            polys = markup.loc[:, 'geometry']

        with rasterio.open(image_path) as image:
            polys = polys.to_crs({'init': image.crs})
            mask = features.rasterize(shapes=polys,
                                      out_shape=(image.height, image.width),
                                      transform=image.transform,
                                      default_value=255)

        if filter_by_date:
            filename = '{}/{}.png'.format(
                data_path,
                re.split(r'[/.]', image_path)[-2]
            )
        else:
            filename = f'{data_path}/full_mask.png'

        imageio.imwrite(filename, mask)
        return filename, markup

    def divide_into_pieces(self, filename, image_path, data_path):
        os.makedirs(f'{data_path}/images', exist_ok=True)
        os.makedirs(f'{data_path}/geojson_polygons', exist_ok=True)

        full_mask = imageio.imread(f'{data_path}/full_mask.png')

        with rasterio.open(image_path) as src, open(f'{data_path}/image_pieces.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([
                'original_image', 'piece_image', 'piece_geojson',
                'start_x', 'start_y', 'width', 'height'
            ])

            for j in tqdm(range(0, src.height // self.height)):
                for i in range(0, src.width // self.width):
                    raster_window = src.read(
                        window=Window(i * self.width, j * self.height, 
                                      self.width, self.height)
                    )
                    image_array = reshape_as_image(raster_window)[:, :, :3]

                    is_mask = full_mask[j * self.height: j * self.height + self.height,
                                        i * self.width:  i * self.width  + self.width].sum() > 0
                    
                    if image_non_zero(image_array) and is_mask:
                        image_format = 'tiff'
                        piece_name = f'{filename}_{j}_{i}.{image_format}'

                        poly = Polygon([
                            src.xy(j * self.height, i * self.width),
                            src.xy(j * self.height, (i + 1) * self.width),
                            src.xy((j + 1) * self.height, (i + 1) * self.width),
                            src.xy((j + 1) * self.height, i * self.width),
                            src.xy(j * self.height, i * self.width)
                        ])
                        gs = GeoSeries([poly])
                        gs.crs = src.crs
                        piece_geojson_name = f'{filename}_{j}_{i}.geojson'
                        gs.to_file(
                            f'{data_path}/geojson_polygons/{piece_geojson_name}',
                            driver='GeoJSON'
                        )
                        image_array = reshape_as_image(raster_window)

                        meta = src.meta
                        meta['height'] = image_array.shape[0]
                        meta['width'] = image_array.shape[1]
                        with rasterio.open(f'{data_path}/images/{piece_name}', 'w', **meta) as dst:
                            for ix in range(image_array.shape[2]):
                                dst.write(image_array[:, :, ix], ix + 1)

                        writer.writerow([filename, piece_name, piece_geojson_name,
                                         i * self.width, j * self.height, 
                                         self.width, self.height])

    def split_mask(self, mask_path, save_mask_path, image_pieces_path):
        pieces_info = pd.read_csv(
            image_pieces_path, dtype={
                'start_x': np.int64, 'start_y': np.int64,
                'width': np.int64, 'height': np.int64
            }
        )
        mask = imageio.imread(mask_path)
        self.image_width, self.image_height = mask.shape[0], mask.shape[1]
        for i in range(pieces_info.shape[0]):
            piece = pieces_info.loc[i]
            piece_mask = mask[
                piece['start_y']: piece['start_y'] + piece['height'],
                piece['start_x']: piece['start_x'] + piece['width']
            ]
            filename_mask = '{}/{}.png'.format(
                save_mask_path,
                re.split(r'[/.]', piece['piece_image'])[-2]
            )
            imageio.imwrite(filename_mask, piece_mask)

    def split_cloud(self, cloud_path, save_cloud_path, image_pieces_path):
        pieces_info = pd.read_csv(
            image_pieces_path, dtype={
                'start_x': np.int64, 'start_y': np.int64,
                'width': np.int64, 'height': np.int64
            }
        )
        
        with rasterio.open(cloud_path) as cld:
            clouds = reshape_as_image(cld.read())
            clouds = cv2.resize(clouds, (self.image_width, self.image_height), 
                                interpolation = cv2.INTER_CUBIC)
            # to [-1; 1]
            clouds = np.clip(clouds, 0, 100)
            clouds = (clouds/100 * 2 - 1)
            # clouds = img_as_ubyte(clouds)
            for i in range(pieces_info.shape[0]):
                piece = pieces_info.loc[i]
                piece_cloud = clouds[
                    piece['start_y']: piece['start_y'] + piece['height'],
                    piece['start_x']: piece['start_x'] + piece['width']
                ]
                filename_cloud = '{}/{}.png'.format(
                    save_cloud_path,
                    re.split(r'[/.]', piece['piece_image'])[-2]
                )
                imageio.imwrite(filename_cloud, piece_cloud)

    def process(self):
        for filename in os.listdir(self.tiff_path):
            data_path = path_exists_or_create(os.path.join(PIECES_DIR, filename))
            image_path = os.path.join(self.tiff_path, filename, f"{filename}_merged.tiff")
            cloud_path = os.path.join(self.tiff_path, filename, "clouds.tiff")
            self.poly2mask(filename, image_path, data_path, filter_by_date=False)
            mask_path, markup = self.poly2mask(filename, image_path, 
                                            data_path, filter_by_date=True)
            self.divide_into_pieces(filename, image_path, data_path)
            pieces_info = os.path.join(data_path, 'image_pieces.csv')
            
            save_mask_path = path_exists_or_create(os.path.join(data_path, 'masks'))
            self.split_mask(mask_path, save_mask_path, pieces_info)
            save_cloud_path = path_exists_or_create(os.path.join(data_path, 'clouds'))
            self.split_cloud(cloud_path, save_cloud_path, pieces_info)
