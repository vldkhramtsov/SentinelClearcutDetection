import os
from os.path import join

import imageio
import rasterio
import numpy as np

from tqdm import tqdm

from utils import path_exists_or_create
from settings import DOWNLOADED_IMAGES_DIR, MODEL_TIFFS_DIR

ROOT = '.SAFE'

def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32 --quiet'
    )

def to_tiff(input_jp2_file, output_tiff_file, output_type='Float32'):
    os.system(
        f'gdal_translate -ot {output_type} \
        {input_jp2_file} {output_tiff_file}'
    )

def scale_img(img_file, output_file=None, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)

        output_file = os.path.splitext(img_file)[0] if output_file is None else output_file

        os.system(
            f'gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {output_file}'
        )

def prepare_tiff(filename):
    save_path = path_exists_or_create(join(MODEL_TIFFS_DIR, f"{filename}"))
    
    to_tiff(join(DOWNLOADED_IMAGES_DIR, f'{filename}{ROOT}_TCI.jp2'), 
                 join(save_path, f'{filename}_TCI.tif'), 'Byte')
    output_tiffs = {}
    for band in ['B04', 'B08', 'B8A', 'B11', 'B12']:
        output_tiffs[f'tiff_{band}_name'] = join(save_path, f'{filename}_{band}.tif')
        to_tiff(join(DOWNLOADED_IMAGES_DIR, f'{filename}{ROOT}_{band}.jp2'),
                output_tiffs[f'tiff_{band}_name'])

    output_tiffs['tiff_rgb_name'] = join(save_path, f'{filename}_TCI.tif')
    output_tiffs['tiff_ndvi_name'] = join(save_path, f'{filename}_ndvi.tif')
    output_tiffs['tiff_ndmi_name'] = join(save_path, f'{filename}_ndmi.tif')

    print('\nndvi band is processing...')
    get_ndvi(output_tiffs.get('tiff_B04_name'),
             output_tiffs.get('tiff_B08_name'),
             output_tiffs.get('tiff_ndvi_name'))

    print('\nndmi band is processing...')
    get_ndvi(output_tiffs.get('tiff_B11_name'),
             output_tiffs.get('tiff_B8A_name'),
             output_tiffs.get('tiff_ndmi_name'))

    for band in ['B04', 'B08', 'B8A', 'B11', 'B12', 'ndvi', 'ndmi']:
        output_tiffs[f'scaled_{band}_name'] = f"{output_tiffs[f'tiff_{band}_name']}_scaled.tif"
        scale_img(output_tiffs[f'tiff_{band}_name'], output_tiffs[f'scaled_{band}_name'])

    tiff_output_name = join(save_path, f'{filename}_merged.tiff')
    os.system(
        f"gdal_merge.py -separate -o {tiff_output_name} \
        {output_tiffs.get('tiff_rgb_name')} \
        {output_tiffs.get('scaled_B08_name')} {output_tiffs.get('scaled_B8A_name')} \
        {output_tiffs.get('scaled_B11_name')} {output_tiffs.get('scaled_B12_name')} \
        {output_tiffs.get('scaled_ndvi_name')} {output_tiffs.get('scaled_ndmi_name')}"
    )

    to_tiff(join(DOWNLOADED_IMAGES_DIR, f'{filename}{ROOT}_MSK_CLDPRB_20m.jp2'),
            f'{join(save_path, "clouds.tiff")}')

    for item in os.listdir(save_path):
        if item.endswith('.tif'):
            os.remove(join(save_path, item))
    


def preprocess():
    filenames = [file.split(ROOT)[0] for file in os.listdir(DOWNLOADED_IMAGES_DIR)]
    filenames = set(filenames)
    for filename in tqdm(filenames):
        print(filename)
        prepare_tiff(filename)
        print('==============')
