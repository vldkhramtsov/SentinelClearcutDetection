import os
import cv2
import csv
import random
import imageio
import argparse
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


def diff(img1, img2, width, height):
    dim = (width, height)
    I1 = np.clip(cv2.resize(img1.astype(np.float32), dim, interpolation = cv2.INTER_CUBIC), 0, 255)
    I2 = np.clip(cv2.resize(img2.astype(np.float32), dim, interpolation = cv2.INTER_CUBIC), 0, 255)
    d = ( (I1 - I2) / (I1 + I2) )
    d = ((d+1)*127).astype(np.uint8)
    return np.concatenate((d, I1.astype(np.uint8), I2.astype(np.uint8)), axis=-1)


def imgdiff(tile1, tile2, diff_path, save_path, data_path, 
            img_path, msk_path, cloud_path, writer, width, height):
    
    def path_to_image(tile, path, x, y, ext='.png'):
        return os.path.join(data_path, tile, path, tile + '_' + x + '_' + y + ext)
    
    xs = [piece.split('_')[-2:][0] for piece in os.listdir(os.path.join(data_path, tile1, img_path))]
    ys = [piece.split('_')[-2:][1].split('.')[0] for piece in os.listdir(os.path.join(data_path, tile1, img_path))]
    for i in range(len(xs)):
        is_path_tile_1 = os.path.exists(path_to_image(tile1, img_path, xs[i], ys[i], ext='.tiff'))
        is_path_tile_2 = os.path.exists(path_to_image(tile2, img_path, xs[i], ys[i], ext='.tiff'))
        if is_path_tile_1 and is_path_tile_2:
            img1, meta = readtiff(path_to_image(tile1, img_path, xs[i], ys[i], ext='.tiff')) 
            img2, _ = readtiff(path_to_image(tile2, img_path, xs[i], ys[i], ext='.tiff'))
            
            msk1 = imageio.imread(path_to_image(tile1, msk_path, xs[i], ys[i]))
            msk2 = imageio.imread(path_to_image(tile2, msk_path, xs[i], ys[i]))

            cld1 = imageio.imread(path_to_image(tile1, cloud_path, xs[i], ys[i]))/255
            cld2 = imageio.imread(path_to_image(tile2, cloud_path, xs[i], ys[i]))/255
        else:
            continue

        if np.sum(cld1)/cld1.size < 0.2 and np.sum(cld2)/cld2.size < 0.2:
            img2 = match_histograms(img2, img1, multichannel=True)
            diff_img = diff(img1, img2, width, height)

            diff_msk = (np.abs(msk1 - msk2) > 0) * 255
            name = diff_path.split('/')[-1] + '_' + xs[i] + '_' + ys[i] + '.png'
            diff_msk = (gaussian_filter(diff_msk, 0.5) > 0) * 255
            diff_msk = diff_msk.astype(np.uint8)
            diff_msk = cv2.resize(diff_msk, (height, width),
                                  interpolation = cv2.INTER_NEAREST)
		
            meta['width'] = width
            meta['height'] = height
            meta['count'] = diff_img.shape[2]

            with rs.open(os.path.join(diff_path, img_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.tiff'), 'w', **meta) as dst:
                for ix in range(diff_img.shape[2]):
                    dst.write(diff_img[:, :, ix], ix + 1)
            dst.close()

            imageio.imwrite(os.path.join(diff_path, msk_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.png'), diff_msk)
            writer.writerow([
                diff_path.split('/')[-1], diff_path.split('/')[-1], xs[i]+'_'+ys[i], int(diff_msk.sum()/255)
            ])
        else: pass

def get_diff_and_split(data_path, save_path, polys_path, img_path, msk_path, cloud_path, width, height, neighbours, train_size, test_size, valid_size):
    tiles_description = getdates(data_path)

    df = pd.DataFrame(tiles_description,
                      columns=['tileID', 'img_date'])

    df = df.sort_values(['img_date'],ascending=False)
    
    infofile = os.path.join(save_path,'data_info.csv')

    markups = [gp.read_file(os.path.join(polys_path, shp)) for shp in os.listdir(polys_path)]
    for shp in markups:
        shp['img_date'] = shp['img_date'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        )

    with open(infofile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'dataset_folder', 'name', 'position', 'mask_pxl'
		])

        for i in range(len(df) - 1):
            for j in range(i + 1, i + 1 + neighbours):
                if j < len(df):
                    print(str(df['img_date'].iloc[i].date())+' - '+str(df['img_date'].iloc[j].date()))
                    print(f"dt={(df['img_date'].iloc[i]-df['img_date'].iloc[j]).days} days")
                    diff_path = os.path.join(save_path, str(df['img_date'].iloc[i].date())+'_'+str(df['img_date'].iloc[j].date()))                    
                    markup_number_i, markup_number_j = 0, 0
                    for shp_num in range(len(markups)):
                        if date_limit(df['img_date'].iloc[i], markups[shp_num]): 
                            markup_number_i = shp_num
                        if date_limit(df['img_date'].iloc[j], markups[shp_num]): 
                            markup_number_j = shp_num
                    
                    dt = df['img_date'].iloc[i] - df['img_date'].iloc[j]
                    if dt.days > (neighbours + 1) * 5:
	                    pass
                    elif markup_number_i != markup_number_j:
	                    pass
                    else:
                        path_exists_or_create(diff_path)
                        path_exists_or_create(os.path.join(diff_path,img_path))
                        path_exists_or_create(os.path.join(diff_path,msk_path))

                        imgdiff(df['tileID'].iloc[i], df['tileID'].iloc[j],
                                diff_path, save_path,
                                data_path, img_path, msk_path, cloud_path,
                                writer, width, height)

            
    df = pd.read_csv(infofile)
    xy = df['position'].unique()
    
    np.random.seed(seed=59)
    rand = np.random.random(size=len(xy))
    
    train = []
    test = []
    valid = []
    for i in range(len(xy)):
        if rand[i] <= train_size:
            train.append(xy[i])
        elif rand[i] > train_size and rand[i] < train_size + test_size:
            test.append(xy[i])
        else:
            valid.append(xy[i])
    
    path_exists_or_create(f'{save_path}/onlymasksplit')
    for data_type, name_type in zip([train, test, valid],
                                    ['train', 'test', 'valid']):
        markups = 0
        position_save = os.path.join(save_path, 'onlymasksplit', f'{name_type}_df.csv')
        output_file = os.path.join(save_path, f'{name_type}_df.csv')
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
    

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for dividing images into smaller pieces.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='data/output', help='Path to input data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='data/diff',
        help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        default=settings.POLYS_PATH,
        help='Path to the polygons'
    )
    parser.add_argument(
        '--img_path', '-ip', dest='img_path',
        default='images', help='Path to pieces of image'
    )
    parser.add_argument(
        '--msk_path', '-mp', dest='msk_path',
        default='masks', help='Path to pieces of mask'
    )
    parser.add_argument(
        '--cld_path', '-cp', dest='cld_path',
        default='clouds', help='Path to pieces of cloud map'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=settings.PIECE_WIDTH,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=settings.PIECE_HEIGHT,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--neighbours', '-nbr', dest='neighbours', default=6,
        type=int, help='Number of pairs before the present day (dt=5 days, e.g. neighbours=3 means max dt = 15 days)'
    )
    parser.add_argument(
        '--train_size', '-tr', dest='train_size',
        default=0.7, type=float, help='Represent proportion of the dataset to include in the train split'
    )
    parser.add_argument(
        '--test_size', '-ts', dest='test_size',
        default=0.15, type=float, help='Represent proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--valid_size', '-vl', dest='valid_size',
        default=0.15, type=float, help='Represent proportion of the dataset to include in the valid split'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    assert args.train_size + args.test_size + args.valid_size == 1.0
    
    path_exists_or_create(args.save_path)
    get_diff_and_split(args.data_path, args.save_path, args.polys_path, args.img_path, args.msk_path, args.cld_path,
                       args.width,args.height, args.neighbours,
                       args.train_size, args.test_size, args.valid_size)
