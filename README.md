# Scripts for deforestation detection on the Sentinel-2 Level-A images 

## Description
This is a source code repository for downloading specified Sentinel-2 tiles and learning segmentation model. 

## Repository structure info
 * `data` - datafiles (polygons of deforestation regions in Ukraine)
 * `sentinel_download` - scripts for downloading S2A images
 * `model` - **TBD:** scripts for training U-Net segmentation model

## Setup: `sentinel_download`
All dependencies are given in requirements.txt. To download the tiles, you need to put the `key.json` file (with your GCP key) in the `sentinel_download` path.

The tiles which should be downloaded are listed in `../data/tiles/tiles_time-dependent.txt`, but the actual file, containing names of tiles could be changed, correcting the `self.labeled_tiles_to_download` in `class SentinelDownload` (`sentinel_download.py`). The files will be downloaded into the `DOWNLOADED_IMAGES_DIR = path_exists_or_create('data/source_images/')`. Update `gcp_config.ini` to specify which tiles and which bands (10m, 20m) should be downloaded; the cloud masks (20m) are retrieved dy default with any set of the bands.

Tiles are preprocessed (merging bands into `MODEL_TIFFS_DIR = path_exists_or_create('data/model_tiffs')`) and cropped into small pieces with `from prepare_tiff import preprocess` and `from prepare_pieces import PreparePieces` accordingly.

## Dataset
You can download our datasets directly from Google drive for the baseline and time-dependent models. The image tiles from Sentinel-2, which were used for our research, are listed in [this folder](https://nositeyet).

The data include *.geojson polygons:
* [baseline](https://nositeyet): 2318 polygons, **36UYA** and **36UXA**, **2016-2019** years;
* [time-dependent](https://nositeyet): **36UYA** (two sets of separated annotations, 278 and 123 polygons -- for spring and summer seasons respectively, **2019** year) and **36UXA** (1404 polygons, **2017-2018** years).
The files contain the following columns: `tileID` (ID of a tile, which was annotated), `img_date` (the date, at which the tile was observed), and `geometry` (polygons of deforestation regions). 

**TBD:** Also, we provide the set of images and masks prepared for training segmentation models as the [Kaggle dataset](https://nositeyet).

## Training
**TBD**

## Questions
If you have questions after reading README, please email to [Vladyslav Khramtsov](mailto:v.khramtsov@quantumobile.com).
