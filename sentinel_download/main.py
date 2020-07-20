import os

from sentinel_download import SentinelDownload
from prepare_pieces import PreparePieces
from prepare_tiff import preprocess
from image_difference import ImageDifference

if __name__ == '__main__':
    sentinel_downloader = SentinelDownload()
    sentinel_downloader.process_download()
    sentinel_downloader.executor.shutdown()

    preprocess()
    
    prepare_pieces = PreparePieces()
    prepare_pieces.process()

    difference_pieces = ImageDifference()
    difference_pieces.get_diff_and_split()
