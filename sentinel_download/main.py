import os

from sentinel_download import SentinelDownload
from prepare_pieces import PreparePieces
from prepare_tiff import preprocess

if __name__ == '__main__':
    try: 
        sentinel_downloader = SentinelDownload()
        sentinel_downloader.process_download()
        sentinel_downloader.executor.shutdown()
        
        preprocess()
        
        prepare_pieces = PreparePieces()
        prepare_pieces.preprocess()
        os.system('python image_difference.py')

    except Exception as error:
        pass
