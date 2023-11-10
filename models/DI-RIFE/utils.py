import os
import os.path as osp
from datetime import datetime


class Logger:
    """
    Logger class to record training log
    """

    def __init__(self, file_path, verbose=True):
        self.verbose = verbose
        self.create_dir(file_path)
        self.logger = open(file_path, 'a+')

    def create_dir(self, file_path):
        dir = osp.dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, prefix='', timestamp=False):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        if prefix == '':
            info = prefix + now
        else:
            info = prefix + ' ' + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if self.verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()
