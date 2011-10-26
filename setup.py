#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('qcinv',parent_package,top_path)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='qcinv',
          packages=['qcinv'],
          configuration=configuration)
