#!/usr/bin/env python

healpix_incdir = "/Users/dhanson/Desktop/lib/Healpix_2.15a/src/f90/mod"
healpix_libdir = "/Users/dhanson/Desktop/lib/Healpix_2.15a/lib"

cfitsio_libdir = "/Users/dhanson/Desktop/lib/cfitsio3250"

import glob

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('qcinv',parent_package,top_path)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='qcinv',
          packages=['qcinv'],
          configuration=configuration)
