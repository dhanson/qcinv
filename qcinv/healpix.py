import numpy as np
import quickbeam as qb
import _healpix

def map2vlm(lmax, vmap, spin):
    assert( np.iscomplexobj(vmap) )
    return _healpix.map2vlm( lmax, vmap, spin )


def vlm2map(nside, vlm, spin):
    assert( np.iscomplexobj(vlm) )
    return _healpix.vlm2map( nside, vlm, spin )
