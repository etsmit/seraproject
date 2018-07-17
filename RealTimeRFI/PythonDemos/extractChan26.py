#!/users/rprestag/venv/bin/python

import numpy as np
from  GbtRaw import *
def main():
    """Extarct channel 26 (25 0-relative) from the GUPPI RAW file"""

    infile = "/hyrule/data/users/rprestage/guppi_56465_J1713+0747_0006.0000.raw"
    g = GbtRaw(infile)
    data = g.extract(0,3)
    chan26 = (data[25,:,:])

    # save to a npy file
    np.save('chan26.npy', chan26)

if __name__ == "__main__":
    main()
    
