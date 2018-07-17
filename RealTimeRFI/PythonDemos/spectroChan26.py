#!/users/rprestag/venv/bin/python

# from astropy.io import fits
import numpy as np
import pylab as plt
from matplotlib import rcParams

# from GbtRaw import *

def spectroFITS(array, tStart, tRes, fStart, fRes, file_name):
    """Writes out array as an image in a FITS file"""

    # create the dynamic spectrum as the primary image
    hdu = fits.PrimaryHDU(array)

    # add the axes information
    hdu.header['CRPIX1'] = 0.0
    hdu.header['CRVAL1'] = tStart
    hdu.header['CDELT1'] = tRes
    hdu.header['CRPIX2'] = 0.0
    hdu.header['CRVAL2'] = fStart
    hdu.header['CDELT2'] = fRes

    # create the bandpass and timeseries
    bandpass    = np.average(array, axis=1)
    timeseries  = np.average(array, axis=0)

    # and create new image extensions with these
    bphdu = fits.ImageHDU(bandpass,name='BANDPASS')
    tshdu = fits.ImageHDU(timeseries,name='TIMESERIES')
    # uodate these headers.
    bphdu.header['CRPIX1'] = 0.0
    bphdu.header['CRVAL1'] = fStart
    bphdu.header['CDELT1'] = fRes
    tshdu.header['CRPIX1'] = 0.0
    tshdu.header['CRVAL1'] = tStart
    tshdu.header['CDELT1'] = tRes


    hdulist = fits.HDUList([hdu, bphdu, tshdu])
    hdulist.writeto(file_name)

def main():

    rcParams.update({'figure.autolayout' : True})
    rcParams.update({'axes.formatter.useoffset' : False})

    pol = 0

    # create 512 time bins of 1024 spectra, each with 32 integrations.
    # there may be a few bytes not used.
    nfreq = 1024
    nint  = 32

    # get the data
    tsData = np.load("chan26.npy")
    tsLen = tsData.shape[0]

    # required polarization channels                                       
    sp = 2*pol
    ep = sp+2



    # empty list of power spectra
    spec_list = []

    print tsLen
    print nfreq * nint
    nspec = int(tsLen / (nfreq * nint))
    print "processing ", nspec, "spectra..."


    # do the work
    for s in range(nspec):
        print "spectrum: ", s
        winStart = s * (nfreq * nint)
        accum = np.zeros(nfreq)
        for i in range(nint):
            start = winStart + i * nfreq
            end = start + nfreq
            in_arr = np.zeros((nfreq), dtype=np.complex_)
            in_arr.real = tsData[start:end, 0]
            in_arr.imag = tsData[start:end, 1]
            out_arr = np.fft.fftshift(np.fft.fft(in_arr))
            accum += np.abs(out_arr)**2
            spec_list.append(accum/nint)


    # convert back to numpy array and transpose to desired order
    dyn_spec = np.transpose(np.asarray(spec_list))

    # plot the results - first the spectrogram. Something is clearly wrong!
    plt.imshow(dyn_spec)
    plt.show()
    # This should be the time series - I am not sure why it looks like this...
    plt.plot(np.average(dyn_spec, axis=0),"r+")
    plt.show()
    # this is the average power-spectrum
    plt.plot(np.average(dyn_spec, axis=1))
    plt.show()

if __name__ == "__main__":
    main()


