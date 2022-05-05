import argparse
import configparser
import time
import numpy as np
from astropy.io import fits
import sep
import scipy.interpolate as interpolate
import construct_psfs as cpsf

#########################################
# Routine to parse command line arguments
#########################################

def create_parser():
    #Handle user input with argparse
    parser = argparse.ArgumentParser(description="matched detection flags and options from user")

    parser.add_argument('-i', '--input',
                default = 'flux_list.txt',
                metavar = 'flux images',
                type=str,
                help='Specify the list of input flux images for detection.')

    parser.add_argument('-e', '--error',
                default = 'error_list.txt',
                metavar = 'error images',
                type=str,
                help='Specify the list of uncertainty images for detection.')
#
#
#    parser.add_argument('-x', '--xmap',
#                default = None,
#                metavar = 'exposure maps',
#                type=str,
#                help='Specify the list of exposure map images for computing exposure time.')
#
#    parser.add_argument('-m', '--mask',
#                default = None,
#                metavar = 'mask image',
#                type=str,
#                help='Specify the mask previously used for source detection.')
#
#    parser.add_argument('-s', '--sources',
#                default = 'source.cat',
#                metavar = 'source catalog',
#                type=str,
#                help='Specify the input source catalog with objects on which to perform photometry.')
#
#    parser.add_argument('-t', '--template',
#                default = 'snr.cat.fits',
#                metavar = 'source catalog',
#                type=str,
#                help='Specify the input source FITS catalog with objects on which to perform photometry.')
#
#
    parser.add_argument('-o','--output',
                default = 'output',
                metavar = 'output directory',
                type=str,
                help='Directory to hold output files.')


#
#    parser.add_argument('-c',
#                                '--circ_aper',
#                        nargs='+',
#                                type=float,
#                        help='list of aperture diameters (in arcsec)')
#
#    parser.add_argument('-ps', '--pixel_shift',
#                dest='pixel_shift',
#                action='store_true',
#                help='Shift pixels to align with FITS origin (1,1) convention? (default: True)',
#                default=True)
#
    parser.add_argument('-g', '--with-gpu',
                dest='gpu',
                action='store_true',
                help='Use GPU to accelerate? (default: False)',
                default=False)

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)

    return parser

###########################################
#
# Create class defining SEP parameters
#
###########################################


class SEPParameters:
        def __init__(self, thresh=1.8, deblend_cont=0.0001, minarea=14, filter_type='matched', deblend_nthresh=32, clean=True, clean_param=1.0):
#       def __init__(self, thresh=1.4, deblend_cont=0.0001, minarea=14, filter_type='matched', deblend_nthresh=32, clean=True, clean_param=1.0):
#       def __init__(self, thresh=1.4, deblend_cont=0.0001, minarea=5, filter_type='matched', deblend_nthresh=32, clean=True, clean_param=1.0):

                self.thresh = thresh  # threshold in sigma
                self.deblend_cont = deblend_cont  # deblending contrast
                self.minarea = minarea  # minimum pixel area for a source
                self.filter_type = filter_type  # detection filter type
                self.deblend_nthresh = deblend_nthresh  # deblending pixel threshold
                self.clean = clean  # clean?
                self.clean_param = clean_param  # clean parameter
                self.int_nan = np.nan  # -999999                 #integer representation of nan


###########################################
#
# Simple routine to read fits data and
# header for use with SEP
#
###########################################
def read_fits(fname):
    hdul_data = fits.open(fname, memmap=True)
    data = hdul_data[0].data
    header_data = hdul_data[0].header
    #data = data.byteswap().newbyteorder()
    data = data.byteswap(inplace=True).newbyteorder()
    return data, header_data

###########################################
#
# Parse an input file with SEP parameters
#
###########################################


#def parse_sep_parameters(fp,args):
#        config = configparser.ConfigParser()
#        config.read(args.sep_param_file)
#        thresh = config['default']['thresh']
#        deblend_cont = config['default']['deblend_cont']
#        minarea = config['default']['minarea']
#        filter_type = config['default']['filter_type']
#        deblend_nthresh = config['default']['deblend_nthresh']
#        clean = config['default']['clean']
#        clean_param = config['default']['clean_param']
#
#        if(args.verbose):
#                print("SEP parameter file inputs:")
#                print("thresh = ", thresh)
#                print("deblend_cont = ", deblend_cont)
#                print("minarea = ", minarea)
#                print("filter_type = ", filter_type)
#                print("deblend_nthresh = ", deblend_nthresh)
#                print("clean = ", clean)
#                print("clean_param = ", clean_param)


#load a list of strings
def read_string_list(fname):
    fp = open(fname,"r") #open file
    fl_in = fp.readlines() #read all lines as strings
    fl = [] #create a list
    for l in fl_in: #loop over lines in file
        fl.append(l.strip('\n')) #append line to list w/o return
    fp.close() #close file
    return fl #return list of lines

#load images from file lists
def load_images(args):

    #list of flux image and error filenames
    fname_data = read_string_list(args.input)
    fname_err  = read_string_list(args.error)

    #how many images?
    nimg = len(fname_data)

    #loop over images
    for i in range(nimg):

        #print file names?
        if(args.verbose):
            print("For entry ",i)
            print("Science image file name: ",fname_data[i])
            print("Error   image file name: ",fname_err[i])

        #read in the current files
        sci_in, header_sci_in = read_fits(fname_data[i])
        err_in, header_err_in = read_fits(fname_err[i])

        #create flux images
        if(i==0):
            sci = np.zeros((nimg,sci_in.shape[0],sci_in.shape[1]))
            err = np.zeros((nimg,sci_in.shape[0],sci_in.shape[1]))
            header = header_sci_in.copy()

        #save input images
        sci[i,:,:] = sci_in.copy(order='C')
        err[i,:,:] = err_in.copy(order='C')

    #return images
    return nimg, sci, err



#create a mask from the input data
def create_mask(data, data_err):
    # time mask computation time
    t_start = time.time()

    # get indices of real and NaN values in data
    idx_real = np.where((np.isnan(data) == False) & (np.isnan(data_err) == False))
    idx_nan = np.where((np.isnan(data) == True) & (np.isnan(data_err) == True))

    # ### Make a mask of NaN pixels
    mask = np.empty_like(data, dtype=bool)
    mask[idx_nan] = True

    #return the mask
    return mask


#compute the norm of the correlation filter (e.g., the PSF)
def correlation_norm(psi):
  return np.sum(psi**2)**0.5


#actual correlation implementation
def correlate(x, psi, sg, normalize=True, gpu=False):

    if(gpu):
        import cupy as cp

    #here x is a 2d numpy array
    #and so is psi, which is the filter

    #normalize by norm(psi)**2?
    psi_norm = 1.
    if(normalize==True):
        psi_norm = correlation_norm(psi)**2

    #decide which correlation function to use
    if(gpu==False):
        d = sg.correlate(x,psi,mode='same')/psi_norm
    else:
        x_gpu = cp.asarray(x)
        psi_gpu = cp.asarray(psi)
        d_gpu = sg.correlate2d(x_gpu,psi_gpu,mode='same')
        d = cp.asnumpy(d_gpu)/psi_norm

    #return image correlated
    #with the kernel
    return d


#perform the correlations
def perform_correlations(nimg, sci, err, psf, sg, args):


    #allocate detection image
    det_sci = np.zeros_like(sci)

    #create a mask
    mask = np.zeros_like(sci,dtype=bool)
    for i in range(nimg):
        mask[i,:,:] = create_mask(sci[i,:,:],err[i,:,:])


    #create detection images
    # time mask computation time
    t_start = time.time()
    for i in range(nimg):
        if(args.verbose):
            print("Performing science image correlation for image ",i)

        x = sci[i,:,:].copy()
        m = mask[i,:,:].copy()
        idx_nan = np.where(m==True)
        x[idx_nan] = 0.0
        d = correlate(x, psf[i,:,:], sg, gpu=args.gpu)
        #d[idx_nan] = np.nan
        det_sci[i,:,:] = d
    # time mask computation time
    t_end = time.time()

    print("Time to compute sci correlations  = ",t_end-t_start)

    #allocate the detection error images
    derr = np.zeros_like(err)

    #create detection error images
    # time mask computation time
    t_start = time.time()
    for i in range(nimg):
        if(args.verbose):
            print("Performing error image correlation for image ",i)
        x = err[i,:,:].copy()
        m = mask[i,:,:].copy()
        idx_nan = np.where(m==True)
        x[idx_nan] = 0.0
        d = correlate(x**2, psf[i,:,:], sg, gpu=args.gpu)**0.5 #supply the stddev image
        #d[idx_nan] = np.nan
        derr[i,:,:] = d
    # time mask computation time
    t_end = time.time()

    print("Time to compute err correlations  = ",t_end-t_start)

    #return the detection and det error images
    return det_sci, derr


def print_arguments(args):
    print("args.input ",args.input)
    print("args.error ",args.error)
    print("args.gpu   ",args.gpu)

#define the main program
def main():

    #parse arguments

    # read in command line arguments
    parser=create_parser()
    args=parser.parse_args()

    #print arguments
    if(args.verbose):
        print_arguments(args)

    #choose a correlation library
    if(args.gpu):
        import cusignal.convolution as sg
    else:
        import scipy.signal as sg


    #load the images from the file lists
    #assumes all images are on the same
    #pixel grid
    nimg, sci, err = load_images(args)

    #construct the PSFs

    #psf = gather_psfs(nimg, sci, err, args)

#    psf = construct_psfs(nimg, sci, err, args, thresh=100.0)
#    x_size = 30
    x_size = 15
    psf = np.zeros((nimg,2*x_size+1,2*x_size+1))

    for i in range(nimg):

        #find psf centers
        x, y = cpsf.find_psf_centers(sci[i,:,:], err[i,:,:], thresh=50.0, x_size=x_size, flag_xwin=True, verbose=True)

        #make psf
        psf[i,:,:] = cpsf.calculate_psf(x, y, sci[i,:,:], err[i,:,:], x_size=x_size, verbose=True)

    for i in range(nimg):
        fname = "psf.%d.fits" % i
        fits.writeto(fname,psf[i,:,:],overwrite=True)

    #perform the mosaic-PSFs correlations
    #note this returns the sci and err (stdev) images
    det_sci, det_err = perform_correlations(nimg, sci, err, psf, sg, args)

    #create a mask
    mask = np.zeros_like(sci,dtype=bool)
    for i in range(nimg):
        mask[i,:,:] = create_mask(sci[i,:,:],err[i,:,:])


    for i in range(nimg):
        m = mask[i,:,:].copy()
        idx_nan = np.where(m==True)
        d = det_sci[i,:,:].copy()
        d[idx_nan] = np.nan
        fname = "det_sci.%d.fits" % i
        fits.writeto(fname,d.astype(np.float32),overwrite=True)


    for i in range(nimg):
        m = mask[i,:,:].copy()
        idx_nan = np.where(m==True)
        d = det_err[i,:,:].copy()
        d[idx_nan] = np.nan
        fname = "det_err.%d.fits" % i
        fits.writeto(fname,d.astype(np.float32),overwrite=True)

    #construct the SED-matched image
    snr = np.zeros((det_sci.shape[1],det_sci.shape[2]))
    F_star = np.zeros((det_sci.shape[1],det_sci.shape[2]))
    sigma_F_star = np.zeros((det_sci.shape[1],det_sci.shape[2]))

    #inverse variance weight images
    #into an SNR image
    n_nan = np.zeros(nimg)
    for i in range(nimg):
        m = mask[i,:,:].copy()
        F_i = det_sci[i,:,:].copy()
        sigma_i = det_err[i,:,:].copy()

        idx_nan = np.where(m==True)
        n_nan[i] = len(idx_nan[0])
        idx_real = np.where(m==False)
        F_star[idx_real] += F_i[idx_real]/sigma_i[idx_real]**2
        sigma_F_star[idx_real] += 1./sigma_i[idx_real]**2

    F_star /= sigma_F_star
    sigma_F_star = 1./np.sqrt(sigma_F_star)

    #compute SNR image
    snr = F_star/sigma_F_star
    fname = "snr.matched.fits"
    idx_nan_min = np.argmin(n_nan)
    m = mask[idx_nan_min,:,:].copy()
    idx_nan = np.where(m==True)
    snr[idx_nan] = np.nan
    fits.writeto(fname,snr.astype(np.float32),overwrite=True)

    #perform object detection


#run the program
if __name__=="__main__":
    main()