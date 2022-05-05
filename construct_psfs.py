import time
import numpy as np
import sep
import scipy.interpolate as interpolate
from scipy.spatial import KDTree


########################################################
def match_sources(n_cat, cat_pos, ref_tree, r_search, n_max = 10):


    n_match = np.zeros(n_cat,dtype=np.int32)
    idx_match = np.zeros( (n_cat, n_max) ,dtype=np.int32)

    #loop through objects
    for i in range(n_cat):

        #find nearest neighbors to cat_post in ref_tree
        pts = ref_tree.query_ball_point(cat_pos[i,:],r_search)

        ##print(i, "Number of matches = ",len(pts), " pos = ",cat_pos[i,:],ref_tree.data[pts])

        if(len(pts)>n_max):
            print("Error -> number of matches within search radius exceeds n_max = ", n_max)
            exit()

        n_match[i] = len(pts)
        idx_match[i,0:n_match[i]] = pts

        #if(len(pts)>1):
        #    print(i,pts)
        #    exit()

        #d, idx = ref_tree.query(cat_pos[i,:],k=n_max)
        #print(i, "query results ", d*3600., idx, ref_tree.data[idx])

    return n_match, idx_match

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


#find psf centers using sep
def find_psf_centers(sci, err, thresh=25.0, x_size=15, flag_xwin=False, verbose=False):

    #how large are the images
    nx = sci.shape[0]
    ny = sci.shape[1]

    #set square psf
    y_size  = x_size

    #enable a large pixstack in sep
    sep.set_extract_pixstack(1000000)

    #create a mask
    mask = create_mask(sci, err)

    #first perform object detection with very high threshold
    objects = sep.extract(sci, thresh, err=err, mask=mask)


    #use a more accuate centering, or not
    if(flag_xwin):

        #from sep read the docs

        #compute a flux
        flux, fluxerr, flag = sep.sum_circle(sci, objects['x'], objects['y'], 3.0,
                                     mask=mask)

        #compute a 0.5 flux radius
        r, flag = sep.flux_radius(sci, objects['x'], objects['y'], 6.*objects['a'], 0.5,
                          normflux=flux, subpix=5)

        #compute xwin
        sig = 2. / 2.35 * r  # r from sep.flux_radius() above, with fluxfrac = 0.5
        xc, yc, flag = sep.winpos(sci, objects['x'], objects['y'], sig)

    else:
        xc = objects['x']
        yc = objects['y']

    x2 = objects['x2']
    y2 = objects['y2']

    #what did we find?
    if(verbose):
        print("Number of high-threshold objects found: ", len(objects))
        print("sqrt of average x2 [in pixels]: ",np.nanmean(objects['x2'])**0.5)

    flag_use = np.zeros_like(xc,dtype=bool)
    flag_use[:] = True

    #loop over objects
    for j in range(len(objects)):

        x = xc[j]
        y = yc[j]

        #ensure that psf objects lie away from the edges
        if( (x//1 < 2*x_size+1)|( (x+1)//1 > nx-2*x_size)|(y//1 < 2*x_size+1)|( (y+1)//1 > ny-2*x_size)):
            flag_use[j] = False

        #ensure that the size of the object is small
        if((x2[j]**0.5>0.5*x_size)|(y2[j]**0.5>0.5*y_size)):
            flag_use[j] = False

    #OK, we can build a tree of these objects
    #and then censor ones that are too close

    #lengths of the two catalogs
    n_ref = len(xc)
    n_cat = len(xc)

    #separate arrays to store positions
    ref_pos = np.zeros((n_ref,2))
    cat_pos = np.zeros((n_cat,2))

    #build kDtree from reference
    ref_pos[:,0] = xc[:]
    ref_pos[:,1] = yc[:]
    ref_tree = KDTree(ref_pos)

    #build kDtree from catalog
    cat_pos[:,0] = xc[:]
    cat_pos[:,1] = yc[:]
    cat_tree = KDTree(cat_pos)


    r_search = 50.

    n_match, idx_match = match_sources(n_cat, cat_pos, ref_tree, r_search, n_max = 10)

    idx = np.where(n_match>1)
    flag_use[idx] = False


    #which objects should we use?
    idx_use = np.where(flag_use==True)


    #objects to use
    print("number of objects to use = ",len(idx_use[0]))

    #return x and y positions of PSF centers
    return xc[idx_use], yc[idx_use]


def calculate_psf(xpsf, ypsf, sci, err, x_size=15, verbose=False):

    #sigma is threshold for star detection
    #size of output PSF is 2*x_size +1


    #make the psf 2D array
    psf = np.zeros((2*x_size+1,2*x_size+1))
    npsf = psf.copy()

    #make a psf grid for interpolation
    y_size  = x_size
    psf_int = np.zeros( (4*x_size+1,4*y_size+1) )
    x_int = np.arange(4*x_size+1)
    y_int = np.arange(4*y_size+1)

    #how large are the images
    nx = sci.shape[0]
    ny = sci.shape[1]

    #how many objects contribute to PSF?
    n_obj = 0

    #loop over objects
    for j in range(len(xpsf)):

        #get x,y center of this object
        x = xpsf[j]
        y = ypsf[j]

        #store object value on a grid
        ix = np.floor(y)
        iy = np.floor(x)
        psf_int[:,:] = sci[int(ix-2*x_size):int(ix+2*x_size+1),int(iy-2*y_size):int(iy+2*y_size+1)].copy()

        #create a 2D interpolation of image around object
        rbs = interpolate.RectBivariateSpline(x_int,y_int,psf_int)

        dx = x_size
        dy = y_size

        for k in range(2*x_size+1):
            for l in range(2*y_size+1):
                xi = dx + k + (y-ix)
                yi = dy + l + (x-iy)
                z = rbs.ev(xi,yi)
                if(np.isnan(z)==False):
                    psf[k,l] += z
                    npsf[k,l] += 1.
#                   psf[i,k,l] += psf_int[dx+k,dy+l]

    if(verbose):
        print("Maximimum of psf = %e" % (psf[:,:].max()))

    #normalize psf
    psf /= npsf
    psf[:,:] /= np.sum(psf[:,:])

    #return psf
    return psf
