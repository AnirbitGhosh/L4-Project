
#%%
import io
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pixel_analysis import find_mean_std_pixel_value
import glob
from skimage import io
import cv2

# Alpha and beta and IO values from Macenko 2009: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
def macenko_norm_HnE(img, Io=240, alpha=1, beta=0.15):
    # reference H&E OD matrix.
    HERef = np.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]])
    # reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])


    # height, width and num of channels
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log10((img.astype(np.float64)+1)/Io) 
    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)

    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above β

    # Calculate SVD on the OD tuples
    # Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))


    # project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3]) 
    
    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)


    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
        
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T


    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    # recreate the normalized image using reference mixing matrix 
    Mnorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Mnorm[Mnorm>255] = 254
    Mnorm = np.reshape(Mnorm.T, (h, w, 3)).astype(np.uint8)  

    # Separating H component
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    # Separating E component
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    return (Mnorm, H, E)

#%%
if __name__ == "__main__":
    dir_name = sys.argv[1] 
    
    tile_dir = os.path.join(dir_name, "tiles")
    for file in os.listdir(tile_dir):
        # img_arr = np.array(io.imread(tile_dir+ "/" + file))
        img_arr = cv2.imread(tile_dir+ "/" + file, 1)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        
        if img_arr.std() > 20:
            try:
                im, h = macenko_norm_HnE(img_arr)
                io.imsave(dir_name + "/h_norm_tiles/" + file, h) 
                io.imsave(dir_name + "/macenko_tiles/" + file, im)
                print(f"Saved normalized tile for: {file}")
                
            except np.linalg.LinAlgError as LinAlgError:
                print(f"Convergence error for {file}, saving un-normalized tile")
                io.imsave(dir_name + "/h_norm_tiles/" + file , img_arr) 
                io.imsave(dir_name + "/macenko_tiles/" + file, img_arr)
                print(f"Saved un-normalized tile for: {file}")
        else:
            print(f"Image contrast too low for {file}, saving un-normalized tile")
            io.imsave(dir_name + "/h_norm_tiles/" + file, img_arr)  
            io.imsave(dir_name + "/macenko_tiles/" + file, img_arr)
            print(f"Saved un-normalized tile for: {file}")
 