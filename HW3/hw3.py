from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


#Load the dataset from the provided .npy file, center it around the origin
#Return it as a numpy array of floats
def load_and_center_dataset(filename):
    dataset = np.load(filename) # Load the dataset from .npy

    return dataset - np.mean(dataset, axis=0) #Center it around the origin and return it 

# Calculate 'S' and return the covariance matrix of the dataset as a numpy matrix (dxd array)
def get_covariance(dataset):
    return 1/(len(dataset) - 1) * np.dot(np.transpose(x), x) # The equation and return

# perform eigendecomposition on the covariance matrix S and return a diagonal matrix
# (numpy array) with the largest m eigenvalues on the diagonal in descending order, and a matrix (numpy
# array) with the corresponding eigenvectors as columns.
def get_eig(S, m):
    w,v = eigh(S, subset_by_index = [len(S) - m, len(S) - 1]) # Get the eigenvalues and eigenvectors between m and the last index

    w = w[::-1] # Reverse the eigenvalues in array
    w = np.diag(w) #Make it a diagonal matrix

    # Reverse all the inner arrays in the eigenvectors
    for i in range(len(S)): 
        v[i] = v[i][::-1]
    
    return w, v #Return the eigenvalues and their eigenvectors

# similar to get_eig, but instead of returning the first m, return all eigen-
# values and corresponding eigenvectors in a similar format that explain more than a prop proportion
# of the variance (specifically, please make sure the eigenvalues are returned in descending order).
def get_eig_prop(S, prop):
    sum = np.trace(S) #Get the sum of all eigenvalues

    w,v = eigh(S, subset_by_value = [sum*prop, np.inf]) #Get the eigenvalues and eigenvectors that is greater than prop*sum

    w = w[::-1] # Reverse the eigenvalues in array
    w = np.diag(w) #Make it a diagonal matrix

    # Reverse all the inner arrays in the eigenvectors
    for i in range(len(S)): 
        v[i] = v[i][::-1]

    return w,v 

# project the image into the m-dimensional subspace and then project back
# into d×1 dimensions and return that.
def project_image(image, U):    
    alpha = np.dot(np.transpose(U), image) #Get alpha by doing dot product of tranpose of U and the image

    return np.dot(U, alpha) # Return a 1-d array of length image by doing dot product of U and alpha

# use matplotlib to display a visual representation of the original image
# and the projected image side-by-side.
def display_image(orig, proj):
    orig = np.reshape(orig, (32,32))
    proj = np.reshape(proj, (32,32))
    
    fig, (ax1,ax2) = plt.subplots(1, 2)

    ax1.set_title("Original")
    ax2.set_title("Projection")

    left = ax1.imshow(orig, aspect = 'equal')
    right = ax2.imshow(proj, aspect = 'equal')

    fig.colorbar(left, ax = ax1)
    fig.colorbar(right, ax = ax2)

    plt.show()
    

if __name__ == '__main__':
    # THIS WORKS
    x = load_and_center_dataset('YaleB_32x32.npy')

    # THIS WORKS
    S = get_covariance(x)

    # THIS WORKS
    m = 2
    Lambda, U = get_eig(S, m)

    # THIS WORKS
    prop = 0.07
    Lambda, U = get_eig_prop(S, prop)

    #THIS WORKS
    projection = project_image(x[0], U)
    
    #THIS DOESN'T WORK
    display_image(x[0], projection)


    
