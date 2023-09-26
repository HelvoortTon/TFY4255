'''
Template Code for Lab 2 - Computer Lab (Code written by Tina Bergh)
TFY4255 Materials Physics 2020
Lecturer: Antonius T.J. van Helvoort
Lab assistant: Dipanwita Chatterjee

This is a template Python code (Python 3.6) written in a Jupyter notebook
for lab exercise 2 in materials physics.
The instructions are given in the lab text.

Last update: 2019-09-13 by Tina Bergh.

Jupyter notebook bascis:
- To run the cell where your cursor is, press **Shift and Enter**
- In [*] indicates that a cell is running, and a number appears after
    completion

To modify cells, click at the [ ]-icon outside of text edit mode, and
- a inserts a cell above
- b inserts a cell below
- m changes the cell to markdown, text mode
- y changes it back to code
- Shift and Ctrl and - splits the cell at the position of your cursor
- Shift and Tab inside the parathesis after a function name shows you the
    docstring, where you easily can find input variables and a description
    of the function

For tips on how to better use Jupyter notebook, there are many online
resources, see e.g.
https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

'''

# Import the required libraries.
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi


# Part 1: Diffraction from 1D crystals

# 1.1 Model the electron density of 1D atomic chains

# Define a function that returns the x-axis.
# Useful e.g. if you want to customize a plot of the electron density.


def get_x_axis(a_0=2., num_atoms=50, x_steplength=0.1):
    
    """Returns an x-axis with the given steplength within the bounds
    given by num_atoms.

    Parameters
    ----------
    x_steplength : float
        steplength between each value on the x-axis.
    a_0: float
        lattice constant; equilibrium position of atoms in [Å]. 
    num_atoms: float
        Number of atoms
        
    Returns
    -------
    x_axis : ndarray
        x_axis corresponding to the x_axis where the electron density is
        calculated by get_electron_density.
    """
    
    return np.arange(start=-(num_atoms)*a_0/2,
                     stop=(num_atoms)*a_0/2, 
                     step=x_steplength)


# Define a function that calculates the total electron density from a
# customisable 1D atomic chain.


def get_electron_density(x_steplength = .1,
                         model = 'Monoatomic',
                         sigma_1 = .1,
                         sigma_2 = .2,
                         a_0 = 2.,
                         thermal_disorder = False,
                         num_atoms = 50,
                         epsilon = 0.2,
                         epsilon_glass = 0.2,
                         amplitude_1 = 1.,
                         amplitude_2 = 2.,
                         plot_electron_density = True,
                         return_x_axis = False):
    
    """Calculates the total electron density of a 1D atomic chain.

    Parameters
    ----------
    x_steplength : float
        steplength between each value on the x-axis: the value where the electron
        density is calculated.   
    a_0: float
        lattice constant; equilibrium position of atoms. 
    num_atoms: float
        Number of atoms.
    model : string
        Model type: 'Monoatomic', 'Diatomic' or 'Glass', determines the expression
        for electron density.
    amplitude_1, amplitude_2: float
        Amplitude; factor multiplied with the Gaussian at each atom position.
        amplitude_2 is used in the diatomic model, otherwise only amplitude_1.
    sigma_1, sigma_2: float
        Width of the Gaussians in the expression for electron density.
        sigma_2 is used in diatomic model, otherwise only sigma_1.
    thermal_disorder: bool
        True if thermal disorder is to be included in the model, and False (default) 
        for no thermal disorder.
    epsilon: float
        Only used for thermal_disorder=True. 
        Limit of the random number (epsilon * random_number, where random_number
        is taken from the interval (-1,1)) that is added to the atom position. 
        Only used for thermal_disorder=True. 
    epsilon_glass: float
        Only used for model='glass'. 
        Limit of the random number (epsilon * random_number, where random_number
        is taken from the interval (-1,1)) that is added to the atom position. 
        Only used for thermal_disorder=True. 
    plot_electron_density: bool
        Determines if electron density should be plotted.
    return_x_axis : bool
        If True (False is default), the x_axis is returned in addition to
        the electron_density.
        
    Returns
    -------
    electron_density : ndarray
        Electron density at positions defined by x_axis.
    """
    
    # Spatial discretisation. Create an x-axis by using the function get_x_axis.
    x_axis = get_x_axis(a_0=a_0,
                        num_atoms=num_atoms,
                        x_steplength=x_steplength) 
    #Initiate variables
    electron_density = np.zeros(np.shape(x_axis))
    counter = 0
    # Initial atom position
    atom_position = - (num_atoms - 1 ) * a_0 / 2 - a_0
    
    for i in range(num_atoms):
        if thermal_disorder: 
            atom_position = -(num_atoms - 1) * a_0 / 2 + a_0*(i) + \
                            epsilon*(2 * np.random.random() - 1)
        else:
            atom_position = atom_position + a_0
        if model == 'Monoatomic':
            this_atoms_electron_density = amplitude_1 * np.exp(
                -(x_axis-atom_position)**2 / (2 * sigma_1**2))
        elif model == 'Diatomic':
            if np.fmod(counter,2):
                sigma_i = sigma_1
                amplitude_i = amplitude_1
            else:
                sigma_i = sigma_2
                amplitude_i = amplitude_2
            this_atoms_electron_density = amplitude_i * np.exp(
                -(x_axis-atom_position)**2 / ( 2 * sigma_i**2) )
            counter = counter+1
        elif model == 'Glass':
            atom_position = atom_position + epsilon_glass * \
                            (2 * np.random.random() - 1)
            this_atoms_electron_density = amplitude_1 * np.exp(
                -(x_axis-atom_position)**2 / ( 2 * sigma_1**2) )
        else: 
            print('Error: Must specify valid model type in string format.')
            
        electron_density = electron_density + this_atoms_electron_density
    
    if plot_electron_density:
        plt.figure()
        plt.plot(x_axis,
                 electron_density, 'k')
        plt.ylabel('Electron density [arb.]')
        plt.xlabel('Distance [$\AA$]')
        if thermal_disorder:
            plt.title(model+' 1D model with thermal disorder and with '+str(num_atoms)+' atoms.')
        else:
            plt.title(model+' 1D model with '+str(num_atoms)+' atoms.')
    if return_x_axis:
        return x_axis, electron_density
    else:
        return electron_density


# ### Explore different models and parameters

# - Read the documentation in the function above to see what the different
# parameters are. Select parameters that will be given to the function get_electron_density.
# - Explore different parameters and different models. Note the differences
# between the models and how the different parameters affect the results.

# Steplength between each value on the x-axis in [Å].
x_steplength = 0.001

# Model type: 'Monoatomic', 'Diatomic' or 'Glass', determines the expression
# for electron density.
model='Monoatomic'

# Number of atoms.
num_atoms = 5

# Distance between atoms in [Å].
a_0 = 2.

# The distribution width of the electron density of each atom,
# i.e. the width of the Gaussians in [Å].
sigma_1 = .1

# Magnitude of the electron density of each atom, 
# i.e. factor multiplied with the Gaussians at each atom position.
amplitude_1 = 1

# Only used for thermal_disorder=True. 
# Limit of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
epsilon = 0.2

# Width of the Gaussians in the expression for electron density.
# sigma_2 is used in diatomic model.
sigma_2 = 0.15

# Amplitude; factor multiplied with the Gaussian at each atom position.
# amplitude_2 is used in the diatomic model.
amplitude_2 = 1.3

# Only used for model='glass'. 
# Limit of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
epsilon_glass = 0.5


# Calculate and plot the electron density.

electron_density_mono_atomic = get_electron_density(x_steplength=x_steplength, 
                                                    model=model, 
                                                    sigma_1=sigma_1, 
                                                    a_0=a_0, 
                                                    num_atoms=num_atoms, 
                                                    amplitude_1=amplitude_1,
                                                    thermal_disorder=False)


# Plot several 1D models

x_axis = get_x_axis(a_0=a_0,
                    num_atoms=num_atoms,
                    x_steplength=x_steplength) 

electron_density_mono_atomic = get_electron_density(x_steplength=x_steplength,
                                                    sigma_1=sigma_1,
                                                    a_0=a_0,
                                                    num_atoms=num_atoms,
                                                    amplitude_1=amplitude_1, 
                                                    thermal_disorder=False,
                                                    plot_electron_density=False)

electron_density_diatomic = get_electron_density(x_steplength=x_steplength,
                                                 model='Diatomic',
                                                 sigma_1=sigma_1,
                                                 sigma_2=sigma_2,
                                                 a_0=a_0,
                                                 num_atoms=num_atoms,
                                                 amplitude_1=amplitude_1,
                                                 amplitude_2=amplitude_2,
                                                 thermal_disorder=False,
                                                 plot_electron_density=False)

electron_density_mono_atomic_thermal = get_electron_density(x_steplength=x_steplength,
                                                            sigma_1=sigma_1,
                                                            a_0=a_0,
                                                            num_atoms=num_atoms,
                                                            amplitude_1=amplitude_1,
                                                            thermal_disorder=True,
                                                            epsilon = epsilon,
                                                            plot_electron_density=False)

electron_density_glass = get_electron_density(x_steplength=x_steplength,
                                              model='Glass',
                                              sigma_1=sigma_1,
                                              a_0=a_0,
                                              num_atoms=num_atoms,
                                              epsilon_glass=epsilon_glass,
                                              amplitude_1=amplitude_1,
                                              plot_electron_density=False)



plt.figure(figsize=(12,5))
plt.plot(x_axis,
         electron_density_mono_atomic, 'orange', label=('mono-atomic'))
plt.plot(x_axis,
         electron_density_mono_atomic_thermal, 'b--', label=('mono-atomic thermal'))
plt.plot(x_axis,
         electron_density_glass, 'r', label=('glass'), linestyle='dashdot')
plt.plot(x_axis,
         electron_density_diatomic, 'g', label=('diatomic'), linestyle='dotted')
plt.ylabel('Electron density [arb.]')
plt.xlabel('Distance [Å]')
axes = plt.gca()
#axes.set_xlim([-(num_atoms+2)*a_0/2,
#               (num_atoms+2)*a_0/2])
plt.legend(fontsize = 'x-small')
plt.title('Electron densities of 1D models')


# ## 1.2 Fast Fourier Transforms of the Electron Density

from scipy.signal.windows import tukey


# The Tukey window can be useful when computing the FFTs. 
# Take a look at the Tukey (tapered cosine) window, calcualte and plot it.

alpha=0.5

x_axis = get_x_axis() 

tukey_window = tukey(len(x_axis),alpha=alpha)
FFT_tukey_window = np.fft.fftshift(np.fft.fft(tukey_window))
Q_axis = (np.arange(len(FFT_tukey_window)) - len(FFT_tukey_window)/2) / (len(FFT_tukey_window)/2 * x_steplength * pi)

electron_density_mono_atomic_tw = get_electron_density(plot_electron_density=False) * tukey_window

fig, axes = plt.subplots(ncols=3, figsize=(18, 6), sharex=False, sharey=False)
ax = axes.ravel()
ax[0].plot(x_axis,tukey_window, 'k')
ax[0].set_title('tukey window')
ax[1].plot(Q_axis,FFT_tukey_window, 'k')
ax[1].set_title('FFT of tukey window (note xlim)')
#ax[1].set_xlim([-1,1])
ax[1].set_xlim([x_axis[0], x_axis[-1]])
ax[2].plot(x_axis,electron_density_mono_atomic_tw, 'k')
ax[2].set_title('tukey window * electron_density_mono_atomic')


# Define a function that calculates the FFT and the intensity of an electron density. 
# 
# NB: Here intensity is simply given by the FFT multiplied with its
# complex conjugate, even though the theory says that this is only a proportionality.
# 

def get_intenisty_of_electron_density(x_steplength=.1,
                                      model='Monoatomic',
                                      sigma_1=.1,
                                      sigma_2=.2,
                                      a_0=2.,
                                      thermal_disorder=False,
                                      num_atoms=50,
                                      epsilon=0.2,
                                      epsilon_glass=0.2,
                                      amplitude_1=1.,
                                      amplitude_2=2.,
                                      tukey_window=True, 
                                      plot_on=True):
    
    """Calculates the intenisty of the electron density.

    Parameters
    ----------
    x_steplength : float
        steplength between each value on the x-axis: the value where the electron
        density is calculated.   
    a_0: float
        lattice constant; equilibrium position of atoms. 
    num_atoms: float
        Number of atoms.
    model : string
        Model type: 'Monoatomic', 'Diatomic' or 'Glass', determines the expression
        for electron density.
    amplitude_1, amplitude_2: float
        Amplitude; factor multiplied with the Gaussian at each atom position.
        amplitude_2 is used in the diatomic model, otherwise only amplitude_1.
    sigma_1, sigma_2: float
        Width of the Gaussians in the expression for electron density.
        sigma_2 is used in diatomic model, otherwise only sigma_1.
    thermal_disorder: bool
        True if thermal disorder is to be included in the model, and False (default) 
        for no thermal disorder.
    epsilon: float
        Only used for thermal_disorder=True. 
        Limit of the random number (epsilon * random_number, where random_number
        is taken from the interval (-1,1)) that is added to the atom position. 
    epsilon_glass: float
        Only used for model='glass'. 
        Limit of the random number (epsilon * random_number, where random_number
        is taken from the interval (-1,1)) that is added to the atom position. 
    tukey_window : bool
        If True, Tukey window is applied, i.e. multiplied with the electron density.
    plot_on : bool
        If True, the electron density and its intenisty is plotted.
        
    Returns
    -------
    x_axis : ndarray
        x_axis corresponding to the x_axis where the electron density is
        calculated by get_electron_density.
    """
    
    x_axis, electron_density = get_electron_density(x_steplength=x_steplength,
                                                    model=model,
                                                    sigma_1=sigma_1,
                                                    sigma_2=sigma_2,
                                                    a_0=a_0,
                                                    thermal_disorder=thermal_disorder,
                                                    num_atoms=num_atoms,
                                                    epsilon=epsilon,
                                                    epsilon_glass=epsilon_glass,
                                                    amplitude_1=amplitude_1,
                                                    amplitude_2=amplitude_2,
                                                    plot_electron_density=False,
                                                    return_x_axis=True)
    
    if tukey_window:
        tukey_window = tukey(len(x_axis))
        electron_density = electron_density * tukey_window

    FFT_of_electron_density = np.fft.fftshift(np.fft.fft(electron_density))
    
    Q_axis = (np.arange(len(FFT_of_electron_density)) -
              len(FFT_of_electron_density)/2) /\
              (len(FFT_of_electron_density)/2 * x_steplength * pi)
    
    intensity_of_electron_density = FFT_of_electron_density * FFT_of_electron_density.conjugate()

    if plot_on: 
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6), #dpi = 200,
                                 sharex=False, sharey=False)
        ax = axes.ravel()
        ax[0].plot(x_axis,electron_density, 'k')
        ax[0].set_title('Electron density')
        ax[0].set_xlabel('Distance [Å]')
        ax[0].set_ylabel('Intensity [a.u.]')
        ax[1].plot(Q_axis,intensity_of_electron_density, 'k')
        ax[1].set_title('Intensity of electron density')
        ax[1].set_xlabel('Frequency [1/Å]')
        ax[1].set_ylabel('Intensity [a.u.]')
        plt.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
        
    return intensity_of_electron_density


# Study the intensity. Again, investigate different models and paramters
# and understand how that influence the results.
# 
# Note the limits on the q_x-axis in reciprocal space. You might need to zoom in.

# Steplength between each value on the x-axis in [Å].
x_steplength = 0.001

# Model type: 'Monoatomic', 'Diatomic' or 'Glass', determines the expression
# for electron density.
model='Diatomic'

# Number of atoms. 
num_atoms = 50

# Distance between atoms in [Å].
a_0 = 2. 

# The distribution width of the electron density of each atom, 
# i.e. the width of the Gaussians in [Å].
sigma_1 = .1 

# Magnitude of the electron density of each atom, 
# i.e. factor multiplied with the Gaussians at each atom position.
amplitude_1 = 1

# True if thermal disorder is to be included in the model, 
# and False for no thermal disorder.
thermal_disorder=False

# Only used for thermal_disorder=True. 
# Limit of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
epsilon = 0.2

# Width of the Gaussians in the expression for electron density.
# sigma_2 is used in diatomic model.
sigma_2 = 0.15

# Amplitude; factor multiplied with the Gaussian at each atom position.
# amplitude_2 is used in the diatomic model.
amplitude_2 = 1.3

# Only used for model='glass'. 
# Limit of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
epsilon_glass = 0.5


intensity_of_electron_density_mono_atomic = get_intenisty_of_electron_density(
    x_steplength=x_steplength,
    model=model,
    sigma_1=sigma_1,
    sigma_2=sigma_2,
    a_0=a_0,
    thermal_disorder=thermal_disorder,
    num_atoms=num_atoms,
    epsilon=epsilon,
    epsilon_glass=epsilon_glass,
    amplitude_1=amplitude_1,
    amplitude_2=amplitude_2,
    plot_on=True,
    tukey_window=False)


# # Part 2: Fast Fourier Transformations and Imaging

import matplotlib.image as mpimg


# Import an image. If it is not located in the same folder as your Jupyter
# notebook, you need to include the whole file directory.

image = mpimg.imread('.\\grid_1000.tif')


# You probably know that the image is simply a large matrix, where each
# element corresponds to the value of a certain pixel. Plot the original
# image and take a look.

fig = plt.figure()
plt.imshow(image)
ax = fig.gca()
ax.set_axis_off()


# Noise
# Optionally add noise to the image, in order to study how that affects
# the simulated diffraction pattern, and in order to study noise-reduction
# and blurring.


def add_noise(image_temp, mu=0, sigma=0.1):
    """Adds Gaussian noise to an image. 

    Parameters
    ----------
    image_temp: ndarray
        Color or grayscale image. 
    mu : float
        Mean (centre) of the Gaussian distribution.
    scale : float
        Standard deviation (spread or width) of the distribution.

    Returns
    -------
    noise : ndarray
        The noise that is added to the image. 
    noisy_image : ndarray
        Image with noise added. 
    """
    C = np.max(image_temp)
    noise = (np.random.normal(loc=mu, scale=sigma, size=image_temp.shape)*C).astype('int')
    noisy_image = image_temp + noise
    noisy_image[noisy_image<0] = 0
    noisy_image[noisy_image>C] = C
    
    return noise, noisy_image


noise, image = add_noise(image)


plt.figure()
plt.imshow(noise)

fig = plt.figure()
plt.imshow(image)
ax = fig.gca()
ax.set_axis_off()


# Define a function that calculates a grayscale image from a color image.
# This is useful, since it is easier to compute the FFT from a grayscale
# image (or an image with only one color channel).


def get_grayscale_image(image_temp):
    """Converts a grayscale image to a color image.

    Parameters
    ----------
    image_temp: ndarray
        Color image, i.e. image with several channels (multi-dimensional). 
        
    Returns
    -------
    gray_image : ndarray
        Grayscale image, i.e. image with only one channel (two-dimensional).
    """
    try: 
        if np.shape(image_temp[0,0])[0] > 2: 
            gray_image = image_temp[...,0]*0.299 + image_temp[...,1]*0.587 + image_temp[...,2]*0.114
            if np.shape(image_temp[0,0])[0] > 3: 
                gray_image[np.where(image_temp[...,3] == 0.)] = 1.
            return gray_image
    except IndexError as e: 
        return image_temp


# Define a function that applies a Tukey window in 2D to an image,
# similar to the 1D case in part 1.


def apply_window_to_image(image_temp, alpha = 1.0):
    """Multiplies a grayscale image with a Tukey window function.
    
    Parameters
    ----------
    image_temp: ndarray
        Grayscale image, i.e. image with one channel (two-dimensional). 
    alpha : float, optional
        (From the scipy documentation) Shape parameter of the Tukey window, 
        representing the fraction of the window inside the cosine tapered region. 
        If zero, the Tukey window is equivalent to a rectangular window. 
        If one, the Tukey window is equivalent to a Hann window.
    
    Returns
    -------
    image_temp * window_2D : ndarray
        Grayscale image multiplied with the window function.
    
    """
    Y,X = np.indices((image_temp.shape[0], image_temp.shape[1]))
    Radius = (np.sqrt((X-int(image_temp.shape[0]/2))**2+
                      (Y-int(image_temp.shape[1]/2))**2)+0.5).astype(int)
    
    window_2D = np.array(tukey(
        int(np.sqrt(2*image_temp.shape[0]**2)), 
        alpha = alpha)[Radius.ravel()].reshape((
        image_temp.shape[0], image_temp.shape[1]))*(-1)+1)

    return image_temp * window_2D


# Define a function that calculates the intensity of a 2D image.

def calculate_intensity_of_image(image_temp):
    """Calculates the FFT of a grayscale image, and the intensity
    (as FFT times conjugate(FFT)).

    Parameters
    ----------
    image_temp: ndarray
        Grayscale or one-channel image. 
        
    Returns
    -------
    FFT_image : ndarray
        The FFT of the image.
    intensity_image : ndarray
        Absolute value of the FFT times its complex conjugate. 
    """
    
    FFT_image = np.fft.fftshift(np.fft.fft2(image_temp))
    intensity_image = np.abs(FFT_image * FFT_image.conjugate())

    return FFT_image, intensity_image


# Define a function that calculates the intensity of a 2D image and
# optionally plots it on a log-scale, after ensuring that the image is square and gray-scale, and optionally after applying a Tukey window function.


def get_intensity_of_image(image_temp, apply_window=True, alpha=1.0, zoom_factor=0.2, 
                           plot_intensity=True, cmap='inferno'):
    """Calculates the the intensity of a grayscale image (optionally
    multiplied by a window function), as FFT times conjugate(FFT) and
    plots the image, the intensity and the central part of the intensity.

    Parameters
    ----------
    image_temp: ndarray
        Grayscale or one-channel image. 
    apply_window : bool, optional 
        If True (default), the iamge is multiplied by a window function.
    alpha : float
        (From the scipy documentation) Shape parameter of the Tukey window, 
        representing the fraction of the window inside the cosine tapered region. 
        If zero, the Tukey window is equivalent to a rectangular window. 
        If one, the Tukey window is equivalent to a Hann window.
    zoom_factor : float, optional
        Factor representing the fraction of the image that should be dispalyed.
        Takes values (1,0).
    plot_intensity : bool, optional
        If True (default), the image and the intensity of the image will be plotted. 
    cmap : matplotlib.cm
        Matplotlib colormap. See e.g. for the options: 
        https://matplotlib.org/3.1.0/api/cm_api.html#matplotlib.cm.get_cmap
        Tips: by adding '_r', you invert the colormap!  
        
    Returns
    -------
    image_temp : ndarray
        The input image as a grayscale, square image, optionally with a Tukey window applied.
    FFT_image : ndarray
        The FFT of the image.
    intensity : ndarray
        Absolute value of the FFT of the image times its complex conjugate. 
    """
    # Convert to a grayscale image, since then it is easier to compute the FFT. 
    image_temp = get_grayscale_image(image_temp)
    
    # If the image is not square, crop the image, since otherwise, 
    # this would give artifacts in the FFT. 
    if image_temp.shape[0] != image_temp.shape[1]:
        min_length = np.min([image_temp.shape[0], image_temp.shape[1]])
        image_temp = image_temp[:min_length,:min_length]

    # Optionally apply a Tukey window to the image. 
    if apply_window:
        image_temp = apply_window_to_image(image_temp, alpha = alpha)

    FFT_image, intensity = calculate_intensity_of_image(image_temp)
    
    if plot_intensity:
            
        # Also, crop the image to view only the central part
        half_shape = int(intensity.shape[0]/2)
        limit = [int((intensity.shape[0]/2)*(1-zoom_factor)),
                int((intensity.shape[0]/2)*(1+zoom_factor))]
        intensity_cropped = intensity[limit[0]:limit[1], limit[0]:limit[1]]
        
        fig, axes = plt.subplots(ncols=3, figsize=(12,4))
        ax = axes.ravel()
        ax[0].imshow(image_temp, cmap=cmap)
        ax[0].set_title('Image')
        ax[0].set_axis_off()
        ax[1].imshow(np.log(intensity), cmap=cmap)
        ax[1].set_title('Intensity')
        ax[1].set_axis_off()
        ax[2].imshow(np.log(intensity_cropped), cmap=cmap)
        ax[2].set_title('Intensity - Central Part')
        ax[2].set_axis_off()
    
    return image_temp, FFT_image, intensity


img, FFT_img, img_int = get_intensity_of_image(image)


# Define a function that enables masking of the intensity image.

def mask_FFT_image(FFT_image,
                   intensity_image,
                   mask_type='circular',
                   mask_length=None,
                   x_offset=0,
                   y_offset=0,
                   invert_mask=True,
                   make_plot=True,
                   cmap='inferno'): 
    
    """Masks an FFT and intensity, and calculates its inverse transform. 

    Parameters
    ----------
    FFT_image: ndarray
        FFT of a one-channel image. 
    intensity_image : ndarray
        Absolute value of the FFT times its complex conjugate. 
    mask_type: string
        'circular' or 'rectangular': determines the shape of the mask. 
    mask_length: float
        Length of the mask. If type is 'circular', this is the radius, and for 'rectangular,
        it is half of the rectangle's length. 
    x_offset, y_offset : float
        mask center offset in horisontal or vertical direction 
    invert_mask : bool
        If True, the mask will keep only the region of the FFT that inside of the mask. 
        If False, only that outside of the mask. 
    make_plot : bool
        If True, the FFT, the masked FFT and the inverse tranformed masked FFT will be plotted. 
    cmap : matplotlib.cm
        Matplotlib colormap. See e.g. for the options: 
        https://matplotlib.org/3.1.0/api/cm_api.html#matplotlib.cm.get_cmap
        Tips: by adding '_r', you invert the colormap!
        
    Returns
    -------
    FFT_image : ndarray
        FFT of image.
    intensity_image : ndarray
        Absolute value of the FFT times its complex conjugate. 
    """
    
    FFT_mask = np.ones(np.shape(FFT_image),dtype=bool)
    cx = np.round(np.shape(FFT_image)[1]/2)
    cy = np.round(np.shape(FFT_image)[0]/2)

    if mask_length is None: 
        mask_length=1/10*np.min(np.shape(FFT_image)[:1])
    
    if mask_type == 'rectangular':
        FFT_mask[int(cy+y_offset-mask_length):int(cy+y_offset+mask_length), 
                 int(cx+x_offset-mask_length):int(cx+x_offset+mask_length)] = False
        
    elif mask_type == 'circular':
        Y,X=np.indices((np.shape(FFT_image)[0],np.shape(FFT_image)[1]))+0.5
        Radius = np.sqrt((X-(cx+x_offset))**2+(Y-(cy+y_offset))**2)+0.5
        Radius = Radius.astype(int)
        Radius -= int(Radius.min())
        FFT_mask[np.where(Radius<mask_length)]=False
    
    if invert_mask: 
        masked_FFT = np.logical_not(FFT_mask) * FFT_image
    else: 
        masked_FFT = FFT_mask * FFT_image
        
    abs_masked_image = np.abs(np.fft.ifft2(masked_FFT))
    
    if make_plot:
        fig, axes = plt.subplots(ncols=3, figsize=(18, 6), 
                                 sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        ax[0].imshow(np.log(intensity_image.astype(np.float32)), cmap=cmap, interpolation='none')
        ax[0].set_title('FFT of image')
        ax[1].imshow(np.log(np.abs(masked_FFT)), cmap=cmap, interpolation='none')
        ax[1].set_title('Masked FFT of image')
        ax[2].imshow(abs_masked_image, cmap=cmap, interpolation='none')
        ax[2].set_title('IFFT of masked FFT')
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()

    return np.abs(masked_FFT), abs_masked_image


# Define a function that incorporates all the functions defined earlier:
# calculates the intensity of the image, masks it in the frequency domain,
# before the image is back-transformed.


def back_transform_image(image_temp,
                         apply_window=True,
                         alpha = 1.,
                         mask_FFT_on = True,
                         mask_type = 'circular',
                         mask_length = None,
                         x_offset = 0,
                         y_offset = 0,
                         invert_mask = True,
                         make_FFT_mask_plot = False,
                         make_back_transform_image_plot = True,
                         cmap='inferno'):
    """Masks an FFT and intensity, and calculates its inverse transform. 

    Parameters
    ----------
    image: ndarray
        Color or grayscale image. If color, it will be tranformed into grayscale. 
        If not square, it will be cropped. 
    apply_window : bool
        If True, a Tukey window will be applied to the grayscale image. 
    alpha: float 0.-1.
        Shape parameter of the Tukey window.
    mask_FFT_on : bool
        If True, a mask will be applied to the FFT of the image. 
    mask_type: string
        'circular' or 'rectangular': determines the shape of the mask. 
    mask_length: float
        Length of the mask. If type is 'circular', this is the radius, and for 'rectangular,
        it is half of the rectangle's length. 
    x_offset, y_offset : float
        Mask center offset in horisontal or vertical direction 
    invert_mask : bool
        If True, the mask will keep only the region of the FFT that is inside of the mask. 
        If False, only that outside of the mask will be kept. 
    make_FFT_mask_plot : bool
        If True, the FFT, the masked FFT and the inverse tranformed masked FFT will be plotted. 
    make_back_transform_image_plot : bool
        If True, all image processing steps will be shown in a figure. 
    cmap : matplotlib.cm
        Matplotlib colormap. See e.g. for the options: 
        https://matplotlib.org/3.1.0/api/cm_api.html#matplotlib.cm.get_cmap
        Tips: by adding '_r', you invert the colormap!
    
    Returns
    -------
    FFT_image : ndarray
        FFT of image.
    intensity_image : ndarray
        Absolute value of the FFT times its complex conjugate. 
    """
    
    image, FFT_image, intensity_image = get_intensity_of_image(
        image_temp, apply_window=apply_window, alpha=alpha, plot_intensity=False)
    
    if mask_FFT_on:
        abs_masked_FFT, abs_IFFT = mask_FFT_image(FFT_image,
                                                  intensity_image,
                                                  mask_type=mask_type,
                                                  mask_length=mask_length,
                                                  x_offset=x_offset,
                                                  y_offset=y_offset,
                                                  invert_mask=invert_mask,
                                                  make_plot=make_FFT_mask_plot,
                                                  cmap=cmap) 
    else:
        abs_IFFT = np.abs(np.fft.ifft2(FFT_image))
        
    if make_back_transform_image_plot:
        if mask_FFT_on: 
            fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6, 4))#, 
                                     #sharex=True, sharey=True, subplot_kw={'adjustable': 'box'})
            ax = axes.ravel()
            ax[1].imshow(image, cmap=cmap, interpolation='none')
            ax[1].set_title('Image with window')
            ax[-3].imshow(np.log(intensity_image.astype(np.float32)), cmap=cmap, interpolation='none')
            ax[-3].set_title('FFT')
            ax[-2].imshow(np.log(abs_masked_FFT), cmap=cmap, interpolation='none')
            ax[-2].set_title('Masked FFT')
        else: 
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))
            #, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
            ax = axes.ravel()
            ax[1].imshow(image, cmap=cmap, interpolation='none')
            ax[1].set_title('Image with window')
            ax[-2].imshow(np.log(intensity_image.astype(np.float32)), cmap=cmap, interpolation='none')
            ax[-2].set_title('FFT')
        ax[0].imshow(image_temp, cmap=cmap, interpolation='none')
        ax[0].set_title('Image')
        ax[-1].imshow(abs_IFFT, cmap=cmap, interpolation='none')
        ax[-1].set_title('IFFT')
        for a in ax:
            a.set_axis_off()
        fig.tight_layout(h_pad = 0.1, w_pad = 0.1)
        fig.set_dpi(200)

    return abs_IFFT


# Adjust the input parameters to customize the image processing. 

# Check the shape of the image to get an idea of suitable values for mask_length, x_offset, y_offset.
print(np.shape(image))

back_transform = back_transform_image(image_temp=image,
                                      apply_window = False,
                                      alpha = 1.,
                                      mask_FFT_on = True,
                                      mask_type = 'circular',
                                      mask_length = 100,
                                      x_offset = 0,
                                      y_offset = 0,
                                      invert_mask = False,
                                      make_FFT_mask_plot = False,
                                      make_back_transform_image_plot = True,
                                      cmap='inferno')


# # Part 3: Diffraction from 2D lattices


def get_electron_density_2D(steplength = .1,
                            model = 'Monoatomic',
                            sigma_1 = .1,
                            sigma_2 = .2,
                            a_0 = 2.,
                            num_atoms_per_row = 5,
                            amplitude_1 = 1.,
                            amplitude_2 = 2.,
                            thermal_disorder = False,
                            epsilon = 0.2,
                            plot_electron_density = True,
                            cmap='inferno'):
    '''Calculates the total electron density of a 2D crystal.

    Parameters
    ----------
    steplength : float
        steplength, e.g. spatial discretisation
    model : string
        Model type: 'Monoatomic' or 'Diatomic'
    sigma_1, sigma_2: float
        Width of the Gaussians in the expression for electron density.
        sigma_2 is used in diatomic model, otherwise only sigma_1.
    a_0: float
        lattice constant in [Å].
    num_atoms_per_row: int
        Number of atoms per row. The total number of atoms is num_atoms_per_row**2.
    amplitude_1, amplitude_2: float
        Amplitude; factor multiplied with the Gaussian at each atom position.
        amplitude_2 is used in the diatomic model, otherwise only amplitude_1.
    thermal_disorder: bool
        True if thermal disorder is to be included in the model, and False (default) 
        for no thermal disorder.
    epsilon: float
        Limits of the random number (epsilon * random_number, where random_number
        is taken from the interval (-1,1)) that is added to the atom position. 
        Only used for thermal_disorder=True. 
    plot_electron_density: bool
        Determines if the electron density should be plotted.
    cmap : matplotlib.cm
        Matplotlib colormap. See e.g. for the options: 
        https://matplotlib.org/3.1.0/api/cm_api.html#matplotlib.cm.get_cmap
        Tips: by adding '_r', you invert the colormap!
    
    Returns
    -------
    electron_density_2D : ndarray
        The electron density in 2D. 
    '''
    
    half_length = (num_atoms_per_row-1)*a_0/2
    atom_positions = np.meshgrid(np.arange(-half_length,half_length+a_0,a_0),
                                np.arange(-half_length,half_length+a_0,a_0))    
    if model == 'Glass':
        print('Glass model is not defined in 2D.')
        return 0
    
    if thermal_disorder: 
        atom_positions[0] = atom_positions[0] + epsilon*(
                2 * np.random.random(size = np.shape(atom_positions[0])) -1)
        atom_positions[1] = atom_positions[1] + epsilon*(
                2 * np.random.random(size = np.shape(atom_positions[1])) -1)
            
    if model == 'Diatomic':
        if np.fmod(num_atoms_per_row,2): # odd:
            amplitudes = np.reshape(np.tile(np.array([amplitude_1,amplitude_2]), 
                                            int(num_atoms_per_row**2/2+num_atoms_per_row/2)),
                                    newshape = (-1,num_atoms_per_row))[:num_atoms_per_row,:]
            sigmas = np.reshape(np.tile(np.array([sigma_1,sigma_2]), 
                                            int(num_atoms_per_row**2/2+num_atoms_per_row/2)),
                                    newshape = (-1,num_atoms_per_row))[:num_atoms_per_row,:]
        else: 
            amplitudes = np.reshape(np.tile(np.array([amplitude_1,amplitude_2]), 
                                int(num_atoms_per_row**2/2+num_atoms_per_row/2)),
                        newshape = (num_atoms_per_row,-1))[:num_atoms_per_row,:num_atoms_per_row]
            sigmas = np.reshape(np.tile(np.array([sigma_1,sigma_2]), 
                                int(num_atoms_per_row**2/2+num_atoms_per_row/2)),
                        newshape = (num_atoms_per_row,-1))[:num_atoms_per_row,:num_atoms_per_row]
    else:
        amplitudes = np.zeros_like(atom_positions[0])+amplitude_1
        sigmas = np.zeros_like(atom_positions[0])+sigma_1
        
    def calc_this_atoms_electron_density_2D(atom_position_x, atom_position_y, amplitude, sigma):    
        length = int( (num_atoms_per_row) * a_0 / steplength + 1)
        center = length/2 - 0.5  
        this_atoms_electron_density_2D = np.zeros((length, length))
        X,Y = (np.indices((length, length)) - center) * steplength 
        return amplitude * np.exp(-((X - atom_position_x)**2 + (Y - atom_position_y)**2) / ( 2 * sigma**2) )

    electron_density_2D = np.sum(list(map(calc_this_atoms_electron_density_2D,
                               np.reshape(atom_positions[0],(-1,1)),
                               np.reshape(atom_positions[1],(-1,1)),
                               np.reshape(amplitudes,(-1,1)),
                               np.reshape(sigmas,(-1,1)))),
                            axis=0)
    
    if plot_electron_density:
        plt.figure(dpi = 200)
        plt.imshow(electron_density_2D, cmap=cmap)
        plt.colorbar()
        plt.minorticks_on()
        pos = np.linspace(start = 0,
                          stop = int( (num_atoms_per_row) * a_0 / steplength), 
                          num = 5, dtype = int)
        plt.xticks(pos, (pos*steplength-half_length-a_0/2))
        plt.yticks(pos, (pos*steplength-half_length-a_0/2))
        plt.xlabel('Distance [Å]')
        if thermal_disorder:
            plt.title(model+' 1D model with thermal disorder and '+str(num_atoms_per_row**2)+' atoms.')
        else:
            plt.title(model+' 1D model with '+str(num_atoms_per_row**2)+' atoms.')
    
    return electron_density_2D

# Steplength between each value on the axes in [Å].
# PS A too big value here might give you a memory error or consume time...
steplength = 0.05

# Model type: 'Monoatomic', 'Diatomic' or 'Glass', determines the expression
# for electron density.
model='Diatomic'

# Number of atoms. The total number of atoms is num_atoms_per_row**2.
num_atoms_per_row = 10

# Distance between atoms in [Å]; lattice constant.
a_0 = 2. 

# Width of the Gaussians in the expression for electron density in [Å].
# sigma_2 is used in diatomic model.
sigma_1 = .2 
sigma_2 = 0.3

# Amplitude; factor multiplied with the Gaussian at each atom position.
# amplitude_2 is used in the diatomic model.
amplitude_1 = 3
amplitude_2 = 5

# True if thermal disorder is to be included in the model, 
# and False for no thermal disorder.
thermal_disorder=False

# Only used for thermal_disorder=True. 
# Limits of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
# Only used for thermal_disorder=True. 
epsilon = 0.2

# Only used for model='glass'. 
# Limit of the random number (epsilon * random_number, where random_number
# is taken from the interval (-1,1)) that is added to the atom position. 
epsilon_glass = 0.5

electron_density_diatomic_2D = get_electron_density_2D(steplength=steplength,
                                                       model='Diatomic',
                                                       sigma_1=sigma_1,
                                                       sigma_2=sigma_2,
                                                       a_0=a_0,
                                                       num_atoms_per_row=num_atoms_per_row,
                                                       amplitude_1=amplitude_1,
                                                       amplitude_2=amplitude_2,
                                                       thermal_disorder=False)


# Optionally add some noise to the electron density
electron_density_diatomic_2D_noisy = add_noise(electron_density_diatomic_2D, sigma=0.15)[1]


# Use the same function that you used to calculate the FFT, mask and
# calculate the IFFT of images, defined in section 2, on the 2D crystal.

back_transform = back_transform_image(image_temp = electron_density_diatomic_2D_noisy,
                                      apply_window = False,
                                      mask_FFT_on = True,
                                      mask_type = 'circular',
                                      mask_length = 50,
                                      x_offset = 0,
                                      y_offset = 0,
                                      invert_mask = True,
                                      make_FFT_mask_plot = False,
                                      make_back_transform_image_plot = True)
