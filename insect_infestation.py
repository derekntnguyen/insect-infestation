#  -*- coding: utf-8 -*-
'''
P10 Bore, Baby, Bore

Simulation of a wood-boring insect infestiation on a forest
Boundary conditions: wall or periodic

Author: Derek Nguyen
Created: 2020-04-24
Modified: 2020-04-24
Due: 2020-04-24
'''

import numpy as np
import scipy.ndimage as ndimage
import matplotlib # used to create interactive plots in the Hydrogen package of the Atom IDE
matplotlib.use('Qt5Agg') # used to create interactive plots in the Hydrogen package of the Atom IDE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

import matplotlib # used to create interactive plots in the Hydrogen package of the Atom IDE
matplotlib.use('Qt5Agg') # used to create interactive plots in the Hydrogen package of the Atom IDE

def borebabybore(density = 0.6, neighborhood = 'vonNeumann', radius = 1, nGen = None,
         pbc = False, grid = True):
    '''
    Cellular automata simulation of a forest infestation based on percolation
    theory. The forest is initially randomly seeded with trees at a parameter,
    density, and then undergoes infestiation utilzing ndimage.generic_filter
    function.


    Derek Nguyen


    Parameters:
        density: seed pattern of trees at time zero
        neighborhood: 'Moore' or 'vonNeumann'
        radius: radius to apply to the neighborhood, 1 or 2
        nGen: number of generations; if None then animation automatically stops
              when no more trees are infested
        pbc: periodic (True) or deadzone (False) boundary conditions
        grid: gridlines on (True) or off (False) (adding grid slows performance)
    '''

    #At bit o' error handling
    if neighborhood == 'Moore':
        if radius == 1:
            mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        elif radius == 2:
            mask = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 0, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],])
        else:
            raise ValueError('radius must be 1 or 2')
    elif neighborhood == 'vonNeumann':
        if radius == 1:
            mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        elif radius == 2:
            mask = np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 0, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0],])
        else:
            raise ValueError('radius must be 1 or 2')
    else:
        raise ValueError("neighborhood must be 'Moore' or 'vonNeumann'")

    #Set the mode parameter for sp.ndimage.gaussian_filter() function
    if pbc:
        bc_mode = 'wrap' #periodic boundary conditions
    else:
        bc_mode = 'constant' #deadzone boundary conditions

    M = 70 #row grid size
    N = 140 #column grid size

    #Seed the forest at specified density
    z = np.random.binomial(1, density, (M,N))

    #Set all trees in the center 5x5 on fire
    for row in range(32,37):
        z[row,67:71] = [2 if i == 1 else 0 for i in z[row,67:71]]

    """
    Define colormap to use. Can pick from the many built-in colormaps, or,
    as shown below, create our own. First create a dictionary object with
    keys for red, green, blue. The value for each pair is a tuple of tuples.
    Must have at least 2 tuples per color, but can have as many as you wish. The
    first element in each tuple is the position on the colormap, ranging from
    0 (bottom) to 1 (top). For our CA, each position corresponds to one of the
    four possible states: 0 (bare ground), 1 (green tree) , 2 (burning tree),
    3 (burned-out stump).

    The second element is the brightness (gamma) of the color. The third element
    is not used when we only have two tuples per color. The conventional red-green
    blue (RGB) color scale has gamma values ranging from 0 to 255 (256 total
    levels); these are normalized 0 to 1. I.e., a gamma of 1 in the color tuple
    denotes gamma 255.

    The code below creates a color map with
        state 0 => position = 0.0   white (255, 255, 255) => (1, 1, 1)
        state 1 => position = 0.33  green (0, 204, 0) => (0, 0.8, 0)
        state 2 => position = 0.67  orange (255, 102, 0) => (1, 0.4, 0)
        state 3 => position = 1.0   grey (120, 120, 120) => (0.47, 0.47, 0.47)
    """
    cdict = {'red':   ((0.00, 1.00, 1.00),
                       (0.33, 0.00, 0.00),
                       (0.67, 1.00, 1.00),
                       (1.00, 0.47, 0.47)),
             'green': ((0.00, 1.00, 1.00),
                       (0.33, 0.80, 0.80),
                       (0.67, 0.40, 0.40),
                       (1.00, 0.47, 0.47)),
             'blue':  ((0.00, 1.00, 1.00),
                       (0.33, 0.00, 0.00),
                       (0.67, 0.00, 0.00),
                       (1.00, 0.47, 0.47))}

    #Now create the colormap object
    colormap = colors.LinearSegmentedColormap('mycolors', cdict, 256)

    #Set up plot object
    fig, ax = plt.subplots()
    plt.axis('scaled')
    plt.axis([0, N, 0, M])
    #pcolormesh creates a quadmesh object. vmin and vmax specify the min and
    #max values in z. If these are not specified, then if z is all zeros (or
    #all ones),  plot won't work because function doesn't know what color
    #to use if all values are the same.

    cplot = plt.pcolormesh(z, cmap = colormap, vmin = 0, vmax = 3)

    plt.title('Forest infestation with initial density = ' + str(density) +
              '\nGeneration 0')

    if grid:
        #Adding gridlines seems a bit more complicated than it should be...
        plt.grid(True, which = 'both', color = '0.5', linestyle = '-')
        plt.minorticks_on()
        xminorLocator = ticker.MultipleLocator(1)
        yminorLocator = ticker.MultipleLocator(1)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

    stopFlag = False

    # setting denominator for probabilities
    if neighborhood == 'vonNeumann'and radius == 1:
        denom = 5
    elif neighborhood == 'vonNeumann'and radius == 2:
        denom = 15
    elif neighborhood == 'moore'and radius == 1:
        denom = 10
    elif neighborhood == 'moore'and radius == 2:
        denom = 30

    nGeneration = 1
    while not stopFlag:
        if nGeneration == nGen:
            stopFlag = True
        nInfested = ndimage.generic_filter(z==2, np.sum, footprint = mask,
                                       mode = bc_mode, output = int)

        #Rules rule!
        """
        States: bare (0), green tree (1), infested tree (2)
        Rules:
          rule 1: green tree with one or more infested neighbors has a chance to become infested
        """
        r1 = (z == 1) & (nInfested > 0)
        r2 = (z == 2)

        # applying the probabilities chance to spread is nInfested/denominator denoted above and in P10 document
        counts1 = np.count_nonzero(nInfested == 1)
        counts2 = np.count_nonzero(nInfested == 2)

        print(counts1)
        if nGeneration == 1:
            p = (counts1)/(denom * counts1)
        else:
            p = (counts1)/(denom * counts2)
        print(p)
        if np.random.binomial(1,p) == 1:
            z[r1] = 2


        cplot.set_array(z.ravel()) #set_array requires a 1D array (no idea why...)
        plt.title('Forest infestation with initial density = ' + str(density) +
                  '\nGeneration ' + str(nGeneration))
        plt.pause(0.4)

        if not (z == 2).any(): #if no trees are infested
            stopFlag = True
        else:
            nGeneration += 1
