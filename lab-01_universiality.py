#!/usr/bin/env python3

# First, import needed libraries.
# Numpy is used for arrays, random, and other miscellany.
import numpy as np
# Pyplot is the main plotting tool.
import matplotlib.pyplot as plt
# Import ListedColormap as an object to create a specialized color map:
from matplotlib.colors import ListedColormap

# Now, we define our constants.

nx, ny = 3, 3 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as a bare patch.
prob_start = 0.0 # Chance of cell to start on fire.


def is_burning(i, j, forest_array):
    '''
    Given the coordinates of a cell and an array of forest conditions, return a boolean of whether the cell is on fire currently. 

    Parameters
    ==========
    i: int
        The x-coordinate of the cell in question.
    j: int
        The y-coordinate of the cell in question.
    forest_array: numpy array
        The coordinate grid to check.
    
    Returns
    =======
    on_fire: boolean
        Whether or not the cell in question is currently on fire.
    '''
    return forest_array[i, j] == 3

def cell_is_edge(i, j):
    '''
    Given a cell's x-coordinate and y-coordinate, return a string representing the edge status (i.e., "not an edge", "top edge",
    "top-left cell", and so on.)

    Parameters
    ==========
     i: int
        The x-coordinate of the cell in question.
     j: int
        The y-coordinate of the cell in question. 

    Returns
    =======
    cell_status: string
        A string qualitatively indicating the edge status of the cell.
    '''
 #   print(str(i)+","+str(j))
 #   print(forest[0,2])
 #   print(forest)
 #   print('-----')
    if i == 0 and j == 0: # If both indexes are zero, the cell is in the bottom-left corner.
        return "bottom left"
    elif i == 0 and j == ny - 1: # If the x-index is zero and the y-index is the last numbered-cell (note 1: due to zero-indexing, this is ny MINUS one),
                                ## the cell is in the top-left corner.
        return "top left"
    elif i == 0:    # If the y-index is not an edge but the x-index is zero, this cell is on the left edge.
        return "left"
    elif i == nx - 1 and j == 0: # If the x-index is the last-numbered cell (1) and the y-index is zero, the cell is in the bottom-right corner.
        return "bottom right"
    elif i == nx -1 and j == ny - 1: # If both the x- and y- indexes are the last-numbered cell (1), the cell is in the top-right corner.
        return "top right"
    elif i == nx - 1: # If the cell isn't in any of the corners (note 2: due to prior checks) and the x-index is the last-numbered cell (1), the cell is on the
                        ## right edge.
        return "right"
    elif j == ny - 1: # If the cell isn't in any of the corners (2), and the y-index is the last-numbered cell (1), the cell is on the top edge.
        return "top"
    elif j == 0: # If the y-index is zero, the cell is on the bottom edge.
        return "bottom"
    else:   # If none of the above are true (2), the cell is not an edge.
        return "Not an edge"





# Create an initial grid, set all values to "2". dtype sets the value
# type in our array to integers only.
forest = np.zeros([ny, nx], dtype =int) + 2

# Set the center cell to "burning":
forest[1, 1] = 3

# Create an array of randomly generated number of range [0, 1):
isbare = np.random.rand(nx, ny)

# Turn it into an array of True/False values:

isbare = isbare < prob_bare
    
# We can use this array of booleans to reference any existing array
# and change only the values corresponding to True:
forest[isbare] = 1

def iterate():
    '''
    This is the main function. It takes in no arguments, since everything it modifies is defined globally.
    '''
    # First, make a copy of the forest. This makes it so that we can check the conditions of a cell we've already iterated over.
    forest_copy = forest
    # Loop in the "x" direction:
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            print("-----")
            print("start of the loop: " + str(i)+","+str(j))
            print(forest)
            # If the cell is burning, it will stop burning and become barren.
            if is_burning(i, j, forest_copy):
                forest[i, j] = 1
                isbare[i, j] = True
            print("fire becomes barren now.")
            print(forest)
            #If the cell is barren - and there should be no burning cells currently due to the previous statement - then skip to the next loop.
            if isbare[i, j]:
                print("current cell is barren.")
                continue
            edge_status = cell_is_edge(i, j) # Now that we know the cell is not burning or barren, figure out if our current cell is an edge cell, and if so, in what way.
            if 'left' not in edge_status: # If the cell is not on the left, then there is a cell to the left which we should check to potentially spread to this cell.
                if is_burning(i-1,j,forest_copy): # If the cell to the left is currently burning, determine whether it should spread.
                    if np.random.rand(1)[0] < prob_spread: # Based on the probability of spreading, decide whether the fire should spread.
                        forest[i,j] = 3 # Set the cell's state to currently burning.
                        continue # We don't need to check any other cells if we already know the fire is going to spread.
            if 'right' not in edge_status: # If the cell is not on the right, then there is a cell to the right which we should check to potentially spread to this cell.
                if is_burning(i+1,j,forest_copy):   # If the cell to the right is currently burning, determine whether it should spread.
                    if np.random.rand(1)[0] < prob_spread:  # Based on the probability of spreading, decide whether the fire should spread.
                        forest[i,j] = 3 # Set the cell's state to currently burning.
                        continue # We don't need to check any other cells if we already know the fire is going to spread.
            if 'top' not in edge_status: # If the cell is not on the top, then there is a cell above which we should check to potentially spread to this cell.
                if is_burning(i,j+1,forest_copy): # If the cell above is currently burning, determine whether it should spread.
                    if np.random.rand(1)[0] < prob_spread:  # Based on the probability of spreading, decide whether the fire should spread.
                        forest[i,j] = 3 # Set the cell's state to currently burning.
                        continue # We don't need to check any other cells if we already know the fire is going to spread.
            if 'bottom' not in edge_status: # If the cell is not on the bottom, then there is a cell above which we should check to potentially spread to this cell.
                if is_burning(i, j-1,forest_copy):  # If the cell below is currently burning, determine whether it should spread.
                    if np.random.rand(1)[0] < prob_spread:  # Based on the probability of spreading, decide whether the fire should spread.
                        forest[i,j] = 3 # Set the cell's state to currently burning.
                        continue # We don't need to check any other cells if we already know the fire is going to spread.

# Generate a custom segmented color map for this project.
# Can specify colors by names and then create a colormap that only uses
# those names. There are 3 fundamental states - barren, burning, and not yet on fire.
# Therefore, three colors are needed. Color info can be found at the following:
# https://matplotlib.org/stable/gallery/color/named_colors.html
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

# Create figure and set of axes:
fig, ax = plt.subplots(1,1)

# Given the "forest" object, a 2D array that contains numbers 1, 2, or 3,
# Plot this using the "pcolor" method. Need to use our color map as well as set
# both *vmin* and *vmax*.
ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)