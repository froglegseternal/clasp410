#!/usr/bin/env python3

# First, import needed libraries.
# Numpy is used for arrays, random, and other miscellany.
import numpy as np
# Pyplot is the main plotting tool.
import matplotlib.pyplot as plt
# Import ListedColormap as an object to create a specialized color map:
from matplotlib.colors import ListedColormap


# Now, we define our constants.

nx, ny = 3, 5 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as a bare patch.
prob_start = 0.0 # Chance of cell to start on fire.
iterations = 5 # Number of times to iterate time.
 

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

    if i == 0 and j == 0: # If both indexes are zero, the cell is in the top-left corner.
        return "top left"
    elif i == 0 and j == ny - 1: # If the row index is zero and the column index is the last numbered-cell (note 1: due to zero-indexing, this is ny MINUS one),
                                ## the cell is in the top-right corner.
        return "top right"
    elif i == 0:    # If the column index is not an edge but the row index is zero, this cell is on the top.
        return "top"
    elif i == nx - 1 and j == 0: # If the row index is the last-numbered cell (1) and the column index is zero, the cell is in the bottom-left corner.
        return "bottom left"
    elif i == nx - 1 and j == ny - 1: # If both the row and column indexes are the last-numbered cell (1), the cell is in the bottom-right corner.
        return "bottom right"
    elif i == nx - 1: # If the cell isn't in any of the corners (note 2: due to prior checks) and the row index is the last-numbered cell (1), the cell is on the
                        ## bottom edge.
        return "bottom"
    elif j == ny - 1: # If the cell isn't in any of the corners (2), and the column index is the last-numbered cell (1), the cell is on the right edge.
        return "right"
    elif j == 0: # If the column index is zero, the cell is on the left edge.
        return "left"
    else:   # If none of the above are true (2), the cell is not an edge.
        return "Not an edge"



history_matrix = np.zeros([iterations+1, nx, ny], dtype =int) # 3-D array storing the historical values of our forest array.

# Create an initial grid, set all values to "2". dtype sets the value
# type in our array to integers only.
forest = np.zeros([nx, ny], dtype =int) + 2

# Set a random cell to "burning":
forest[int(np.random.rand() * nx), int(np.random.rand() * ny)] = 3

# Create an array of randomly generated number of range [0, 1):
isbare = np.random.rand(nx, ny)

# Turn it into an array of True/False values:

isbare = isbare < prob_bare
    
# We can use this array of booleans to reference any existing array
# and change only the values corresponding to True:
forest[isbare] = 1

def do_iterate(cur_iter, iter_num):
    '''
    This is the main function, responsible for iterating over time. It is recursive, calling itself as many times as specified by the iter_num parameter.

    Parameters
    ==========
    cur_iter: int
        The current iteration we are on.
    iter_num: int
        The number of iterations to perform.
    '''
    # First, check if we're currently on a bad iteration. If so, we quit.
    if(cur_iter >= iter_num):
        history_matrix[cur_iter] = forest
        return -1
    # First, make a copy of the forest. This makes it so that we can check the conditions of a cell we've already iterated over.
    history_matrix[cur_iter] = forest
    cur_iter = cur_iter + 1 # Locally defined iteration count.


    # Loop in the "x" direction:
    for i in range(nx):
        # Loop in the "y" direction:
        for j in range(ny):
            # If the cell is burning, it will stop burning and become barren.
            if is_burning(i, j, history_matrix[cur_iter-1]):
                forest[i, j] = 1
                isbare[i, j] = True
                edge_status = cell_is_edge(i, j) # Now that we know the cell is burning, figure out if our current cell is an edge cell, and if so, in what way.
                if 'left' not in edge_status and forest[i,j-1] != 1 and np.random.rand() < prob_spread:
                    forest[i,j-1] = 3
                if 'right' not in edge_status and forest[i,j+1] != 1 and np.random.rand() < prob_spread:
                    forest[i,j+1] = 3
                if 'bottom' not in edge_status and forest[i+1, j] != 1 and np.random.rand() < prob_spread:
                    forest[i+1,j] = 3
                if 'top' not in edge_status and forest[i-1, j] != 1 and np.random.rand() < prob_spread:
                    forest[i-1,j] = 3
                

            #If the cell is barren - and there should be no burning cells currently due to the previous statement - then skip to the next loop.
            if isbare[i, j]:
                continue
    do_iterate(cur_iter, iter_num) # Recursion call
            
def do_process(num_states, matrix):
    '''
    After we have run our model, this is the code that will actually process the model. It takes in the number of states in the given array, and the array to process.
    The way it works is by creating several arrays, each of which represent the change in a certain value over time. These values, in turn, are the different states
    the cells can have. It returns an array of these arrays. It assumes that the states are integers - given how the rest of the code is laid out, this should not
    be an issue, but is not the ideal representation.

    Parameters
    ==========
    num_states: int
        The number of different states in the array
    matrix: numpy array
        A 3-dimensional numpy array that is to be processed.
    Returns
    ======
    state_arrays: numpy array
        A 2-dimensional array, with each row representing a different state, and each column representing a point in time.
    '''
    state_arrays = np.zeros([num_states, iterations]) # Define our array that we will eventually return.
    for x in range(iterations): # Iterate over each moment in time.
        for i in range(nx): # Iterate over each row in the matrix we're counting.
            for j in range(ny): # Iterate over each column in the matrix we're counting.
                state_arrays[matrix[x][i][j] - 1][x] = state_arrays[matrix[x][i][j] - 1][x] + 1 # Increment the count for the state this cell is in at this specific moment in time.
    return state_arrays

def plot_last_moment():
    '''
    The function responsible for checking the state of the forest array at the last moment in time. This is its own function to allow for scalability.
    '''
    # Generate a custom segmented color map for this project.
    # Can specify colors by names and then create a colormap that only uses
    # those names. There are 3 fundamental states - barren, burning, and not yet on fire.
    # Therefore, three colors are needed. Color info can be found at the following:
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    # Create figure and set of axes:
    fig, ax = plt.subplots(1,1)

    # Set graph title
    ax.set_title("Forest Having Burnt for " + str(iterations) + " Iterations.")

    # Set axes titles.
    ax.set_xlabel("X Direction (m)")
    ax.set_ylabel("Y Direction (m)")

    # Because of the way that numpy arrays are indexed, we need to explicitly set the axes.
    plt.xlim(0,ny)
    plt.ylim(0,nx)

    # Given the "forest" object, a 2D array that contains numbers 1, 2, or 3,
    # Plot this using the "pcolor" method. Need to use our color map as well as set
    # both *vmin* and *vmax*.
    ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)
    ax.imshow(forest)

    plt.show()

def plot_statistics(stat_matrix):
    '''
    The function responsible for plotting our statistics once we have a functional model. Takes in the matrix of calculated statistics.

    Parameters
    ==========
    stat_matrix: numpy array
        A 2-dimensional array that is the output of the do_process function and represents count over time.
    '''

do_iterate(0, iterations) # Run the model
statistics = do_process(3, history_matrix) # Do the statistics