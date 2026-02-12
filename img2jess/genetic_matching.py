import numpy as np
import pandas as pd

def im2cartesian(img:np.ndarray):
    """
    Docstring for im2cartesian
    Given an Input Image it return the data 
    as a table with columns x,y,R,G,B
    :param img: Input Image
    :type img: np.ndarray
    """

    assert img.ndim > 1 and "Images must be passed as 2-D Arrays"

    side_x = img.shape[1] / 2
    side_y = img.shape[0] / 2
    xs, ys = np.indices(img.shape[:2]).reshape(2,-1)

    if img.ndim == 2:
        vec = img.flatten()
        colours = np.stack([vec,vec,vec], axis = 0)
    else:
        colours = img.reshape(-1, 3).transpose()
    print(xs.shape, ys.shape,colours.shape)
    df = pd.DataFrame()
    df['x'] = xs - side_x
    df['y'] = ys - side_y
    df[['R','G',"B"]] = colours.T
    return df

def color_fitness():
    """
    Docstring for color_fitness
    """
    pass


def mutate(template:np.ndarray):
    """
    Docstring for mutate
    
    :param template: Description
    :type template: np.ndarray
    """
    pass

def combine(parent1:np.ndarray, parent2:np.ndarray):
    pass

def initialize_population():
    """
    Docstring for initialize_population
    """
    pass




from skimage import data

print(im2cartesian(data.astronaut()))

