from skimage.util import random_noise
from PIL import Image
import numpy as np


class SpNoise(object):
    """ Add random salt and pepper noise. Must be called before
        the ToTensor transform.
    Args:
        amount (float): Amount of noise percentage from 0.0 to 1.0
    """
    def __init__(self, amount):
        assert isinstance(amount,(float))
        self.amount = amount

    def __call__(self,sample):
        imarray = np.array(sample)
        noisy = random_noise(imarray, mode='s&p', amount=self.amount, clip=True)
        nimage = Image.fromarray(np.uint8(noisy*255))

        return nimage

class GausNoise(object):
    """Add Gaussian noise. Must be called before ToTensor transform.
       Converts PIL Image to ndarray, performs transform, returns
       PIL Image.
    
    Args:
        seed (int): Random generator seed.
    """
    def __init__(self, seed):
        assert isinstance(seed,(int))
        self.seed = seed

    def __call__(self,sample):
        imarray = np.array(sample)
        noisy = random_noise(imarray, mode='gaussian', seed=self.seed, clip=True)
        nimage = Image.fromarray(np.uint8(noisy*255))

        return nimage