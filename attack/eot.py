"""
eot.py

Differentiable Expectation Over Transformation (EOT)

Author:
    Rishab Shetty
"""

class EOT:

    def __init__(self, cfg):
        self.cfg = cfg

    def random_rotation(self, image):
        raise NotImplementedError

    def random_scale(self, image):
        raise NotImplementedError

    def random_brightness(self, image):
        raise NotImplementedError

    def random_contrast(self, image):
        raise NotImplementedError

    def apply(self, image):
        raise NotImplementedError

    def __call__(self, image):
        return self.apply(image)

    def __repr__(self):
        return "EOT()"