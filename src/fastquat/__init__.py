from importlib.metadata import version

from .quaternion import Quaternion

__all__ = ['Quaternion']
__version__ = version('fastquat')
