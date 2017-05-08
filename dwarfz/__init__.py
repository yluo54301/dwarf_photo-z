__all__ = [
	"datasets",
	"matching",
	"hsc_release_query",
	"hsc_credentials",
]
from . import *


import os

package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_default = os.path.join(package_dir, "data")

