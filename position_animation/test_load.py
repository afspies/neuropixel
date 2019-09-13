import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-18])
from classes.DataGetter import DataGetter

path = "/home/alex/Desktop/NPix_dataset/ratL/RatL_241018.mat"
x, y, fire = DataGetter(path).get_data("clean")