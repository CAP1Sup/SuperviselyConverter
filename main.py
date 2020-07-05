# Import libraries
from annotations import convert_Supervisely_2_Pascal_VOC, get_image_objects
import os

# Testing parameters
input_folder = os.path.abspath("./testing/input/Dual Balls Video/")
output_folder = os.path.abspath("./testing/output")

# Parameters


def main():
    convert_Supervisely_2_Pascal_VOC(input_folder, output_folder, False)

main()