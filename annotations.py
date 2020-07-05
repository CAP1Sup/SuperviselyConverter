# Import libraries
import xml.etree.ElementTree as xml
from xml.etree import ElementTree
from xml.dom import minidom
from PIL import Image, ImageOps
from zipfile import ZipFile
from random import randint
import shutil
import time
import json
import os

# Setup constants for later use
supported_formats = '.jpg', '.JPG', '.png', '.PNG' # .............................. Specifies the allowable image formats
# allowed_percent_crop_lower = 1 # .................................................. Specifies the smallest allowable crop percentage
# allowed_percent_crop_upper = 5 # .................................................. Specifies the largest allowable crop percentage
# minimum_box_percent = 2 # ......................................................... Specifies the smallest allowable size of a box in percent of image dimension


def convert_Supervisely_2_Pascal_VOC(input_supervisely_folder, output_folder, cleanup):
    """Converts a folder of Supervisely annotations to PascalVOC format 

    :param input_supervisely_folder: The folder of Supervisely annotations and images for conversion
    :param output_folder: The folder for the final zip file
    :param cleanup: Should the Supervisely folder be deleted

    """

    # Take note of the start time
    start_time = time.time()

    # Provide feedback
    print("Creating output folder structure")

    # Check if output folder structure is present
    if not os.path.isdir(os.path.join(output_folder, "JPEGImages")):
        # Image folder doesn't exist, we need to create it
        os.mkdir(os.path.join(output_folder, "JPEGImages"))

    if not os.path.isdir(os.path.join(output_folder, "Annotations")):
        # Image folder doesn't exist, we need to create it
        os.mkdir(os.path.join(output_folder, "Annotations"))

    
    # Provide feedback
    print("Beginning counting available images")

    # Create an empty accumulator for a readout of the progress
    image_name_list = []

    # Get all of the files in the input directory
    for filename in os.listdir(os.path.join(input_supervisely_folder, 'img')):
        # Make sure that each file is an image
        for image_format in supported_formats:
            # Check if the image format is an image
            if filename.endswith(image_format):
                image_name_list.append(filename)

    # Give feedback on current process
    print("Starting conversion process for " + str(len(image_name_list)) + " images")

    # Declare counter and begin the conversion process
    current_image_index = 0
    for image_name in image_name_list:
        # Convert the original image
        convert_original_image(image_name, input_supervisely_folder, output_folder)

        # Count that this image is finished
        current_image_index = current_image_index + 1

        # Print the progress
        print("Progress: " + str(current_image_index) + "/" + str(len(image_name_list)))


    # Find all of the converted files and folders
    image_file_paths = get_all_file_paths(os.path.join(output_folder, "JPEGImages"))
    annotation_file_paths = get_all_file_paths(os.path.join(output_folder, "Annotations"))

    # Format the folder name
    unformatted_supervisely_folder_name = os.path.basename(input_supervisely_folder)
    formatted_supervisely_folder_name = unformatted_supervisely_folder_name.replace(" ", "_")

    # Write each of the files to a zip
    file_name = os.path.join(output_folder, (formatted_supervisely_folder_name + ".zip"))
    with ZipFile(file_name,'w') as zip_file: 

        # Write each image 
        for file in image_file_paths: 
            zip_file.write(file, arcname = os.path.join("JPEGImages", os.path.basename(file)))

        # Write each annotation
        for file in annotation_file_paths: 
            zip_file.write(file, arcname = os.path.join("Annotations", os.path.basename(file)))

    # Clean up the output of temporary folders
    shutil.rmtree(os.path.join(output_folder, "JPEGImages"))
    shutil.rmtree(os.path.join(output_folder, "Annotations"))

    # Clean up the Supervisely folder as well if indicated
    if cleanup:
        # Need to delete the input folder
        shutil.rmtree(input_supervisely_folder)

    end_time = time.time()

    print("Finished converting " + str(len(image_name_list)) + " images")
    print("Conversion time: " + str(round(end_time - start_time, 2)) + " seconds")


def convert_original_image(filename, input_supervisely_folder, output_folder_path):
    # Get the file's name for splitting
    raw_filename, file_ext = os.path.splitext(filename)

    # Create an image for copying
    image = Image.open(os.path.join(input_supervisely_folder, 'img', filename))

    # Save the output image to the specified directory
    image.save(os.path.join(output_folder_path, "JPEGImages", (raw_filename + ".jpg")))

    # Get annotation data from the respective annotation file
    image_objects = get_image_objects(filename, input_supervisely_folder)

    # Build an xml with the old file
    build_xml_annotation(image_objects, (raw_filename + ".jpg"), output_folder_path)


def get_image_objects(image_name, input_supervisely_folder):
    # Create a accumulator
    objects = []

    # Set up and open the json annotation file
    with open(os.path.join(input_supervisely_folder, 'ann', (image_name + '.json'))) as annotation_file:
        annotations = json.load(annotation_file)

    # For each of the objects in the annotation file, add a listing to the accumulator
    for object_ in annotations["objects"]:
        # Grab the name of the object
        object_name = object_["classTitle"]

        # Navigate to the correct level of the dictionary and grab the point data
        points = object_["points"]
        exterior_points = points["exterior"]
        left, upper = exterior_points[0]
        right, lower = exterior_points[1]

        # Add the object to the list of objects
        objects.append([object_name, left, upper, right, lower])

    return objects 


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def build_xml_annotation(objects, image_name, output_folder):

    # Get the file's name for splitting
    raw_image_name, file_ext = os.path.splitext(image_name)

    # Create master annotation element
    annotation = xml.Element('annotation')

    # Get folder name (not important)
    folder = xml.SubElement(annotation, 'folder')
    folder.text = os.path.basename( os.path.join(output_folder, "JPEGImages") )

    # Filename
    filename = xml.SubElement(annotation, 'filename')
    filename.text = image_name

    # Path 
    path = xml.SubElement(annotation, 'path')
    path.text = os.path.join(output_folder, "JPEGImages", image_name)

    # Database (not important)
    source = xml.SubElement(annotation, 'source')
    database = xml.SubElement(source, 'database')
    database.text = "None"

    # Open the image to get parameters from it
    image = Image.open(os.path.join(output_folder, "JPEGImages", image_name))
    image_width, image_height = image.size

    # Image size parameters
    size = xml.SubElement(annotation, 'size')
    width = xml.SubElement(size, 'width')
    width.text = str(image_width)
    height = xml.SubElement(size, 'height')
    height.text = str(image_height)

    # Depth is 3 for color, 1 for black and white
    depth = xml.SubElement(size, 'depth')
    depth.text = str(3)

    # Segmented (not important)
    segmented = xml.SubElement(annotation, 'segmented')
    segmented.text = str(0)
  
    # Objects... where the fun begins
    for object_list in objects:
        # Declare an object
        object_ = xml.SubElement(annotation, 'object')

        # Name
        name = xml.SubElement(object_, 'name')
        name.text = object_list[0]

        # Bounding box
        bndbox = xml.SubElement(object_, 'bndbox')
        xmin = xml.SubElement(bndbox, 'xmin')
        ymin = xml.SubElement(bndbox, 'ymin')
        xmax = xml.SubElement(bndbox, 'xmax')
        ymax = xml.SubElement(bndbox, 'ymax')
        xmin.text = str(round(object_list[1]))
        ymin.text = str(round(object_list[2]))
        xmax.text = str(round(object_list[3]))
        ymax.text = str(round(object_list[4]))
    
    with open(os.path.join(output_folder, "Annotations", (raw_image_name + '.xml')), 'w') as xml_file:
        xml_file.write(prettify_xml(annotation))
        xml_file.close()
    

def get_all_file_paths(directory): 
  
    # initializing empty file paths list 
    file_paths = []
  
    # Get all of the files in the  directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
  
    # returning all file paths 
    return file_paths













# For later... maybe?

# ! Broken
"""
def flip_image(filename, output_folder_path):
    # Get the file's name for splitting
    raw_filename, file_ext = os.path.splitext(filename)

    # Create an image for manipulation
    image = Image.open(input_supervisely_folder_path + 'img/' + filename)

    # Flip the image
    flipped_image = ImageOps.mirror(image)

    image_width, image_height = flipped_image.size

    # Save the output image to the specified directory
    new_filename = raw_filename + '_flipped' + file_ext
    flipped_image.save(output_folder_path + new_filename)

    # Get the annotation data from the respective annotation file
    image_objects = get_image_objects(filename)

    # Apply adjustments
    for object_ in image_objects:
        # Flip only the X coords, not the Ys
        object_offset = image_width - object_[1]
        object_[1] = object_offset

        object_offset = image_width - object_[3]
        object_[3] = object_offset

    # Save a new xml file for the annotation data
    build_xml_annotation(image_objects, new_filename, output_folder_path)
"""

# ! Broken
"""
def crop_image(filename, output_folder_path):
    # Get the file's name for splitting
    raw_filename, file_ext = os.path.splitext(filename)

    # Create an image for manipulation
    image = Image.open(input_supervisely_folder_path + 'img/' + filename)

    # Get the bounding box of the image
    (left, upper, right, lower) = image.getbbox()

    # Get random values for the adjustment by getting a random percent
    left_adjust = right * (randint(allowed_percent_crop_lower, allowed_percent_crop_upper) / 100)
    upper_adjust = lower * (randint(allowed_percent_crop_lower, allowed_percent_crop_upper) / 100)
    right_adjust = right * (randint(allowed_percent_crop_lower, allowed_percent_crop_upper) / 100)
    lower_adjust = lower * (randint(allowed_percent_crop_lower, allowed_percent_crop_upper) / 100)

    # Adjust the coordinates for the changes
    left = left + left_adjust
    upper = upper + upper_adjust
    right = right - right_adjust
    lower = lower - lower_adjust

    # Crop the image to the specified size
    cropped_image = image.crop((left, upper, right, lower))

    # Get the size of the image for sanity checking
    cropped_image_width, cropped_image_height = cropped_image.size

    # Save the output image to the specified directory
    new_filename = raw_filename + '_cropped' + file_ext
    cropped_image.save(output_folder_path + new_filename)

    # Get the annotation data from the respective annotation file
    image_objects = get_image_objects(filename)

    # Apply adjustments
    current_object = 0
    for object_ in image_objects:
        # Create a check that will delete the object if it is no longer on screen
        invalid_coordinates = False

        # Cycle through the coordinates, ajusting and checking each one
        for coordinate_num in range(1, 5):
            if (coordinate_num % 2) == 0:
                # Number is even, and is a Y coordinate
                object_[coordinate_num] = object_[coordinate_num] - upper_adjust

            else:
                # Number is odd, is an X coordinate
                object_[coordinate_num] = object_[coordinate_num] - left_adjust


        # Sanity checking to make sure the the coords are still valid
        # Check to see if the coordinate is out of bounds

        # X
        if object_[1] > cropped_image_width:
            # Whole box is out of range
            invalid_coordinates = True
        elif object_[3] > cropped_image_width:
            # Only the right of the box is out of range, we can just move the right coord to meet the right side of the image
            object_[3] = cropped_image_width

            # Make sure that the adjusted box isn't too small
            box_width = object_[3] - object_[1]
            smallest_allowable_width = cropped_image_width * (minimum_box_percent / 100)

            if box_width < smallest_allowable_width:
                # Box is too small
                invalid_coordinates = True

        # Y
        if object_[2] > cropped_image_height:
            # Whole box is out of range
            invalid_coordinates = True
        elif object_[4] > cropped_image_height:
            # Only the bottom of the box is out of range, we can just move the lower coord to meet the bottom of the image
            object_[4] = cropped_image_height

            # Make sure that the box isn't too small
            box_height = object_[4] - object_[2]
            smallest_allowable_height = cropped_image_height * (minimum_box_percent / 100)

            if box_height < smallest_allowable_height:
                # Box is too small
                invalid_coordinates = True


        # Delete the coordinate if it is invalid
        if invalid_coordinates:
            unused = image_objects.pop(current_object)
        
        # Continue to count the current object
        current_object = current_object + 1

    # Save a new xml file for the annotation data
    build_xml_annotation(image_objects, new_filename, output_folder)
"""