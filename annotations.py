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

