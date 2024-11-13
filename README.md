# Project Idea


## Installation 
we might need to install additional libraries or packages for various reasons

[Uploading essentials.py…]()import os
import openai
HOME = os.getcwd()
print("HOME:", HOME)
openai.api_key = "sk-proj-KPC79c1Npr4wH8tXwu5PT3BlbkFJFzEsNl3QeKIzoGCtPntg"

directories = [
    "data",
    "data_removal",
    "data_creation",
    "data_dalle",
    "data_powerpaint",
    "data_controlNet",
    "data_byexample",
    "data_lama",
    "data_sd",
    "data_rem_rep"
]

home_dir = os.path.expanduser("~")  
for directory in directories:
    os.makedirs(os.path.join(home_dir, directory), exist_ok=True)

weights_dir = os.path.expanduser("~/weights")
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
file_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
os.makedirs(weights_dir, exist_ok=True)
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("File downloaded successfully.")
else:
    print("Failed to download the file.")

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

from transformers import BlipProcessor, BlipForConditionalGeneration
des_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
des_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def setup_model():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)

def count_images_and_masks(images_dir, masks_dir):
    images_count = 0
    mask_images = set()
    for image_name in os.listdir(images_dir):
        if image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            images_count += 1
            image_base = os.path.splitext(image_name)[0]
            mask_images.add(image_base)
    unique_mask_images = set()
    for mask_name in os.listdir(masks_dir):
        if mask_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            mask_base = mask_name.split('_')[0]
            if mask_base in mask_images:
                unique_mask_images.add(mask_base)
    images_with_masks_count = len(unique_mask_images)
    return images_count, images_with_masks_count



Use the package manager [pip](https://pip.pypa.io/en/stable/) to install openai==0.28.
pip install Pillow
pip install torch torchvision opencv-python pytorch-lightning timm tqdm
Install using pip install torch torchvision opencv-python pytorch-lightning timm tqdm.

## Packages 

**Some Required Packages:**
**TensorFlow:** A powerful machine learning framework for building and training models.
**NumPy:** A fundamental library for numerical computing in Python.
**Matplotlib:** A library for creating static, animated, and interactive visualizations.
**os** Provides a way to interact with the operating system, including file and directory operations, environment variables, and process management.
**shutil** Offers high-level file operations like copying, moving, and deleting files and directories.
**sys** Provides access to system-specific parameters and functions, such as command-line arguments, standard input/output, and exit codes.
**csv** Enables reading and writing CSV files.
**random** Generates pseudorandom numbers and sequences.
**torch** A powerful machine learning framework, often used for deep learning tasks. It provides tensor operations, automatic differentiation, and neural network building blocks.
**pandas**A data analysis and manipulation library. It offers data structures like DataFrames and Series, along with tools for data cleaning, analysis, and visualization.
**json** Provides functions for encoding and decoding JSON data.
**openai** An API for interacting with OpenAI's language models, allowing you to generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way
**joblib**A library for parallel computing and persistent storage of large data sets. It's often used to speed up machine learning pipelines.

## Usage 

### DALLE 2 - 7 Modes

#### Installation:
Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
Use code with caution.

#### Dependencies:
cd your-repo-name   
pip install -r requirements.txt
Use code with caution.

#### Prepare Your Data:
Ensure your data is in a clean and structured format, such as CSV or Excel.
python data_analyzer.py --input_file your_data.csv

### DALLE NEW

convert_jpg_to_png(image_path, save_path):
This function takes two arguments:
image_path: The file path of the JPG image you want to convert.
save_path: The desired file path where you want to save the converted PNG
convert_jpg_to_png("/path/to/your/image.jpg", "/path/to/output/image.png")

#### To convert an image from one format to another:
from image_converter import convert_image

input_file = "input_image.jpg"
output_file = "output_image.png"
convert_image(input_file, output_file, "png")

#### To convert a text message to Morse code:
from morse_code_converter import text_to_morse

text = "Hello, World!"
morse_code = text_to_morse(text)
print(morse_code)

#### Prepare your data:

Ensure your data is in a clean CSV or Excel format.
Make sure column names are clear and consistent.
Run the script:

python data_cleaner.py --input_file your_data.csv --output_file cleaned_data.csv
Use code with caution.

Replace your_data.csv with the path to your input file.
The script will clean the data and save it to cleaned_data.csv.

### Removement & Replacement
Ensure you have the following libraries installed:
OpenAI API
Pillow (PIL Fork)
Torchvision
Other necessary libraries based on your chosen image processing models.

Configure the Script:
Update the folder paths (data_png_folder, transparent_folder, crop_folder, output_dalle_creation_folder) to match your data locations.
Set the LEVELS variable to define the desired editing actions for each image.
Configure the des_processor and des_model with your chosen image processing models and their corresponding configuration.

## Most Important Function we used 

### DALLE 2 - 7 Modes

#### Image Conversion Script
This Python script provides functions to convert JPEG images to PNG format and process images within a specified folder.
python image_converter.py

#### Image Processing Pipeline with Object Detection and Segmentation
This Python script provides a pipeline for processing images, including object detection, segmentation, and visualization. It utilizes YOLOv10 for object detection and a pre-trained segmentation model (SAM) for semantic segmentation.
python your_script_name.py

### Functionality:

#### ImageDataset Class:
Loads images from a specified directory.
Filters images based on file extensions (e.g., JPEG, PNG).

#### Object Detection:
Uses a pre-trained YOLOv10 model to detect objects within images.
Returns bounding box coordinates and class labels for detected objects.

#### Segmentation:
Leverages a pre-trained SAM (Segmentation Attention Module) model to segment objects within images.
Generates binary masks for detected objects.

#### Visualization:
Displays the original image with bounding boxes for detected objects and overlays segmentation masks.


### DALLE NEW

#### Image Conversion Script: JPEG to PNG
This Python script provides functionalities to convert JPEG images from a specified folder to PNG format within another designated folder.
python image_converter.py 

### Functionality:

#### yolov10_detection: 
Takes a YOLOv10 model and an image batch as input.
Runs object detection on the image batch and returns detected bounding boxes and corresponding class labels.

#### show_mask and show_box:
Helper functions for visualizing segmentation masks and bounding boxes on images using Matplotlib.

#### non_overlapping_masks:
Filters non-overlapping segmentation masks based on a user-defined Intersection-over-Union (IoU) threshold.

#### process_image:
The core function that processes individual images.
Performs object detection with YOLOv10 to extract bounding boxes.
Leverages the SAM predictor to generate segmentation masks within the bounding boxes.
Filters masks based on a minimum area threshold.
Saves segmentation masks, crops images based on masks, and generates visualizations.
Writes image information and segmentation details to a CSV file.

#### process_images_in_folder:
Reads images from a specified directory.
Processes each image using process_image.
Creates output directories for masks, temporary files, and cropped images.
Tracks and reports missing or extra masks compared to input images.

### make_black_transparentge:
Making Black Pixels Transparent in Images
This code provides functionalities to convert black pixels in images to transparent pixels, effectively making the black areas clear. It caters to processing individual images and handling entire folders containing images.
python image_transparency.py

### Functionality :

#### make_black_transparent function:
Takes an image path and an optional save path as arguments.
Opens the image using Pillow library.
Converts the image mode to "RGBA" to enable transparency.
Iterates through each pixel's color information (RGBA channels).
Replaces pixels with completely black values (R=0, G=0, B=0) with a transparent value (RGBA = 0, 0, 0, 0).
Updates the image data with the modified pixel information.
Optionally saves the modified image with transparency to a specified path.

#### process_images_in_folder function:
Takes the path to a temporary folder containing images and the output folder path as arguments.
Creates the output folder if it doesn't exist.
Iterates through all files in the temporary folder.
Filters for image files with .png or .jpg extensions.
Constructs the full path for each image file.
Creates a filename with "_transparent.png" appended for the output image with transparency.
Constructs the full output path.
Calls the make_black_transparent function to convert black pixels to transparent and save the modified image.
Prints a message indicating successful processing of each image.


### get_image_prompt
This code facilitates image editing using DALL-E 2, focusing on specific areas within the image based on provided masks and user-defined actions.

### Functionalities:

##### get_image_prompt:
Takes an image path, crop paths, processor, model, and action list as input.
Opens the image and iterates through provided crop paths with corresponding actions (replacement, removal, or creation).
Handles potential missing crop paths and generates descriptions for the full image and each masked area.
Uses the OpenAI API to generate a comprehensive prompt for DALL-E 2 based on descriptions and actions.

#### save_json_with_check:
Saves captions (descriptions and prompts) in a JSON file with error handling and checking for existing data.

#### process_image_pair:
Takes an image path, mask paths, crop paths, output path, processor, model, and action list as input.
Checks if an output image already exists (avoids redundant processing).
Calls get_image_prompt to generate the DALL-E 2 prompt.
Utilizes the OpenAI API to edit the image using the generated prompt and provided masks.
Saves the edited image, generates captions, and displays a comparison visualization (original vs. edited) using Matplotlib.

### Removement & Replacement 
This Python script leverages the power of LaMa for seamless image inpainting. It addresses specific image regions identified using object detection and segmentation techniques.

### Functionality:

### Preprocessing:
Converts JPEG images to PNG for compatibility (if necessary).

#### Mask Generation:
Employs YOLOv10 to detect objects and create bounding boxes.
Utilizes a segmentation model (potentially, Sam model) to predict masks within the bounding boxes.
Refines and selects appropriate masks based on area and potential overlaps.
Saves masks, crops corresponding image regions, and generates visualizations for debugging.

#### Inpainting:
Performs inpainting on masked regions using the LaMa model.
Options include mask feathering, image upscaling for better quality, and post-processing (median filtering and downscaling).
Saves the inpainted image.

#### Visualization:
Optionally visualizes the original image, mask, and inpainted result for comparison.



## License
This a project by Laila

