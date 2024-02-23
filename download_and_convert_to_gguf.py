import argparse
import logging
import os
import subprocess
from huggingface_hub import snapshot_download

# setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# setup parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "-repo_id",
    default="microsoft/phi-2",
    type=str,
    help="Example: microsoft/phi-2",
)
parser.add_argument(
    "-o",
    "-outtype",
    default="f32",
    choices=["f32", "f16", "q8_0"],
    type=str,
    help="Example: f32",
)
parser.add_argument(
    "-d",
    "-download_folder",
    default="./downloaded_models/",
    type=str,
    help="Best to just leave this as is \
unless you have a specific use case",
)
parser.add_argument(
    "-c",
    "-converted_models_folder",
    default="./converted_models/",
    type=str,
    help="Best to just leave this as is \
unless you have a specific use case",
)

# parse the arguments and log the input
args = parser.parse_args()

model_id = args.r  # "microsoft/phi-2"
outtype = args.o  # "f32"
downloaded_models_folder = args.d  # "./downloaded_models/"
converted_models_folder = args.c  # "./converted_models/"

model_name = model_id.split("/")[1]  # phi-2
model_path = f"{downloaded_models_folder}{model_id}/"  # ./downloaded_models/microsoft/phi-2/
converted_models_output_folder = (
    f"{converted_models_folder}{model_id}/"  # ./converted_models/microsoft/phi-2/
)
outfile = f"{converted_models_output_folder}{model_name}_{outtype}.gguf"  # ./converted_models/microsoft/phi-2/phi-2_f32.gguf

# -create the conversion command using our arguments
bash_convert_to_gguf = f"""python3 llama.cpp/convert-hf-to-gguf.py {model_path} \
--outfile {outfile} \
--outtype {outtype}"""
# example output of the bash command:
# python3 llama.cpp/convert-hf-to-gguf.py ./downloaded_models/microsoft/phi-2/ \
# --outfile ./converted_models/microsoft/phi-2/phi-2_f32.gguf \
# --outtype f32

# stupidly verbose logging for all the inputs, i dont like this formatting also.
logging.info(
    "Arguments: {}, Model ID: {}, Outtype: {}, Download Folder: {}, Converted Models Folder: {}, Model Path: {}, Model Name: {}, Converted Models Output Folder: {}, Outfile Name: {}, Bash Command: {}".format(
        args,
        model_id,
        outtype,
        downloaded_models_folder,
        converted_models_folder,
        model_path,
        model_name,
        converted_models_output_folder,
        outfile,
        bash_convert_to_gguf,
    )
)

# verify the folders exist, if not create them
logging.info(
    f"Checking if the root folders {downloaded_models_folder} and {converted_models_folder} exist"
)
if not os.path.exists(downloaded_models_folder) or not os.path.exists(converted_models_folder):
    try:
        os.makedirs(downloaded_models_folder)
        os.makedirs(converted_models_folder)
    except Exception as e:
        logging.info(f"Error creating root folders needed: {e}")
else:
    logging.info(
        f"Root folders {downloaded_models_folder} and {converted_models_folder} already exist"
    )

# download the model from huggingface, if the model exists on the file system, do not download
logging.info(f"Checking if model {model_id} exists on the file system in {model_path}")
if not os.path.exists(model_path):
    logging.info(f"Model {model_id} does not exist on the file system, downloading...")
    # using a try except block to capture if the model exists on huggingface
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            revision="main",
        )
        logging.info(f"Model {model_id} downloaded successfully to {model_path}")
    except Exception as e:
        logging.info(
            f"Could not find model '{model_id}' on huggingface, please check the model id and try again"
        )
        logging.info(f"Error: {e}")
        exit()
else:
    logging.info(
        f"Model {model_id} already exists on the file system, skipping download \
if you would like to or need to redownload the model, please delete the folder \
{model_path} and try again"
    )
logging.info(f"Model {model_id} exists on the file system in {model_path}")

if not os.path.exists(converted_models_output_folder):
    try:
        os.makedirs(converted_models_output_folder)
    except Exception as e:
        logging.info(f"Error creating converted models folder: {e}")
else:
    logging.info(
        f"Converted models folder {converted_models_output_folder} exists"
    )
logging.info(
    "Starting Conversion...this may take a while..do not close the terminal window until you see confirmation, thanks!"
)

output = subprocess.check_output(["bash", "-c", bash_convert_to_gguf])

logging.info(output)
logging.info(f"Conversion completed...converted model saved to {outfile}")
