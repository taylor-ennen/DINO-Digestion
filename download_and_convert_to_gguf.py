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
    "r", "repo_id", default="microsoft/phi-2", type=str, help="Example: microsoft/phi-2"
)
parser.add_argument(
    "o",
    "outtype",
    default="f32",
    choices=["f32", "f16", "q8_0"],
    type=str,
    help="Example: f32",
)
parser.add_argument(
    "d",
    "download_folder",
    default="./models_downloaded/",
    type=str,
    help="Best to just leave this as is \
unless you have a specific use case",
)
parser.add_argument(
    "c",
    "converted_models_folder",
    default="./converted_models/",
    type=str,
    help="Best to just leave this as is \
unless you have a specific use case",
)

# parse the arguments and log the input
args = parser.parse_args()

model_id = args.r
outtype = args.o
models_folder = args.d
converted_models_folder = args.c

model_path = f"{models_folder}{model_id}"
model_name = model_id.split("/")[1]
converted_models_output_folder = f"{converted_models_folder}{model_id}/"
outfile_name = f"{converted_models_output_folder}{model_name}.gguf"

# -create the conversion command using our arguments
bash_convert_to_gguf = f"""python3 llama.cpp/convert-hf-to-gguf.py {model_path}/ \
--outfile {outfile_name} \
--outtype {outtype}"""

# stupidly verbose logging for all the inputs
logging.info(
    f"Arguments: {args}",
    f"Model ID: {model_id}",
    f"Outtype: {outtype}",
    f"Download Folder: {models_folder}",
    f"Converted Models Folder: {converted_models_folder}",
    f"Model Path: {model_path}",
    f"Model Name: {model_name}",
    f"Converted Models Output Folder: {converted_models_output_folder}",
    f"Outfile Name: {outfile_name}",
    f"Bash Command: {bash_convert_to_gguf}",
)

# verify the folders exist, if not create them
if not os.path.exists(models_folder) or not os.path.exists(converted_models_folder):
    try:
        os.makedirs(models_folder)
        os.makedirs(converted_models_folder)
    except Exception as e:
        logging.info(f"Error creating root folders needed: {e}")
else:
    logging.info(
        f"Root folders {models_folder} and {converted_models_folder} already exist"
    )

# download the model from huggingface, if the model exists on the file system, do not download
if not os.path.exists(model_path):
    # using a try except block to capture if the model exists on huggingface
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=models_folder,
            local_dir_use_symlinks=False,
            revision="main",
        )
        logging.info(f"Model {model_id} downloaded successfully to {model_path}")
    # i really have mixed feelings about this try except block
    except:
        logging.info(
            f"Could not find model '{model_id}' on huggingface, please check the model id and try again"
        )
else:
    logging.info(
        f"Model {model_id} already exists on the file system, skipping download \
if you would like to or need to redownload the model, please delete the folder \
{model_path} and try again"
    )

logging.info("Starting Conversion...this may take a while..do not close the terminal window until you see confirmation, thanks!")

output = subprocess.check_output(["bash", "-c", bash_convert_to_gguf])

logging.info(output)
logging.info(f"Conversion completed...converted model saved to {outfile_name}")
