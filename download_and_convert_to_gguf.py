import argparse
import logging
import os
import subprocess
from huggingface_hub import snapshot_download
from typing import NoReturn
# setup logging

def setup_logging():
    """Setup the logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging

def setup_parser(parser=argparse.ArgumentParser()) -> argparse.ArgumentParser:
    """Setup the argument parser and add arguments
    ----------------
    Args:
        parser (argparse.ArgumentParser): the argument parser
        
    Returns:
        argparse.ArgumentParser: the argument parser
    """
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
    return parser

def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the arguments
    ----------------
    Args:
        parser (argparse.ArgumentParser): the argument parser
    Returns:
        argparse.Namespace: the parsed arguments
    """
    args = parser.parse_args()
    return args

def build_convert_to_gguf_command(
    downloaded_models_output_folder: str, outfile: str, outtype: str
) -> str:
    """Build the bash command to convert the model to gguf
    ----------------
    Example:
    python3 llama.cpp/convert-hf-to-gguf.py ./downloaded_models/microsoft/phi-2/ \
    --outfile ./converted_models/microsoft/phi-2/phi-2_f32.gguf --outtype f32
    ----------------
    Args:
        downloaded_models_output_folder (str): path to the downloaded model
        outfile (str): path to the output file
        outtype (str): the type of output file
        
    Returns:
        str: the bash command to convert the model to gguf
    """
    # needs to be left justified on new line because of the string formatting
    return f"""python3 llama.cpp/convert-hf-to-gguf.py {downloaded_models_output_folder} \
--outfile {outfile} --outtype {outtype}""" 

def output_verification(output_list: list) -> NoReturn:
    """Take a list of output folders and verify if they exist, if not create them
    ----------------
    Args:
        output_list (list): list of output folders to verify
        
    Returns:
        NoReturn: None
    """
    logging.info(
        f"Checking if the output folders '{output_list}' exist"
    )
    for output_directory in output_list:
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
            except Exception as e:
                logging.info(f"Error creating {output_directory}: {e}")
        else:
            logging.info(f"{output_directory} exists, continuing")

def download_full_model_from_huggingface(
        huggingface_repo: str,
        local_download_directory: str,
        use_symlinks: bool = False,
        revision: str = "main",
        ) -> NoReturn:
    """Download the model from huggingface
    ----------------
    Args:
        huggingface_repo (str): the huggingface model id
        local_download_directory (str): the local download directory
        use_symlinks (bool): whether to use symlinks or not
        revision (str): the revision of the model
    ----------------
    Returns:
        NoReturn: None
    """
    logging.info(
        f"Checking if model {huggingface_repo} exists on the file system in {local_download_directory}"
    )
    if not os.path.exists(local_download_directory):
        logging.info(
            f"Model {huggingface_repo} does not exist on the file system, downloading..."
        )
        # using a try except block to capture if the model exists on huggingface
        try:
            snapshot_download(
                repo_id=huggingface_repo,
                local_dir=local_download_directory,
                local_dir_use_symlinks=False,
                revision="main",
            )
            logging.info(
                f"Model {huggingface_repo} downloaded successfully to {local_download_directory}"
            )
        except Exception as e:
            logging.info(
                f"Could not find model '{huggingface_repo}' on huggingface, please check the model id and try again"
            )
            logging.info(f"Error: {e}")
            exit()
    else:
        logging.info(
            f"Model {huggingface_repo} already exists on the file system, skipping download \
    if you would like to or need to redownload the model, please delete the folder \
    {local_download_directory} and try again"
        )
    logging.info(
        f"Model {huggingface_repo} exists on the file system in {local_download_directory}"
    )

def convert_to_gguf(bash_command: str) -> NoReturn:
    """Convert the model to gguf
    ----------------
    Args:
        bash_command (str): the bash command to convert the model to gguf
    ----------------
    Returns:
        NoReturn: None
    """
    logging.info(
        f"Starting Conversion...this may take a while..do not close the terminal window until you see confirmation, thanks!"
    )
    output = subprocess.check_output(["bash", "-c", bash_command])
    logging.info(output)
    logging.info(f"Conversion completed...converted model saved to {outfile}")

logging = setup_logging()
parser = setup_parser()
args = parse_args(parser)

huggingface_repo_id, outtype, downloaded_models_folder, converted_models_folder = args.r, args.o, args.d, args.c

model_name = huggingface_repo_id.split("/")[1]  # phi-2
downloaded_models_output_folder = f"{downloaded_models_folder}{huggingface_repo_id}/"  # ./downloaded_models/microsoft/phi-2/
converted_models_output_folder = f"{converted_models_folder}{huggingface_repo_id}/"  # ./converted_models/microsoft/phi-2/
outfile = f"{converted_models_output_folder}{model_name}_{outtype}.gguf"  # ./converted_models/microsoft/phi-2/phi-2_f32.gguf
conversion_bash_command=build_convert_to_gguf_command(downloaded_models_output_folder, outfile, outtype)

# stupidly verbose logging for all the inputs, i dont like this formatting also.
logging.info(
    "Arguments: {}, Model ID: {}, Outtype: {}, Download Folder: {}, Converted Models Folder: {}, Model Path: {}, Model Name: {}, Converted Models Output Folder: {}, Outfile Name: {}, Conversion Command Build: {}".format(
        args,
        huggingface_repo_id,
        outtype,
        downloaded_models_folder,
        converted_models_folder,
        downloaded_models_output_folder,
        model_name,
        converted_models_output_folder,
        outfile,
        conversion_bash_command,
        
    )
)

output_verification([downloaded_models_folder, converted_models_folder, converted_models_output_folder])
download_full_model_from_huggingface(huggingface_repo_id, downloaded_models_output_folder)
convert_to_gguf(conversion_bash_command)
