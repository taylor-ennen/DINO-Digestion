import subprocess
import logging
import os
from huggingface_hub import snapshot_download

# setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

# alias the folders used for the base models and converted models
models_folder = "./models_downloaded/"
converted_models_folder = "./converted_models/"

# make sure the folder exists
if not os.path.exists(models_folder) or not os.path.exists(converted_models_folder):
    try:
        os.makedirs(models_folder)
        os.makedirs(converted_models_folder)
    except Exception as e:
        logging.info(f"Error creating root folders needed: {e}")
else:
    logging.info(f"Root folders {models_folder} and {converted_models_folder} already exist")


# capture the model id from huggingface
model_id = "microsoft/phi-2"
model_path = f"{models_folder}{model_id}"
model_name = model_id.split("/")[1]

logging.info(model_id)
logging.info(model_name)
logging.info(model_path)

# download the model from huggingface, if the model exists on the file system, do not download
if not os.path.exists(model_path):
    # using a try except block to capture if the model exists on huggingface
    try:
        snapshot_download(
            repo_id=str(model_id),
            local_dir=f"{model_path}",
            local_dir_use_symlinks=False,
            revision="main",
        )
        logging.info(f"Model {model_id} downloaded successfully to {model_path}")
    # 
    except:
        logging.info(
            f"Could not find model '{model_id}' on huggingface, please check the model id and try again"
        )
else:
    logging.info(
        f"Model {model_id} already exists on the file system, skipping download \
if you would like to or need to redownload the model, please delete the folder \
{model_path} and try again")

# once the model exists, continue with the conversion using the model id as the folder name
converted_models_output_folder = f"{converted_models_folder}{model_id}/"
outfile_name=f"{converted_models_output_folder}{model_name}.gguf"
outtype="f32"

bash_convert_to_gguf = f"""python3 llama.cpp/convert-hf-to-gguf.py {model_path}/ \
--outfile {outfile_name} \
--outtype {outtype}"""

logging.info(bash_convert_to_gguf)
logging.info("Starting Conversion...")

output = subprocess.check_output(["bash", "-c", bash_convert_to_gguf])

logging.info(output)
logging.info(f"Conversion completed...converted model saved to {outfile_name}")
