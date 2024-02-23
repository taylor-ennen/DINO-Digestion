# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:09:04 2024

@author: mysticmarks
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from threading import Thread
from huggingface_hub import snapshot_download
import subprocess
import os

# Logging message function
def log_message(message):
    log_area.configure(state='normal')
    log_area.insert(tk.END, message + "\n")
    log_area.configure(state='disabled')
    log_area.see(tk.END)

# Browse folder function
def browse_folder(entry):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry.delete(0, tk.END)
        entry.insert(0, folder_selected)

# Download model function
def download_model(model_id, downloaded_models_folder):
    model_path = f"{downloaded_models_folder}/{model_id}/"
    if not os.path.exists(model_path):
        try:
            snapshot_download(repo_id=model_id, cache_dir=model_path)
            log_message(f"Model {model_id} downloaded successfully to {model_path}")
        except Exception as e:
            log_message(f"Could not download model '{model_id}': {e}")
            messagebox.showerror("Error", f"Could not download model '{model_id}'.")
    else:
        log_message(f"Model {model_id} already exists. Skipping download.")

# Convert model function
def convert_model(model_id, downloaded_models_folder, converted_models_folder, outtype):
    model_name = model_id.split("/")[-1]
    model_path = f"{downloaded_models_folder}/{model_id}/"
    converted_models_output_folder = f"{converted_models_folder}/{model_id}/"
    outfile = f"{converted_models_output_folder}{model_name}_{outtype}.gguf"
    
    if not os.path.exists(converted_models_output_folder):
        os.makedirs(converted_models_output_folder)

    bash_convert_to_gguf = f"python3 llama.cpp/convert-hf-to-gguf.py {model_path} --outfile {outfile} --outtype {outtype}"
    
    try:
        subprocess.check_output(["bash", "-c", bash_convert_to_gguf], text=True)
        log_message(f"Conversion completed...converted model saved to {outfile}")
        messagebox.showinfo("Success", f"Model converted successfully and saved to {outfile}")
    except subprocess.CalledProcessError as e:
        log_message(f"Conversion failed: {e.output}")
        messagebox.showerror("Error", "Conversion failed. See log for details.")

# Trigger download
def on_download():
    model_id = model_id_entry.get()
    downloaded_models_folder = download_folder_entry.get()
    Thread(target=lambda: download_model(model_id, downloaded_models_folder)).start()

# Trigger conversion
def on_convert():
    model_id = model_id_entry.get()
    downloaded_models_folder = download_folder_entry.get()
    converted_models_folder = converted_folder_entry.get()
    outtype = outtype_combobox.get()
    Thread(target=lambda: convert_model(model_id, downloaded_models_folder, converted_models_folder, outtype)).start()

# Setup the main application window
app = tk.Tk()
app.title("Model Management Tool")
app.geometry("800x600")

tk.Label(app, text="Model ID:").pack()
model_id_entry = tk.Entry(app)
model_id_entry.pack()

tk.Label(app, text="Output Type:").pack()
outtype_combobox = ttk.Combobox(app, values=["f32", "f16", "q8_0"], state="readonly")
outtype_combobox.pack()

tk.Label(app, text="Download Folder:").pack()
download_folder_entry = tk.Entry(app)
download_folder_entry.pack()
tk.Button(app, text="Browse", command=lambda: browse_folder(download_folder_entry)).pack()

tk.Label(app, text="Converted Models Folder:").pack()
converted_folder_entry = tk.Entry(app)
converted_folder_entry.pack()
tk.Button(app, text="Browse", command=lambda: browse_folder(converted_folder_entry)).pack()

tk.Button(app, text="Download Model", command=on_download).pack()
tk.Button(app, text="Convert Model", command=on_convert).pack()

log_area = scrolledtext.ScrolledText(app, state='disabled', height=10)
log_area.pack(pady=10)

app.mainloop()
