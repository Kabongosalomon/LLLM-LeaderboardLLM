#imports
import pandas as pd
import os, ipdb, re
import random, evaluate
import string
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datasets import DatasetDict, Dataset, load_dataset
import wandb
import ast
import re, os
import subprocess
import argparse
import logging

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)  # Set the logging level

# Create a file handler that logs even debug messages
fh = logging.FileHandler('dataset_creation_logs.log')
fh.setLevel(logging.ERROR)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(fh)

def potential_tdms_context(latex_file, output_folder):
    
    if len(latex_file.rsplit("/",1)) != 2:
        return 
    
    base_source, filename = latex_file.rsplit("/",1)

    if len(filename.rsplit(".",1)) != 2:
        return 
    
    file_id, file_ext = filename.rsplit(".",1)
    
    
    # Read the input LaTeX file
    with open(latex_file, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    pattern = re.compile(
        r'''
        (                             # Start capturing group
            \\section                 # Match \section
            \*?                       # Match optional *
            \s*                       # Match optional whitespace
            \{                        # Match {
            (?!                       # Start negative lookahead
                Result(s?)               # Negative lookahead for Results
                |                     # Or
                Experimentation(s?)       # Negative lookahead for Experimentation
                |
                Experiment(s?)
                |
                Conclusion
            )                         # End negative lookahead
            [^}]*                     # Match any characters except }
            \}                        # Match }
            .*?                       # Match any characters (non-greedy)
            (?=\\section|\\end\{document\}|\\bibliography|\Z)          # Positive lookahead for next \section, \end{document}, \bibliography or end of string
        )                             # End capturing group
        ''',
        re.DOTALL | re.IGNORECASE | re.VERBOSE
    )

    # Remove the matched content
    content_new = re.sub(pattern, '', content)

    if not os.path.exists(f"{output_folder}"):
        os.makedirs(f"{output_folder}")
        
    # if os.path.exists(f"{base_source}/edits/{file_id}_edit.{file_ext}"):
    #     os.remove(f"{base_source}/edits/{file_id}_edit.{file_ext}")
            
    # Write the modified content back to the file
    with open(f"{output_folder}/{file_id}_summarised.{file_ext}", 'w', encoding='utf-8', errors='ignore') as file:
        file.write(content_new)
             
        
def pandoc_latex_to_text(latex_file, pendoc_template, output_folder):
    # Output a plain text file given a valid .tex file. 
    
    file_ID = latex_file.rsplit("/")[-1].rsplit(".", 1)[0]
    
    if not os.path.exists(f"{output_folder}"):
        os.makedirs(f"{output_folder}")
    
    # logger.warning(f"Processing {file_ID}")
    # Construct the command
    command = [
        "pandoc",
        "--to=plain",
        f"--template={pendoc_template}",
        "--wrap=none",
        f"{latex_file}",
        "-o",
        f"{output_folder}/{file_ID}.txt",
        "--quiet"
    ]
    
    try:
        # print(f"Processing file : {file_ID}")
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True, timeout=120)
        result.check_returncode()  # This will raise CalledProcessError if the command failed
        
        # Delete the generated text file if it's empty 
        if os.path.exists(f"{output_folder}/{file_ID}.txt"):
            
            with open(f"{output_folder}/{file_ID}.txt", 'r') as file:
                file_content = file.read()
                
            if len(file_content.split("\n\n")) < 3:
                os.remove(f"{output_folder}/{file_ID}.txt")
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr
        # print(f"File {latex_file} failed with an error: {error_message}")
        logger.error(f"File {latex_file} failed with an error: {error_message}")
    except subprocess.TimeoutExpired as e:
        # Handle the timeout case
        logger.error(f"File {latex_file} processing timed out after 2 minutes")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Data Creation Script")
    parser.add_argument("-stex", "--source_folder", default="/nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_tex", help="Source latex file")
    parser.add_argument("-ptemplate", "--pendoc_template", default="/nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/template.plain", help="Pendoc Template")
    # parser.add_argument("-out_name", "--output_name", default="arxiv_tex_summarised", help="Output name")
    parser.add_argument("-output", "--output_folder", default="/nfs/home/kabenamualus/Research/LLLM-LeaderboardLLM/data_proccess/arxiv_tex_summarised", help="Output folder")

    args = parser.parse_args()

    source_folder = args.source_folder
    output_folder = args.output_folder
    # output_name = args.output_name
    pendoc_template = args.pendoc_template

    for file_id in tqdm(os.listdir(f"{source_folder}")):
        latex_file = f"{source_folder}/{file_id}"
        # potential_tdms_context(latex_file, output_folder=f"{output_folder}/{output_name}")
        potential_tdms_context(latex_file, output_folder=f"{output_folder}")
        
    # for latex_file in tqdm(os.listdir(f"{output_folder}/{output_name}")):
    #     pandoc_latex_to_text(f"{output_folder}/{output_name}/{latex_file}", 
    #         pendoc_template=pendoc_template,
    #         output_folder=f"{output_folder}/{output_name.replace('tex', 'txt')}")
    
    for latex_file in tqdm(os.listdir(f"{output_folder}")):
        pandoc_latex_to_text(f"{output_folder}/{latex_file}", 
            pendoc_template=pendoc_template,
            output_folder=f"{output_folder.replace('tex', 'txt')}")
    