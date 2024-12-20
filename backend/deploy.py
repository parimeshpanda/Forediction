from os import path, makedirs
from distutils.dir_util import copy_tree, remove_tree
from distutils.file_util import copy_file
 
base_path = path.abspath(path.dirname(__name__))
deployment_folder = f"{base_path}/deployments/"
if not path.exists(deployment_folder):
    makedirs(deployment_folder)
 
# Copying deployment resources to deployment folder
deployment_resources = ["app", "config.py", "requirements.txt", "constants.py", "main.py"]
for resource in deployment_resources:
    src = f"{base_path}/{resource}"
    dest = f"{deployment_folder}/{resource}"
    if path.isdir(src): copy_tree(src, dest)
    else: copy_file(src, dest)