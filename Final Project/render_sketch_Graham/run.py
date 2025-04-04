import os
import subprocess

# Define the paths
script_path = r"d:\ahYen Workspace\ahYen Work\CMU_academic\MSCD_Y1_2425\11785-Intro to DL\public\Final Project\render_sketch_Graham\render_files-3.py"
obj_path = r"D:\ahYen Workspace\ahYen Work\CMU_academic\MSCD_Y1_2425\11785-Intro to DL\public\Final Project\render_sketch_Graham\3dmodel\test_1_1743478640.obj"
output_path = r"D:\ahYen Workspace\ahYen Work\CMU_academic\MSCD_Y1_2425\11785-Intro to DL\public\Final Project\render_sketch_Graham\views"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Build the command
command = [
    "python", 
    script_path, 
    "-p", obj_path, 
    "-o", output_path, 
    "-a", "5"
]

# Print what we're about to do
print(f"Running render script with:")
print(f"  OBJ file: {obj_path}")
print(f"  Output: {output_path}")
print(f"  Angles: 5")

# Execute the command
try:
    result = subprocess.run(command, check=True)
    print("Rendering completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error running the render script: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")