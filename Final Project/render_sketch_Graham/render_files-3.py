import numpy as np
import open3d as o3d
import os
import sys
import glob
import cv2
from PIL import Image
from PIL.ImageOps import invert
import argparse
import time
import multiprocessing
from pathlib import Path

# Global variables
OUTPUT_DIR = 'renders'
WIDTH = 800
HEIGHT = 800
NUMBER_OF_CAMERA_ANGLES = 6
TARGET_SIZE = 8.0


def analyze_and_orient_mesh(mesh):
	"""Analyze mesh to detect its orientation and apply corrections."""
	# Compute principal axes using PCA
	points = np.asarray(mesh.vertices)
	mean = np.mean(points, axis=0)
	points_centered = points - mean

	# Get covariance matrix
	cov = np.cov(points_centered.T)

	# Get eigenvectors and eigenvalues
	eigvals, eigvecs = np.linalg.eigh(cov)

	# Sort eigenvalues in descending order
	idx = eigvals.argsort()[::-1]
	eigvals = eigvals[idx]
	eigvecs = eigvecs[:, idx]

	# The eigenvectors now represent the principal axes
	# Usually, the smallest eigenvector is the "up" direction for most models
	# However, this may vary by model type

	# We'll return the mesh with its original orientation and the principal axes
	# so the user can apply transformations based on this information
	return mesh, eigvecs


def apply_orientation_correction(mesh, x_rot=0, y_rot=0, z_rot=0, auto_orient=False):
	"""Apply orientation corrections to the mesh."""
	# Original center
	center = mesh.get_axis_aligned_bounding_box().get_center()

	# Start by auto-orienting if requested
	if auto_orient:
		# Get principal axes
		mesh, axes = analyze_and_orient_mesh(mesh)

		# Try to orient the model so that the "up" axis is along the Y axis
		# This is a heuristic that works for many models
		# Assume the 3rd principal component is the "up" direction (smallest variation)
		up_axis = axes[:, 2]

		# Calculate rotation to align the up_axis with [0, 1, 0] (y-axis up)
		y_axis = np.array([0, 1, 0])

		# Get rotation axis and angle
		rotation_axis = np.cross(up_axis, y_axis)

		if np.linalg.norm(rotation_axis) > 1e-5:  # Check if not parallel
			rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
			dot_product = np.dot(up_axis, y_axis)
			angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

			# Convert to rotation matrix
			rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

			# Apply rotation
			mesh = mesh.rotate(rotation, center=center)

		# Log info about the orientation
		print(f"Auto-oriented model. Principal axes: {axes}")

	# Apply manual rotations if specified
	if x_rot != 0:
		R_x = mesh.get_rotation_matrix_from_xyz((np.radians(x_rot), 0, 0))
		mesh = mesh.rotate(R_x, center=center)

	if y_rot != 0:
		R_y = mesh.get_rotation_matrix_from_xyz((0, np.radians(y_rot), 0))
		mesh = mesh.rotate(R_y, center=center)

	if z_rot != 0:
		R_z = mesh.get_rotation_matrix_from_xyz((0, 0, np.radians(z_rot)))
		mesh = mesh.rotate(R_z, center=center)

	return mesh


def process_obj_file(file_path, model_name, target_faces=100000, x_rot=0, y_rot=0, z_rot=0, auto_orient=False):
	"""Load and prepare an OBJ file for rendering with normalized scaling."""
	try:
		# Load the mesh using Open3D
		mesh = o3d.io.read_triangle_mesh(file_path)

		if not mesh.has_vertices():
			print(f"Warning: No vertices found in {model_name}")
			return None

		# Print original mesh details
		original_vertices = len(mesh.vertices)
		original_triangles = len(mesh.triangles)
		print(f"Loaded {model_name} with {original_vertices} vertices and "
			  f"{original_triangles} faces")

		# Apply orientation corrections
		mesh = apply_orientation_correction(mesh, x_rot, y_rot, z_rot, auto_orient)

		# Compute normals for proper rendering (do this after simplification)
		mesh.compute_vertex_normals()

		# Get bounding box
		bbox = mesh.get_axis_aligned_bounding_box()
		bbox_size = bbox.get_extent()

		# Calculate volume (or proxy for visual footprint)
		volume = (bbox_size[0] * bbox_size[1] * bbox_size[2]) ** (1 / 3)

		# Calculate scale factor based on volume
		scale_factor = TARGET_SIZE / volume

		# Create a scaled mesh
		mesh_scaled = mesh.scale(scale_factor, center=bbox.get_center())

		# Center the mesh at origin
		mesh_center = mesh_scaled.get_axis_aligned_bounding_box().get_center()
		mesh_centered = mesh_scaled.translate(-mesh_center)

		print(f"Model dimensions: {bbox_size}, Scale factor: {scale_factor:.4f}")

		return mesh_centered

	except Exception as e:
		print(f"Error loading {model_name}: {str(e)}")
		return None


def setup_render_environment():
	"""Create a pre-configured visualization environment."""
	vis = o3d.visualization.Visualizer()
	vis.create_window(width=WIDTH, height=HEIGHT, visible=False)

	# Setup rendering options
	render_option = vis.get_render_option()
	render_option.background_color = np.array([0, 0, 0])  # Black background
	render_option.point_size = 1.0
	render_option.light_on = True

	# Enable smooth shading
	render_option.mesh_show_back_face = False

	return vis


def render_mesh_from_angle(mesh, angle, vis=None):
	"""Render the mesh from a specific angle."""
	# Create visualizer if not provided
	if vis is None:
		vis = setup_render_environment()
		need_to_close = True
	else:
		need_to_close = False

	# Clear any existing geometries and add the mesh
	vis.clear_geometries()
	vis.add_geometry(mesh)

	# Set up camera view
	view_control = vis.get_view_control()
	view_control.set_zoom(0.8)
	view_control.set_front([0, 0, -1])
	view_control.set_up([0, 1, 0])
	view_control.set_lookat([0, 0, 0])

	# Rotate camera - convert to degrees if needed
	view_control.rotate(angle * 10, 0)  # Multiply by 10 as Open3D rotation is in smaller units

	# Update the rendering
	vis.update_geometry(mesh)
	vis.poll_events()
	vis.update_renderer()

	# Capture image
	image = vis.capture_screen_float_buffer(do_render=True)

	# Convert to numpy array
	img_np = np.asarray(image)

	# Close visualizer if we created it
	if need_to_close:
		vis.destroy_window()

	return img_np


def convert_to_line_drawing(image):
	"""Convert a rendered image to a line drawing using Canny edge detection."""
	# Convert to grayscale if it's RGB
	if len(image.shape) == 3:
		# Convert from float [0,1] to uint8 [0,255]
		img_uint8 = (image * 255).astype(np.uint8)
		gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
	else:
		gray = (image * 255).astype(np.uint8)

	# Apply Canny edge detection
	edges = cv2.Canny(gray, 50, 150)

	# Invert the image (white edges on black background)
	edges_inverted = 255 - edges

	return edges_inverted


def process_angle(mesh, angle, idx, model_dir, model_name):
	"""Process a single angle (for multiprocessing)."""
	try:
		start_time = time.time()

		# Each process needs its own visualizer
		vis = setup_render_environment()

		# Render the image
		rendered_image = render_mesh_from_angle(mesh, angle, vis)

		# Convert to line drawing
		line_drawing = convert_to_line_drawing(rendered_image)

		# Save the image
		filename = os.path.join(model_dir, f"{model_name}_angle_{idx + 1}.png")
		Image.fromarray(line_drawing).save(filename)

		# Clean up
		vis.destroy_window()

		return idx, time.time() - start_time
	except Exception as e:
		print(f"Error rendering angle {idx}: {e}")
		return idx, -1


def generate_orientation_preview(mesh, output_dir, model_name):
	"""Generate a preview of the model from different angles to check orientation."""
	preview_dir = os.path.join(output_dir, f"{model_name}_preview")
	os.makedirs(preview_dir, exist_ok=True)

	# Generate views from 6 standard directions
	views = [
		("front", 0, 0),  # Front
		("back", 0, 180),  # Back
		("left", 0, 90),  # Left
		("right", 0, 270),  # Right
		("top", 90, 0),  # Top
		("bottom", -90, 0)  # Bottom
	]

	vis = setup_render_environment()

	print("Generating orientation preview...")
	for name, elevation, azimuth in views:
		# Setup camera
		vis.clear_geometries()
		vis.add_geometry(mesh)

		view_control = vis.get_view_control()
		view_control.set_zoom(0.8)
		view_control.set_front([0, 0, -1])
		view_control.set_up([0, 1, 0])
		view_control.set_lookat([0, 0, 0])

		# Apply camera rotation for this view
		view_control.rotate(azimuth * 10, elevation * 10)  # Open3D rotation units

		# Update rendering
		vis.update_geometry(mesh)
		vis.poll_events()
		vis.update_renderer()

		# Capture and save image
		image = vis.capture_screen_float_buffer(do_render=True)
		img_np = (np.asarray(image) * 255).astype(np.uint8)

		filename = os.path.join(preview_dir, f"{model_name}_{name}.png")
		Image.fromarray(img_np).save(filename)

		print(f"  Saved {name} view to {filename}")

	vis.destroy_window()
	print(f"Preview images saved to {preview_dir}")
	print("Check these images to determine if orientation corrections are needed")


def process_single_obj_file(obj_path, num_angles, output_dir=OUTPUT_DIR,
							use_multiprocessing=True, target_faces=100000,
							x_rot=0, y_rot=0, z_rot=0, auto_orient=False,
							preview=False):
	"""Process a single OBJ file, rendering it from multiple angles."""
	model_name = Path(obj_path).stem
	print(f"\nProcessing model: {model_name}")

	# Create model output directory
	model_dir = os.path.join(output_dir, model_name)
	os.makedirs(model_dir, exist_ok=True)

	# Load and process the OBJ file
	start_time = time.time()
	mesh = process_obj_file(obj_path, model_name, target_faces, x_rot, y_rot, z_rot, auto_orient)
	if mesh is None:
		print(f"Skipping model {model_name} due to loading error")
		return False

	# Generate orientation preview if requested
	if preview:
		generate_orientation_preview(mesh, output_dir, model_name)
		print("Preview generated. Re-run the script with orientation corrections if needed.")
		return True

	# Calculate evenly distributed camera angles in degrees
	camera_angles = [i * (360 / num_angles) for i in range(num_angles)]

	# Render from each angle
	render_start = time.time()

	if use_multiprocessing and num_angles > 1 and multiprocessing.cpu_count() > 1:
		print(f"Using {min(multiprocessing.cpu_count(), num_angles)} processes for rendering")

		# Use multiprocessing.Pool for parallel rendering
		with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), num_angles)) as pool:
			tasks = [(mesh, angle, i, model_dir, model_name) for i, angle in enumerate(camera_angles)]
			results = pool.starmap(process_angle, tasks)

			# Process results
			for idx, duration in results:
				if duration > 0:
					print(f"  Angle {idx + 1}: {duration:.2f}s")
				else:
					print(f"  Angle {idx + 1}: Failed")
	else:
		# Process angles sequentially
		print("Processing angles sequentially")
		vis = setup_render_environment()

		for i, angle in enumerate(camera_angles):
			angle_start = time.time()

			# Render the image
			rendered_image = render_mesh_from_angle(mesh, angle, vis)

			# Convert to line drawing
			line_drawing = convert_to_line_drawing(rendered_image)

			# Save the image
			filename = os.path.join(model_dir, f"{model_name}_angle_{i + 1}.png")
			Image.fromarray(line_drawing).save(filename)

			print(f"  Angle {i + 1}: {time.time() - angle_start:.2f}s")

		# Clean up
		vis.destroy_window()

	print(f"Completed rendering {model_name} from {num_angles} angles")
	print(f"Load time: {render_start - start_time:.2f}s, Total render time: {time.time() - render_start:.2f}s")

	return True


def process_directory(directory_path, num_angles, output_dir=OUTPUT_DIR,
					  use_multiprocessing=True, target_faces=100000,
					  x_rot=0, y_rot=0, z_rot=0, auto_orient=False,
					  preview=False):
	"""Process all OBJ files in a directory."""
	# Find all OBJ files in the directory
	obj_files = glob.glob(os.path.join(directory_path, "*.obj"))

	if not obj_files:
		print(f"No OBJ files found in directory: {directory_path}")
		return False

	print(f"Found {len(obj_files)} OBJ files to process")

	# Make sure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	# Process each OBJ file
	for i, obj_file in enumerate(obj_files):
		print(f"\nProcessing file {i + 1}/{len(obj_files)}: {obj_file}")
		process_single_obj_file(obj_file, num_angles, output_dir,
								use_multiprocessing, target_faces,
								x_rot, y_rot, z_rot, auto_orient, preview)

	print(f"\nBatch processing complete. Processed {len(obj_files)} files.")
	return True


def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Render OBJ files from multiple angles using Open3D")
	parser.add_argument("-a", "--angles", type=int, default=NUMBER_OF_CAMERA_ANGLES,
						help="Number of angles to render")
	parser.add_argument("-p", "--path", type=str, required=True,
						help="Path to OBJ file or directory containing OBJ files")
	parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
						help="Output directory for renders")
	parser.add_argument("-s", "--size", type=float, default=4.0,
						help="Target size factor for models (controls how much screen space they occupy)")
	parser.add_argument("-m", "--multiprocessing", action="store_true",
						help="Enable multiprocessing for faster rendering")
	parser.add_argument("-f", "--target-faces", type=int, default=100000,
						help="Target number of faces for mesh simplification (0 to disable)")

	# New orientation options
	parser.add_argument("--x-rot", type=float, default=0,
						help="Manual rotation around X axis in degrees")
	parser.add_argument("--y-rot", type=float, default=0,
						help="Manual rotation around Y axis in degrees")
	parser.add_argument("--z-rot", type=float, default=0,
						help="Manual rotation around Z axis in degrees")
	parser.add_argument("--auto-orient", action="store_true",
						help="Attempt to automatically orient the model")
	parser.add_argument("--preview", action="store_true",
						help="Generate orientation preview images")

	args = parser.parse_args()

	# Update global target size if specified
	global TARGET_SIZE
	TARGET_SIZE = args.size

	# Process the input path
	if os.path.isdir(args.path):
		# Process all OBJ files in the directory
		process_directory(args.path, args.angles, args.output,
						  args.multiprocessing, args.target_faces,
						  args.x_rot, args.y_rot, args.z_rot, args.auto_orient,
						  args.preview)
	elif os.path.isfile(args.path) and args.path.endswith('.obj'):
		# Process a single OBJ file
		process_single_obj_file(args.path, args.angles, args.output,
								args.multiprocessing, args.target_faces,
								args.x_rot, args.y_rot, args.z_rot, args.auto_orient,
								args.preview)
	else:
		print(f"Error: {args.path} is not a valid OBJ file or directory")

	print("Rendering complete")


if __name__ == "__main__":
	main()
 
def convert_to_sketch_images(path, angles=6, output='renders'):
    """
    Convert a 3D model to multiple sketch views from different angles.
    
    Args:
        path (str): Path to the input OBJ file
        angles (int): Number of angles to render (default: 6)
        output (str): Output directory path for the rendered sketches
    """
    # Process the single OBJ file with the specified parameters
    success = process_single_obj_file(
        obj_path=path,
        num_angles=angles,
        output_dir=output,
        use_multiprocessing=True,
        target_faces=100000,
        x_rot=0,
        y_rot=0,
        z_rot=0,
        auto_orient=False,
        preview=False
    )
    
    return success

if __name__ == "__main__":
    # Example usage with absolute paths
    input_path = r'D:\ahYen Workspace\ahYen Work\CMU_academic\MSCD_Y1_2425\11785-Intro to DL\public\Final Project\render_sketch_Graham\3dmodel\test_1_1743478640.obj'
    output_path = r'D:\ahYen Workspace\ahYen Work\CMU_academic\MSCD_Y1_2425\11785-Intro to DL\public\Final Project\render_sketch_Graham\views\test_1_1743478640'
    
    # Convert 3D model to sketch images
    convert_to_sketch_images(path=input_path, angles=5, output=output_path)
