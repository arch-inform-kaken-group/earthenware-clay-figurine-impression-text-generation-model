import glob
from pathlib import Path
import pymeshlab
import open3d as o3d
import numpy as np

# file names to convert as list of str.
files = [f for f in glob.glob('*_1024.obj')]

# Firstly, make the bounding box from all the inpuit files
print('extracting min_bound and max_bound of all the files...')
min_bound = np.array([float('inf'), float('inf'), float('inf')])
max_bound = np.array([-float('inf'), -float('inf'), -float('inf')])
for i, filename in enumerate(files):
	# load obj
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(filename)
	m = ms.current_mesh()
	bb = m.bounding_box()
	v = m.vertex_matrix()
	# create open3d PointCloud
	p = o3d.geometry.PointCloud()
	p.points = o3d.utility.Vector3dVector(v)
	min_b = p.get_min_bound()
	max_b = p.get_max_bound()
	min_bound = np.minimum(min_bound, min_b)
	max_bound = np.maximum(max_bound, max_b)
	print(i, filename, min_b, max_b)

print('result: min_bound=', min_bound, 'max_bound=', max_bound)

# visualize
vis = o3d.visualization.Visualizer()
vis.create_window(width = 350, height = 400)
#vis.create_window()
ctr = vis.get_view_control()
#ctr.set_lookat((min_bound + max_bound) / 2)
#ctr.set_front([0, 0, 1])
#ctr.set_up([0, -1, 0])
#ctr.set_zoom(0.1)

print('generating png files...')
for i, filename in enumerate(files):
	# output file name
	outputfilename = filename[:-4] + '.png'
	#print(outputfilename)
	
	# load obj
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(filename)
	m = ms.current_mesh()
	bb = m.bounding_box()
	v = m.vertex_matrix()
	#print(i, len(v))
	
	# create open3d PointCloud
	p = o3d.geometry.PointCloud()
	p.points = o3d.utility.Vector3dVector(v)

	# add geometry
	if i == 0:
		vis.add_geometry(p, reset_bounding_box = True)
		ctr.set_lookat((min_bound + max_bound) / 2)
		ctr.set_front([0, 0, 1])
		ctr.set_up([0, 1, 0])
		ctr.set_zoom(1)
	else:
		vis.add_geometry(p, reset_bounding_box = False)
	
	#vis.run()
	vis.poll_events()
	vis.update_renderer()

	# capture and output file
	img = np.asarray(vis.capture_screen_float_buffer(do_render = True))
	o3d.io.write_image(outputfilename, o3d.geometry.Image((img * 255).astype(np.uint8)))
	
	#remove geometry
	vis.remove_geometry(p, reset_bounding_box = False)
	
	print(i, filename, outputfilename)

#vis.run()
vis.destroy_window()
	


