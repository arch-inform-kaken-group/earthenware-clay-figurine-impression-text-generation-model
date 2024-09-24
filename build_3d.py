import glob
from pathlib import Path
import pymeshlab
import numpy as np


# Resampling *.obj with PyMeshLab to a specified number + alpha and then reduce to the specified number

# resampling number
resample = 1024
alpha = 8

# file names as list of str. Remove such as *_1024.obj
files = [f for f in glob.glob('*.obj') if not f.endswith('_' + str(resample) + '.obj')]

for i, filename in enumerate(files):
	# output file name
	newfilename = filename[:-4] + '_' + str(resample) + '.obj'
	# load obj
	data = pymeshlab.MeshSet()
	data.load_new_mesh(filename)
	m = data.current_mesh()
	num_vertices = m.vertex_number()
	num_edges = m.edge_number()
	num_faces = m.face_number()
	#pymeshlab.print_filter_list()

	# resampling
	data.generate_sampling_poisson_disk(samplenum = resample + alpha, exactnumflag = True, exactnumtolerance = 0)

	#data.print_status()
	data.set_current_mesh(1)
	m = data.current_mesh()
	num_vertices2 = m.vertex_number()
	v = m.vertex_matrix()
	try:
		# reduce into the resampling number
		selection = np.random.choice(v.shape[0], resample, replace = False)
		#print(v.shape, len(selection), selection)
		new_v = v[selection, :]
		# centroid fit to (0, 0, 0)
		cntr = np.mean(new_v, axis = 0)
		new_v2 = new_v - cntr
		cntr2 = np.mean(new_v2, axis = 0)
		#print('[', i, ']', filename, num_vertices, num_edges, num_faces, 'sampled2', num_vertices2, newfilename, 'center', cntr, 'new_v', new_v.shape, new_v, 'center2', cntr2, 'new_v2', new_v2)
		
		# create data to save
		new_mesh = pymeshlab.Mesh(vertex_matrix = new_v2)
		new_data = pymeshlab.MeshSet()
		new_data.add_mesh(new_mesh)
		num_vertices3 = new_data.current_mesh().vertex_number()
		new_data.save_current_mesh(newfilename)
		print('[', i, ']', filename, num_vertices, num_edges, num_faces, 'sampled2', num_vertices2, newfilename, num_vertices3)
		#break
	except Exception as e:
		print('ERROR: sampling failed... [', i, ']', filename, num_vertices, num_edges, num_faces, 'sampled2', num_vertices2)


