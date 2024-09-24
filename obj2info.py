import glob
from pathlib import Path
import pymeshlab
import numpy as np

# file names as list of str.
files = [f for f in glob.glob('*_1024.obj')]

for i, filename in enumerate(files):
	# load obj
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(filename)
	m = ms.current_mesh()
	bb = m.bounding_box()
	v = m.vertex_matrix()
	print(i, filename, 'box=', bb.dim_x(), bb.dim_y(), bb.dim_z(), 'center=', bb.center(), 'diagonal=', bb.diagonal())
	


