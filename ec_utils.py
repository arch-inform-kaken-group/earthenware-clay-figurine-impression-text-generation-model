###########################################################################
# version 1.1
# updated 2024/6/28
###########################################################################
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###########################################################################
# Tokens Dataset
###########################################################################
class TokensDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.tknsdata = []

  def __len__(self):
    return len(self.tknsdata)

  def __getitem__(self, idx):
    return self.tknsdata[idx]

  def append(self, tkns):
    self.tknsdata.append(tkns)

  def clear(self):
    self.tknsdata.clear()


###########################################################################
# A Transformer decorder 1
###########################################################################
class TfDecoderMini1(nn.Module):
	def __init__(self, num_vocab, embeddim, learned_contextlen):
		super().__init__()
		# number of tokens in the vocabulary
		self.num_vocab = num_vocab
		# embedding dimension
		self.embdim = embeddim
		# context length during learning
		# This is only for keeping what context length was used during training of this model.
		self.learned_contextlen = learned_contextlen
		# QKV attention weights
		self.wq0 = nn.Parameter(torch.randn(self.embdim, self.num_vocab, requires_grad = True))
		self.wk0 = nn.Parameter(torch.randn(self.embdim, self.num_vocab, requires_grad = True))
		self.wv0 = nn.Parameter(torch.randn(self.embdim, self.num_vocab, requires_grad = True))
		# output token: logit for tokens
		self.fc1 = nn.Linear(self.embdim, self.embdim)
		self.fcOut = nn.Linear(self.embdim, self.num_vocab)
	
	# supposing [batch][self.num_vocab][sequence] tensor for x
	# Q, K, V, VA: [batch, emb, sequence]
	# A = Softmax(K^t Q / sqrt(d_emb))
	# c = 1 / sqrt(d_emb)
	# A: [batch, sequence, sequence]
	# This does not use self.learned_contextlen. Sequence length can be any in the forward().
	def forward(self, x):
		#print('dbg0 x=', x.size())
		# embedding dimension
		dimemb = len(self.wq0)
		dtyp = self.wq0.dtype

		# The 1st layer: Word embedding with QKV self attention
		c = 1 / torch.sqrt(torch.tensor(dimemb, dtype = dtyp))
		# Q
		q_cktq_a = torch.matmul(self.wq0, x)
		# K
		#k = torch.matmul(self.wk0, x)
		# c K^t Q
		#q_cktq_a = torch.bmm(k.transpose(1, 2), q_cktq_a) * c
		q_cktq_a = torch.bmm(torch.matmul(self.wk0, x).transpose(1, 2), q_cktq_a) * c
		# A = Softmax(c K^t Q)
		# Softmax should be applied to sequence1 axis (dim=1) in A = [batch][sequence1][sequence2]
		# because the sum of weights in sequence1 should be 1. A column in VA is came from
		# V and a column in A, of which the sum should be 1. The weights are used in a weighted sum
		# of V that represents a n-th token.
		q_cktq_a = F.softmax(q_cktq_a, dim = 1)
		# V:[batch, emb, sequence]; A:[batch, sequence, sequence]; VA:[batch, emb, sequence]
		# V
		#v = torch.matmul(self.wv0, x)
		#va = torch.bmm(v, q_cktq_a) -> x
		x = torch.bmm(torch.matmul(self.wv0, x), q_cktq_a)
		#print('dbg1 x=', x.size())

		# FC layer
		# x=VA: [batch, emb, sequence]; -> transpose to [batch, sequence, emb]
		x = self.fc1(torch.transpose(x, 1, 2))
		x = torch.sigmoid(x)
		x = self.fcOut(x)
		#print('dbg2 x=', x.size())
		
		return x

###########################################################################
# Point cloud Dataset
###########################################################################
import glob
import os.path as osp
import torch_geometric.data
import pymeshlab
class Jomon_Rokutanda_1024_Dataset1(torch_geometric.data.InMemoryDataset):
	def __init__(self, rootdirpath, transform = None, train = True):
		# Specify the directories under self.raw_dir.
		# The names of the directories are target labels.
		# Target IDs are labeled in order from 0. Each the direcory should contains 
		# input files in 'train' and 'test' directories for training and test, respectively.
		self.itos = ['asabachi', 'hachi', 'fukabachi', 'daitsukihachi']
		self.stoi = {}
		self.train = train
		self.datasetnames = ['train', 'test']
		for targetid, targetlabel in enumerate(self.itos):
			self.stoi[targetlabel] = targetid
		#print(self.itos, self.stoi)

		# this initialization shoud call after the above setup
		super().__init__(rootdirpath, transform = transform)

		path = self.processed_paths[0] if train else self.processed_paths[1]
		self.load(path)

	@property
	def raw_file_names(self):
		# downloading process of these names will be skipped if the files exist.
		return self.itos

	@property
	def processed_file_names(self):
		# processing pt files of these names will be skipped if the files exist.
		ret = [str + '.pt' for str in self.datasetnames]
		#print('!!!!!!!!!!!!', ret)
		return ret

	def download(self):
		print('error: download() was invoked. Set up the raw directory correctly...')
		
	def process(self):
		print('raw_dir=', self.raw_dir)
		datasetnames = ['train', 'test']
		cnt = [0 for _ in range(len(datasetnames))]
		#print(cnt)
		for ii, datasetname in enumerate(datasetnames):
			# create a list of torch_geometric.data.Data
			datalist = []
			for targetid, targetlabel in enumerate(self.raw_file_names):
				dir = osp.join(self.raw_dir, targetlabel, datasetname)
				files = glob.glob(f'{dir}/*.obj')
				lenfiles = len(files)
				#print(dir)
				#print(files)
				print(datasetname, targetlabel, '[ID', targetid, ']', lenfiles, ' files.')
				cnt[ii] = cnt[ii] + lenfiles
				
				for iii, file in enumerate(files):
					ms = pymeshlab.MeshSet()
					ms.load_new_mesh(file)
					m = ms.current_mesh()
					# v as np.array
					v = m.vertex_matrix()
					v2 = torch.tensor(v, dtype = torch.float)
					# generate torch_geometric.data.Data with the target id = y
					t = torch.tensor([targetid], dtype = torch.long)
					d = torch_geometric.data.Data(pos = v2, y = t)
					# append Data to the list
					datalist.append(d)
					print(' ', targetlabel, '[ID', targetid, '][', iii, ']', d.pos.size(), 'points', file, d.y)

			print('===', datasetname, cnt[ii], 'files in sum.')

			# save the list of Data
			self.save(datalist, self.processed_paths[ii])
			print('===', 'save', datasetname, len(datalist), 'Data to', self.processed_paths[ii])
		

###########################################################################
# A PointNet 1
###########################################################################
from torch_geometric.nn import global_max_pool
class PointNetMini1(nn.Module):
	def __init__(self, num_target_labels):
		super().__init__()
		self.num_target_labels = num_target_labels
		self.mlp1 = nn.Sequential(
			nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
			nn.Linear(128, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
		)
		self.mlp2 = nn.Sequential(
			nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p = 0.3),
			nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p = 0.3),
			nn.Linear(256, num_target_labels),
		)
		#print(self.mlp1)
		#print(self.mlp2)

	def forward(self, btc):
		# btc.pos = [number of points (does not discreminate batch IDs)][3]
		#print('###0', btc.pos.size())
		
		# self.mlp1 uses nn.BatchNorm1d.
		# nn.BatchNorm1d takes [batch][emb]. In this model its [number of points][xyz = 3].
		# A shared MLP of PointNet is shared in all input points, and thus a point is regarded as a batch (It does not care batch IDs).
		# It normalizes over all the input points regardless of batch IDs.
		x = self.mlp1(btc.pos)
		#print('###1', x.size())
		
		# btc.batch = [number of points]. Each a value corresponds to a batch ID of a corresponding embedding vector in x.
		# torch_geometric.nn.global_max_pool discreminates each an element of btc.batch and then aggregate x into max pool system.
		# The output of global_max_pool() = [number of batch IDs][dimension of embedding vector]
		x = global_max_pool(x, btc.batch)
		#print('###2', x.size())
		
		# self.mlp2 uses nn.BatchNorm1d. Different from self.mlp1, x is [number of batch IDs][dimension of embedding vector]
		# because of global_max_pool(). Therefore, the number of batch IDs shoud be greater than 1.
		# input = [number of batch IDs][dimension of embedding vector]
		# output = [number of batch IDs][number of class IDs]
		x = self.mlp2(x)
		#print('###3', x.size())
		return x

###########################################################################
# A PointNet 0
###########################################################################
from torch_geometric.nn import global_max_pool
class PointNetMini0(nn.Module):
	def __init__(self, num_target_labels):
		super().__init__()
		self.num_target_labels = num_target_labels
		self.mlp1 = nn.Sequential(
			nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
		)
		self.mlp2 = nn.Sequential(
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p = 0.3),
			nn.Linear(64, num_target_labels),
		)
		#print(self.mlp1)
		#print(self.mlp2)

	def forward(self, btc):
		# btc.pos = [number of points (does not discreminate batch IDs)][3]
		#print('###0', btc.pos.size())
		
		# self.mlp1 uses nn.BatchNorm1d.
		# nn.BatchNorm1d takes [batch][emb]. In this model its [number of points][xyz = 3].
		# A shared MLP of PointNet is shared in all input points, and thus a point is regarded as a batch (It does not care batch IDs).
		# It normalizes over all the input points regardless of batch IDs.
		x = self.mlp1(btc.pos)
		#print('###1', x.size())
		
		# btc.batch = [number of points]. Each a value corresponds to a batch ID of a corresponding embedding vector in x.
		# torch_geometric.nn.global_max_pool discreminates each an element of btc.batch and then aggregate x into max pool system.
		# The output of global_max_pool() = [number of batch IDs][dimension of embedding vector]
		x = global_max_pool(x, btc.batch)
		#print('###2', x.size())
		
		# self.mlp2 uses nn.BatchNorm1d. Different from self.mlp1, x is [number of batch IDs][dimension of embedding vector]
		# because of global_max_pool(). Therefore, the number of batch IDs shoud be greater than 1.
		# input = [number of batch IDs][dimension of embedding vector]
		# output = [number of batch IDs][number of class IDs]
		x = self.mlp2(x)
		#print('###3', x.size())
		return x


###########################################################################
# Point cloud-Caption Dataset 1 (for ESTCON2024)
###########################################################################
#import glob
#import os.path as osp
#import torch_geometric.data
#import pymeshlab
class Jomon_Rokutanda_1024_Caption_Dataset1(torch_geometric.data.InMemoryDataset):
	def __init__(self, rootdirpath, transform = None, train = True):
		# Specify the directories under self.raw_dir. It should contain input files
		# in 'train' and 'test' directories for training and test, respectively.
		# a pair of input files is a .obj file, a 3D point cloud, and a .obj.cap_tkn_ids, a corresponding caption, respectively.
		# The .obj.cap_tkn_ids file is a text file that contains a sequence of token ids separated with spaces.
		# a special attribute tkns_len is added (tkns_len.size() = torch.Size([1])).
		
		self.train = train
		self.datasetnames = ['train', 'test']

		#self.itos = ['asabachi', 'hachi', 'fukabachi', 'daitsukihachi']
		#self.stoi = {}
		#for targetid, targetlabel in enumerate(self.itos):
		#	self.stoi[targetlabel] = targetid
		#print(self.itos, self.stoi)

		# this initialization shoud call after the above setup
		super().__init__(rootdirpath, transform = transform)

		path = self.processed_paths[0] if train else self.processed_paths[1]
		self.load(path)

	@property
	def raw_file_names(self):
		# downloading process of these names will be skipped if the files exist.
		return self.datasetnames

	@property
	def processed_file_names(self):
		# processing pt files of these names will be skipped if the files exist.
		ret = [str + '.pt' for str in self.datasetnames]
		#print('!!!!!!!!!!!!', ret)
		return ret

	def download(self):
		print('error: download() was invoked. Set up the raw directory correctly...')
		
	def process(self):
		# This Dataset is defined files in the self.raw_dir directory
		# a .obj file is a 3D point cloud and a .obj.cap_tkn_ids is token ids of a corresponding caption.
		# The sequence of token ids is separated with spaces
		print('raw_dir=', self.raw_dir)
		datasetnames = ['train', 'test']
		cnt = [0 for _ in range(len(datasetnames))]
		#print(cnt)
		for ii, datasetname in enumerate(datasetnames):
			# create a list of torch_geometric.data.Data
			datalist = []
			dir = osp.join(self.raw_dir, datasetname)
			files = glob.glob(f'{dir}/*.obj')
			#print(files)
			for iii, file in enumerate(files):
				# generate a point cloud from the file
				ms = pymeshlab.MeshSet()
				ms.load_new_mesh(file)
				m = ms.current_mesh()
				# v as np.array
				v = m.vertex_matrix()
				v2 = torch.tensor(v, dtype = torch.float)
				# generate torch_geometric.data.Data with a target caption = y
				# read corresponding token ids
				file2 = f'{file}.cap_tkn_ids'
				with open(file2, 'r') as f2:
					tkn_ids = list(map(int, f2.read().split()))
				# convert into a token id tensor
				t = torch.tensor(tkn_ids, dtype = torch.long)
				d = torch_geometric.data.Data(pos = v2, y = t, tkns_len = t.size(0))
				# append Data to the list
				datalist.append(d)
				print('[', iii, ']', file, file2, 'token_ids=', tkn_ids, d.pos.size(), 'points', 'y=', d.y)

			print('===', datasetname, len(files), 'files in sum.')

			# save the list of Data
			self.save(datalist, self.processed_paths[ii])
			print('===', 'save', datasetname, len(datalist), 'Data to', self.processed_paths[ii])

###########################################################################
# data augmentation class: 3D rotation of point cloud
###########################################################################
import torch_geometric.transforms
from scipy.spatial.transform import Rotation
# Random 3D rotation
class DataAugPointCloudRotation(torch_geometric.transforms.BaseTransform):
	def __call__(self, data):
		rot = Rotation.random().as_matrix()
		#print(type(data.pos), data.pos.size())
		data.pos = torch.matmul(data.pos, torch.tensor(rot, dtype = torch.float).t())
		return data

###########################################################################
# point cloud-caption model 1
###########################################################################
class PointcloudcCaption1(nn.Module):
	def __init__(self, td_num_vocab, td_embeddim, td_learned_contextlen, pn_embeddim1, pn_embeddim2, pn_embeddim3, pn_embeddim4):
		super().__init__()
		# PointNet
		self.pn_embeddim1 = pn_embeddim1
		self.pn_embeddim2 = pn_embeddim2
		self.pn_embeddim3 = pn_embeddim3
		self.pn_embeddim4 = pn_embeddim4
		self.pnmlp1 = nn.Sequential(
			nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
			nn.Linear(64, self.pn_embeddim1), nn.BatchNorm1d(self.pn_embeddim1), nn.ReLU(),
			nn.Linear(self.pn_embeddim1, self.pn_embeddim2), nn.BatchNorm1d(self.pn_embeddim2), nn.ReLU(),
		)
		self.pnmlp2 = nn.Sequential(
			nn.Linear(self.pn_embeddim2, self.pn_embeddim3), nn.BatchNorm1d(self.pn_embeddim3), nn.ReLU(), nn.Dropout(p = 0.3),
			nn.Linear(self.pn_embeddim3, self.pn_embeddim4), nn.BatchNorm1d(self.pn_embeddim4), nn.ReLU(), nn.Dropout(p = 0.3),
			#nn.Linear(self.pn_embeddim4, num_target_labels),
		)

		# Transformer decoder
		# number of tokens in the vocabulary
		self.td_num_vocab = td_num_vocab
		# embedding dimension
		self.td_embeddim = td_embeddim
		# context length during learning
		# This is only for keeping what context length was used during training of this model.
		self.td_learned_contextlen = td_learned_contextlen
		# QKV attention weights
		self.wq0 = nn.Parameter(torch.randn(self.td_embeddim, self.td_num_vocab, requires_grad = True))
		self.wk0 = nn.Parameter(torch.randn(self.td_embeddim, self.td_num_vocab, requires_grad = True))
		self.wv0 = nn.Parameter(torch.randn(self.td_embeddim, self.td_num_vocab, requires_grad = True))

		# output token: logit for tokens
		# calculated from Transformer decoder input (self.embdim) and PointNet output (self.pn_embeddim4)
		self.tdfc1 = nn.Linear(self.td_embeddim + self.pn_embeddim4, self.td_embeddim)
		self.tdfcOut = nn.Linear(self.td_embeddim, self.td_num_vocab)


	# btcpos and btcbatch are input for PointNet:
	#  supposing btcpos = [number of points (does not discreminate batch IDs)][3] tensor
	#  supposing btcbatch = [number of points (does not discreminate batch IDs)][batch ID] tensor
	#  The batch ID is for descriminating batches such as 0, 1, ..., etc.
	#
	# x is input for Transformer:
	#  supposing x = [batch][expanded batch][self.num_vocab][sequence] tensor
	#
	# Q, K, V, VA: [batch, expanded batch, emb, sequence]
	# A = Softmax(K^t Q / sqrt(d_emb))
	# c = 1 / sqrt(d_emb)
	# A: [batch, expanded batch, sequence, sequence]
	# This does not use self.td_learned_contextlen. Sequence length can be any in the forward().
	def forward(self, btcpos, btcbatch, x):
		##############
		# PointNet
		##############
		# btcpos = [number of points (does not discreminate batch IDs)][3]
		#print('###0', btcpos.size())
		
		# self.mlp1 uses nn.BatchNorm1d.
		# nn.BatchNorm1d takes [batch][emb]. In this model its [number of points][xyz = 3].
		# A shared MLP of PointNet is shared in all input points, and thus a point is regarded as a batch (It does not care batch IDs).
		# It normalizes over all the input points regardless of batch IDs.
		xx = self.pnmlp1(btcpos)
		#print('###1', xx.size())
		
		# btcbatch = [number of points]. Each a value corresponds to a batch ID of a corresponding embedding vector in input xx.
		# torch_geometric.nn.global_max_pool discreminates each an element of btcbatch and then aggregate xx into max pool system.
		# The output of global_max_pool()
		# = [number of batch IDs (row 0 means batch ID of 0, row 1 means that of 1, ...)][dimension of embedding vector]
		xx = global_max_pool(xx, btcbatch)
		#print('###2', xx.size())
		
		# self.mlp2 uses nn.BatchNorm1d. Different from self.mlp1, input xx is [number of batch IDs][dimension of embedding vector]
		# that is output of global_max_pool(). Therefore, the number of batch IDs shoud be greater than 1 for nn.BatchNorm1d.
		# input xx = [number of batch IDs][dimension of embedding vector]
		# output = [number of batch IDs][self.pn_embeddim4]
		xx = self.pnmlp2(xx)
		#print('###3', xx.size())
		
		##############
		# Transformer decoder
		##############
		#print('dbg0 x=', x.size())

		# embedding dimension
		dimemb = len(self.wq0)
		dtyp = self.wq0.dtype

		# The 1st layer: Word embedding with QKV self attention
		c = 1 / torch.sqrt(torch.tensor(dimemb, dtype = dtyp))
		# Q
		# supposing x = [batch][expanded batch][self.num_vocab][sequence] tensor
		# self.wq0 [self.td_embeddim][self.num_vocab] and x [batch][expanded batch][self.num_vocab][sequence]
		# [batch][expanded batch][self.td_embeddim][sequence] = q_cktq_a
		q_cktq_a = torch.matmul(self.wq0, x)
		# K
		#k = torch.matmul(self.wk0, x)
		# c K^t Q
		#q_cktq_a = torch.bmm(k.transpose(1, 2), q_cktq_a) * c
		# matmul([batch][expanded batch][self.td_embeddim][sequence] -> [batch][expanded batch][sequence][self.td_embeddim],
		# [batch][expanded batch][self.td_embeddim][sequence]) = [batch][expanded batch][sequence][sequence] = q_cktq_a
		q_cktq_a = torch.matmul(torch.matmul(self.wk0, x).transpose(2, 3), q_cktq_a) * c
		#q_cktq_a = torch.bmm(torch.matmul(self.wk0, x).transpose(2, 3), q_cktq_a) * c
		# A = Softmax(c K^t Q); c K^t Q = [batch][expanded batch][sequence1][sequence2]
		#  F.softmax([batch][expanded batch][sequence1][sequence2], dim=2) -> [batch][expanded batch][sequence][sequence]
		# Softmax should be applied to sequence1 axis (dim=2) in A = [batch][expanded batch][sequence1][sequence2]
		# because the sum of weights in sequence1 should be 1. A column in VA is came from
		# V and a column in A, of which the sum should be 1. The weights are used in a weighted sum
		# of V that represents a n-th token.
		q_cktq_a = F.softmax(q_cktq_a, dim = 2)

		# V:[batch][expanded batch][self.td_embeddim][sequence]
		# A:[batch][expanded batch][sequence][sequence]
		#v = torch.matmul(self.wv0, x)
		#va = torch.matmul(v, q_cktq_a) -> x
		# VA:[batch][expanded batch][self.td_embeddim][sequence] = output x
		x = torch.matmul(torch.matmul(self.wv0, x), q_cktq_a)
		#x = torch.bmm(torch.matmul(self.wv0, x), q_cktq_a)
		#print('dbg1 x=', x.size())

		# FC layer
		#####################################################
		# concatenate embedding vector xx (PointNet out) to embedding vector of x (Transformer decoder attention out)
		#####################################################
		# x=VA: [batch][expanded batch][self.td_embeddim][sequence]
		#  -> transpose to [batch][expanded batch][sequence][self.td_embeddim]
		# xx=[batch][self.pn_embeddim4]->[batch][1][1][self.pn_embeddim4]->[batch][expanded batch][sequence][self.pn_embeddim4]
		# concatenate x and xx = [batch][expanded batch][sequence][self.td_embeddim + self.pn_embeddim4]
		n_epnd_batch = x.size(1)
		nsequence = x.size(3)
		x = torch.cat((torch.transpose(x, 2, 3), xx.unsqueeze(1).unsqueeze(1).expand(-1, n_epnd_batch, nsequence, -1)), dim = 3)
		# input x = [batch][expanded batch][sequence][self.td_embeddim + self.pn_embeddim4]
		# output = [batch][expanded batch][sequence][self.td_embeddim]
		x = self.tdfc1(x)
		# output = [batch][expanded batch][sequence][self.td_embeddim]
		x = torch.sigmoid(x)
		# output x = [batch][expanded batch][sequence][self.td_num_vocab]
		x = self.tdfcOut(x)
		#print('dbg2 x=', x.size())
		
		return x

###########################################################################
# Point cloud-Caption Dataset (131072 points model)
#
# caption not implemented.
###########################################################################
#import glob
#import os.path as osp
#import torch_geometric.data
#import pymeshlab
class Jomon_131072_Caption_Dataset1(torch_geometric.data.InMemoryDataset):
	def __init__(self, rootdirpath, transform = None, train = True):
		# Specify the directories under self.raw_dir. It should contain input files
		# in 'train' and 'test' directories for training and test, respectively.
		# a pair of input files is a .obj file, a 3D point cloud, and a .obj.cap_tkn_ids, a corresponding caption, respectively.
		# The .obj.cap_tkn_ids file is a text file that contains a sequence of token ids separated with spaces.
		# a special attribute tkns_len is added (tkns_len.size() = torch.Size([1])).
		
		self.train = train
		self.datasetnames = ['train', 'test']

		# this initialization shoud call after the above setup
		super().__init__(rootdirpath, transform = transform)

		path = self.processed_paths[0] if train else self.processed_paths[1]
		self.load(path)

	@property
	def raw_file_names(self):
		# downloading process of these names will be skipped if the files exist.
		return self.datasetnames

	@property
	def processed_file_names(self):
		# processing pt files of these names will be skipped if the files exist.
		ret = [str + '.pt' for str in self.datasetnames]
		#print('!!!!!!!!!!!!', ret)
		return ret

	def download(self):
		print('error: download() was invoked. Set up the raw directory correctly...')
		
	def process(self):
		# This Dataset is defined files in the self.raw_dir directory
		# a .obj file is a 3D point cloud and a .obj.cap_tkn_ids is token ids of a corresponding caption.
		# The sequence of token ids is separated with spaces
		print('raw_dir=', self.raw_dir)
		datasetnames = ['train', 'test']
		cnt = [0 for _ in range(len(datasetnames))]
		#print(cnt)
		for ii, datasetname in enumerate(datasetnames):
			# create a list of torch_geometric.data.Data
			datalist = []
			dir = osp.join(self.raw_dir, datasetname)
			files = glob.glob(f'{dir}/*.obj')
			#print(files)
			for iii, file in enumerate(files):
				# generate a point cloud from the file
				ms = pymeshlab.MeshSet()
				ms.load_new_mesh(file)
				m = ms.current_mesh()
				# v as np.array
				v = m.vertex_matrix()
				v2 = torch.tensor(v, dtype = torch.float)
				
				###
				### caption not yet implemented. 20240704
				###
				
				# generate torch_geometric.data.Data with a target caption = y
				# read corresponding token ids
				file2 = None
				#file2 = f'{file}.cap_tkn_ids'
				#with open(file2, 'r') as f2:
				#	tkn_ids = list(map(int, f2.read().split()))
				# convert into a token id tensor
				tkn_ids = [0] # dummy 20240704
				t = torch.tensor(tkn_ids, dtype = torch.long)
				d = torch_geometric.data.Data(pos = v2, y = t, tkns_len = t.size(0))
				# append Data to the list
				datalist.append(d)
				print('[', iii, ']', file, file2, 'token_ids=', tkn_ids, d.pos.size(), 'points', 'y=', d.y)

			print('===', datasetname, len(files), 'files in sum.')

			# save the list of Data
			self.save(datalist, self.processed_paths[ii])
			print('===', 'save', datasetname, len(datalist), 'Data to', self.processed_paths[ii])

###########################################################################
# point cloud SSL model 1
###########################################################################
from torch_geometric.nn import MLP, PointNetConv
from torch_cluster import fps, radius
# SAModule: copied from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
class SAModule(torch.nn.Module):
	# ratio = reduced No. of points / total No. of points
	# r = radius
	# nn = PointNet MLP
	def __init__(self, ratio, r, nn):
		super().__init__()
		self.ratio = ratio
		self.r = r
		self.conv = PointNetConv(nn, add_self_loops=False)

	def forward(self, x, pos, batch):
		idx = fps(pos, batch, ratio=self.ratio)
		row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
		#                      max_num_neighbors=64)
		max_num_neighbors=512)
		edge_index = torch.stack([col, row], dim=0)
		x_dst = None if x is None else x[idx]
		x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
		pos, batch = pos[idx], batch[idx]
		return x, pos, batch
        
class PointcloudcSSL1(nn.Module):
	def __init__(self, pn_params, td_params):
		super().__init__()
		# PointNet++ level
		self.pn = SAModule(pn_params['point_ratio'], pn_params['radius'], MLP([3, 8, 8]))


	# btcpos and btcbatch are input for PointNet:
	#  supposing btcpos = [number of points (does not discreminate batch IDs)][3] tensor
	#  supposing btcbatch = [number of points (does not discreminate batch IDs)][batch ID] tensor
	#  The batch ID is for descriminating batches such as 0, 1, ..., etc.
	#
	#from################old and will be updated (20240820)
	# x is input for Transformer:
	#  supposing x = [batch][expanded batch][self.num_vocab][sequence] tensor
	#
	# Q, K, V, VA: [batch, expanded batch, emb, sequence]
	# A = Softmax(K^t Q / sqrt(d_emb))
	# c = 1 / sqrt(d_emb)
	# A: [batch, expanded batch, sequence, sequence]
	# This does not use self.td_learned_contextlen. Sequence length can be any in the forward().
	#to################old and will be updated (20240820)
	def forward(self, btcpos, btcbatch, x):
		# return value is output and thus it was caption when a caption model. (will be updated to kind of feature vector in the case of SSL model)
		x = self.pn(btcpos, None, btcbatch)
		return x














