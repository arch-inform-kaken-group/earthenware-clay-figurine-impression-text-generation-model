import time
import sys
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import pymeshlab
import ec_utils

if len(sys.argv) != 2:
	print(f'Usage: python {sys.argv[0]} file.obj\n\tfile.obj= an input obj file for captioning.\n')
	exit(-1)

print('-------------------------------------------------')
print('Inference engine:')
print('Prototype of Jomon pottery point cloud-Japanese')
print('language captioning model')
print('(c)2024 NUIS')
print('-------------------------------------------------')

objfile = sys.argv[1]
if not Path(objfile).exists():
	print(f'{objfile} does not exists...\n')
	sys.exit(-1)

# measure execution time in seconds
t0 = time.perf_counter()

###########################################################################
# hyper parameters
###########################################################################
#hyp_optm = {'name':'optimizer', 'params':['AdamW', 'Adam', 'SGD']}
#hyp_lrnrate = {'name':'learning_rate', 'params':[1e-5, 1e-6]}
#hyp_btcsize = {'name':'batch_size', 'params':[2, 4, 8, 16, 32]}

###########################################################################
# run string
# several output files and tensorboard logs are named based on this string 
###########################################################################
# replace special characters to _ for a file system
run_str_base = re.sub(r'[\<\>\:\"\/\\\|\?\*]', '_', sys.argv[1])
print('Output files and tensorboard logs are named based on: ' + run_str_base)
#sys.exit(-1)

###########################################################################
# device
###########################################################################
gpuid = 0
#try:
#	gpuid = int(sys.argv[2])
#except ValueError:
#	print('error: gpu_id must be int >= 0...')
#	sys.exit(-1)
device = torch.device('cuda:' + str(gpuid)) if torch.cuda.is_available() else torch.devce('cpu')
#device = 'cpu'
print('device=', device)
#sys.exit(-1)

###########################################################################
# Dataset source
###########################################################################
dstdir = '/home/chika/db/torch_geometric_datasets/jomon1/rokutanda1_pc1024_cap'
#dstdir = '/home/nuis1/db/torch_geometric_datasets/jomon1/rokutanda1_pc1024_cap'
print('dataset=', dstdir)

###########################################################################
# torch_geometric.data.Dataset
###########################################################################
trnsf_trn = ec_utils.DataAugPointCloudRotation()
dstrn = ec_utils.Jomon_Rokutanda_1024_Caption_Dataset1(dstdir, train = True, transform = trnsf_trn)
dstst = ec_utils.Jomon_Rokutanda_1024_Caption_Dataset1(dstdir, train = False, transform = trnsf_trn)
lentrn = len(dstrn)
lentst = len(dstst)
print('train=', len(dstrn), type(dstrn))
print('test=', len(dstst), type(dstst))
#print(dstrn[0])
#print(dstrn[0]['pos'][:10])
##print(dstrn[0]['normal'][:10])
#print(dstrn[0]['y'])
#import pymeshlab
#for ddd in [dstrn, dstst]:
#	for i, d in enumerate(ddd):
#		if (ddd == dstrn and i in [0, 1, 9, 22, 31]) or (ddd == dstst and i in [2, 5, 8, 9]):
#			print(ddd.train, i, d, d.y, d.pos)
#			new_mesh = pymeshlab.Mesh(vertex_matrix = d.pos.numpy())
#			new_data = pymeshlab.MeshSet()
#			new_data.add_mesh(new_mesh)
#			new_data.save_current_mesh('tmp/' + str(ddd.train) + str(i) + '.obj')
#		else:
#			pass
#			#break
#sys.exit(-1)

#############################################################################################
# Vocab
#############################################################################################
vocfile = 'rokutanda1_pc1024_cap.voc.pth'
voc = torch.load(vocfile)
print('vocabulary=', vocfile)
print(type(voc), 'num vocabulary=', len(voc))
print(voc.get_itos())
print()
print(voc.get_stoi())
print()
print('<pad>', voc.get_stoi()['<pad>'], '<unk>', voc.get_stoi()['<unk>'], '<bos>', voc.get_stoi()['<bos>'], '<eos>', voc.get_stoi()['<eos>'])
print()

#############################################################################################
# 1-hot dimension for tokens = len(voc) = the number of token IDs
#############################################################################################
num_vocab = len(voc)

#############################################################################################
# max length for output tokens of the model
#############################################################################################
max_token_length = 512
print('device=', device)
print('max_token_length=', max_token_length)

#############################################################################################
# xInput: a tensor buffer for Transformer decoder inputs
#############################################################################################
td_max_batchsize = 1
#td_max_batchsize = 32
td_max_expansion_batchsize = 1
#td_max_expansion_batchsize = 512
# xInput = tensor buffer for 1-hot vector sequence
# supposing [td_max_batchsize][td_max_expansion_batchsize][num_vocab][max_token_length] tensor for xInput
# this global tensor is initialized one time to zeros only the first time.
xInput = torch.zeros(td_max_batchsize, td_max_expansion_batchsize, num_vocab, max_token_length).to(device)

#############################################################################################
# set_1hot: set 1-hot vectors from batch of sequences of token IDs.
# x = specify the xInput (or another tensor buffer)
# token_index_sequence = [batch][expanded batch][seq of token IDs (applied to these part)][seq (sliding alog seq)]
#############################################################################################unittest
print('xInput', xInput.size())
def set_1hot(x, token_index_sequence):
	#x[0, [0,1], [0,1]] = 1
	#for i, tknidxseq in enumerate(token_index_sequence):
	#	x[i, tknidxseq, [j for j in range((token_index_sequence.size()[1]))]] = 1
	
	# clear to all 0 because xInput is repeatedly used.
	x.zero_()
	bchsize = token_index_sequence.size(0)
	expnd_bchsize = token_index_sequence.size(1)
	b = torch.arange(bchsize).unsqueeze(1).unsqueeze(2)
	#print('b=', b.size())
	#print(b)
	expnd_b = torch.arange(expnd_bchsize).unsqueeze(1).unsqueeze(0)
	#print('expnd_b=', expnd_b.size())
	#print(expnd_b)
	#print('token_index_sequence=', token_index_sequence.size())
	#print(token_index_sequence)
	seqlen = token_index_sequence.size(2)
	s1 = torch.arange(seqlen)
	#print('s1=', s1.size())
	#print(s1)
	s2 = torch.empty(bchsize, expnd_bchsize, seqlen, dtype = int)
	s2[:,:,:] = s1
	#print('s2=', s2.size())
	#print(s2)
	x2 = x[:bchsize, :expnd_bchsize, :, :seqlen]
	x2[b, expnd_b, token_index_sequence, s2] = 1
	#print('bchsize=', bchsize, 'expnd_bchsize=', expnd_bchsize, 'seqlen=', seqlen, 'x=', x.size(), 'x2=', x2.size(), 'b=', b, 'expnd_b=', expnd_b, 's1=', s1, 's2=', s2)
	#print('bchsize=', bchsize, 'expnd_bchsize=', expnd_bchsize, 'seqlen=', seqlen, 'x=', x.size(), 'x2=', x2.size())

	#print('set_1hot() tkn_idx_seq=', len(token_index_sequence), token_index_sequence)
	#print('x=', x.size())
	#for i, b in enumerate(x):
	#	for j, d in enumerate(b):
	#		if d.sum() > 0.5:
	#			print(i, j, d)
	return x2

#############################################################################################
# set_1hot_without_clear: set only a 1-hot vector from a specified index and ID without clear.
# x = specify the xInput (or another tensor buffer)
# token_id = [batch][expanded batch][token ID (this)][sequence_index]
# sequence_index = [batch][expanded batch][token ID][sequence_index (this)]
#############################################################################################unittest
def set_1hot_without_clear(x, token_id, sequence_index):
	#print('sequence_index=', sequence_index)
	#print('token_id=', token_id)
	#print('x=', x)
	x[0, 0, token_id, sequence_index] = 1
	#print(x.size())
	return

#############################################################################################
# load a obj file and return batpos and batbatch:
# return:
#  batpos = [num of points][xyz = 3], batbatch = [num of points]; in batbatch values are all batch ID 0s.
#############################################################################################unittest
def load_point_cloud_from_obj(file):
	# generate a point cloud from the file
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(file)
	m = ms.current_mesh()
	# points as a np.array
	v = m.vertex_matrix()
	# points as a tensor
	v2 = torch.tensor(v, dtype = torch.float)
	# batbatch
	bb = torch.zeros(v2.size(0), dtype = torch.long)
	return v2.to(device), bb.to(device)













print('Model description:')
# embedding dimension

#いらなければnum_vocabに書き換える
dim_token = num_vocab

# index for special tokens
pad = voc['<pad>']
unk = voc['<unk>']
eos = voc['<eos>']
bos = voc['<bos>']
# vocabulary index to str
voc_itos = voc.get_itos()
# model and vocabulary
#input_modelfilename = 'PointcloudcCaption1.pcp1_100k.try0.bc4.Adam1e-06.pth.ckpt20000'
input_modelfilename = 'PointcloudcCaption1.pcp1_100k.try0.bc4.Adam1e-06.pth'
#input_modelfilename = 'PointcloudcCaption1.pcp1_rnd.try0.bc4.Adam1e-05.pth'
model = torch.load(input_modelfilename)
# ouput model info
modelclassname = model.__class__.__name__
print(type(model), modelclassname)
print(model)
print('model parameters:')
#	num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_parameters = 0
for n, p in model.named_parameters():
	if p.requires_grad:
		nprm = p.numel()
		num_parameters = num_parameters + nprm
		print(n, nprm)
print(num_parameters, 'parameters in sum')
print()
#print('parameters=', num_parameters, [n for n, p in model.named_parameters() if p.requires_grad])
model = model.to(device)
if model.td_learned_contextlen > max_token_length:
	print('error: learned_contextlen', model.td_learned_contextlen, 'exceeds max_token_length', max_token_length, '...')
	sys.exit(-1)






# initial input
input_indices = [bos]
set_1hot_without_clear(xInput, bos, 0)
new_seq_index_to_set = 1

#input_indices = [bos, voc['深鉢']]
#set_1hot_without_clear(xInput, bos, 0)
#set_1hot_without_clear(xInput, voc['深鉢'], 1)
#new_seq_index_to_set = 2

initial_tokens = [voc_itos[i] for i in input_indices]
print('initial input=', initial_tokens)
print('------------------------------------')

# generate text
model.eval()
with torch.no_grad():
	##########################################################
	# point cloud from obj file
	##########################################################
	batpos, batbatch = load_point_cloud_from_obj(objfile)
	batpos.to(device)
	batbatch.to(device)
	#print('!!!', batpos.size(), batpos.device, batpos)
	#print('!!!', batbatch.size(), batbatch.device, batbatch)

	##########################################################
	# inference of token IDs based on the point cloud
	##########################################################
	# token generation will be terminated if new_seq_index_to_set (the number of loop + initial tokens) exceeds max_token_length
	while new_seq_index_to_set < max_token_length:
		# generate a token from the token sequence
		# supposing [batch][exp_batch][num_vocab][sequence] tensor for xInput
		# both batch and exp_batch are always 1 for this inference task
		# slice xInput: this does not copy xInput and just reduce the sequence length and thus efficient slightly
		# input begin index
		seq_index_from = 0
		# adjust the input sequence cut down to model.td_learned_contextlen in maximum
		if new_seq_index_to_set > model.td_learned_contextlen:
			seq_index_from = new_seq_index_to_set - model.td_learned_contextlen
		xInput2 = xInput[:, :, :, seq_index_from:new_seq_index_to_set]
		# the 1st input of the model is btcpos
		# the 2nd input of the model is btcbatch
		# the 3rd input of the model is [batch][exp_batch][num_vocab][sliced sequence] came from the xInput
		# output of the model is [batch][exp_batch][sequence][model.td_num_vocab]
		x = model(batpos, batbatch, xInput2)
		#print('[', new_seq_index_to_set, ']', 'xInput2', xInput2.size(), xInput2)
		#for i, t in enumerate(xInput2[0].t()):
		#	idx = torch.argmax(t).item()
		#	itkn = voc_itos[idx]
		#	print('[', new_seq_index_to_set, '][', i, ']input=', idx, itkn)
		
		# token probability vectors in the sequence
		# x = [batch][exp_batch][sequence][model.td_num_vocab]
		# computing token ID probability; dim should be the axis in which probabilities are computed
		# p = [batch][exp_batch][sequence][model.td_num_vocab]
		p = F.softmax(x, dim = 3)
		# probabilistically sampled index sequence
		# an token ID from one [model.td_num_vocab] is sampled for all the sequence, exp_batch, batch.
		# and thus, sampled_idx_seq = [batch][exp_batch][sequence]
		sampler = torch.distributions.Categorical(p)
		sampled_idx_seq = sampler.sample()
		#print('[', new_seq_index_to_set, ']', 'x(out)=', x.size(), x)
		#print('[', new_seq_index_to_set, ']', 'p=', p.size(), p)
		#print('[', new_seq_index_to_set, ']', 'sampler=', type(sampler), sampler)
		#for i, t in enumerate(x[0]):
		#	idx = torch.argmax(t).item()
		#	otkn = voc_itos[idx]
		#	print('[', new_seq_index_to_set, '][', i, ']output max p=', idx, otkn)
		
		# the predicted token ID; an output token is from the probabilistically sampled index at the only latest prediction
		idx = sampled_idx_seq[0, 0, -1].item()
		# the predected token (in string)
		#otkn = voc_itos[idx]
		#result.append(otkn)
		# update input token sequence
		input_indices.append(idx)
		#print('[', new_seq_index_to_set, ']', 'sampled=', idx, otkn)

		# update x (just add the latest token ID as an one-hot vector)
		set_1hot_without_clear(xInput, idx, new_seq_index_to_set)	
		# update new_seq_index_to_set
		new_seq_index_to_set += 1
		if idx == eos:
			# finish when the output token is eos
			break
		
	##########################################################
	# output the text generation result
	##########################################################
	print('new_seq_index_to_set=', new_seq_index_to_set, 'result=', ' '.join([voc_itos[tknid] for tknid in input_indices]))
	#print('new_seq_index_to_set=', new_seq_index_to_set, 'result=', result)
	
# measure execution time in seconds
tEnd = time.perf_counter()

print('------------------------------------')
print('execution time (s):', (tEnd - t0))



