import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import time

# measure execution time in seconds
t0 = time.perf_counter()

#device = 'cuda'
device = 'cpu'

max_token_length = 128
print('------------------------------------')
print('device=', device)
print('max_token_length=', max_token_length)
print('------------------------------------')
print('Model description:')

# model and vocabulary
#modelfile = 'tf1.pth.ctx3_em256_ep0_bc4'
#modelfile = 'tf1.pth.ctx2_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em1_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em2_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em4_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em8_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em16_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em32_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em64_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em128_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em256_ep100_bc4'
#modelfile = 'tf1.pth.ctx3_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx4_em64_ep100_bc4'
#modelfile = 'tf1.pth.ctx4_em256_ep100_bc4'
#modelfile = 'tf1.pth.ctx4_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx5_em512_ep0'
#modelfile = 'tf1.pth.ctx5_em512_ep100_bc4'
modelfile = 'tf1.pth.ctx5_em512_ep1000_bc4'
#modelfile = 'tf1.pth.ctx6_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx7_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx8_em256_ep100_bc4'
#modelfile = 'tf1.pth.ctx8_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx16_em512_ep100_bc4'
#modelfile = 'tf1.pth.ctx32_em512_ep100_bc4'
vocfile = 'rokutanda_doki_n.txt.voc.pth'
voc = torch.load(vocfile)
# embedding dimension
dim_token = len(voc)
# index for special tokens
pad = voc['<pad>']
unk = voc['<unk>']
eos = voc['<eos>']
bos = voc['<bos>']
print(type(voc), 'num vocabulary=', dim_token)
print('<pad>', pad, '<unk>', unk, '<bos>', bos, '<eos>', eos)
# vocabulary index to str
voc_itos = voc.get_itos()

# model
model = torch.load(modelfile)
model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('modelfile=', modelfile)
print('num_parameters=', num_parameters)
print('learned_contextlen=', model.learned_contextlen)
print(model)
print('------------------------------------')

if model.learned_contextlen > max_token_length:
	print('error: learned_contextlen', model.learned_contextlen, 'exceeds max_token_length', max_token_length, '...')
	exit(-1)

# xInput = 1-hot vector sequence
# supposing [batch][self.num_vocab][sequence] tensor for xInput
# batch is always 1 for this inference task
# global tensor for x, which is initialized to zeros.
xInput = torch.zeros(1, dim_token, max_token_length).to(device)
print('xInput', xInput.size())
def set_1hot_without_clear(x, token_index, sequence_index):
	x[0, token_index, sequence_index] = 1
	#print(x.size())
	return

# initial input
input_indices = [bos]
set_1hot_without_clear(xInput, bos, 0)
seq_index = 1

#input_indices = [bos, voc['深鉢']]
#set_1hot_without_clear(xInput, bos, 0)
#set_1hot_without_clear(xInput, voc['深鉢'], 1)
#seq_index = 2

initial_tokens = [voc_itos[i] for i in input_indices]
print('initial input=', initial_tokens)
print('------------------------------------')

# generate text
model.eval()
with torch.no_grad():
	result = []
	# token generation will be terminated if seq_index (the number of loop + initial tokens) exceeds max_token_length
	while seq_index < max_token_length:
		# generate a token from the token sequence
		# supposing [batch][num_vocab][sequence] tensor for xInput
		# batch is always 1 for this inference task
		# slice xInput: this does not copy xInput and just reduce the sequence length and thus efficient slightly
		seq_index_0 = 0
		if seq_index > model.learned_contextlen:
			seq_index_0 = seq_index - model.learned_contextlen
		xInput2 = xInput[:, :, seq_index_0:seq_index]
		
		# input of the model is [batch][num_vocab][sliced sequence] came from the xInput
		# output of the model is [batch][sequence][num_vocab]
		x = model(xInput2)
		#print('[', seq_index, ']', 'xInput2', xInput2.size(), xInput2)
		#for i, t in enumerate(xInput2[0].t()):
		#	idx = torch.argmax(t).item()
		#	itkn = voc_itos[idx]
		#	print('[', seq_index, '][', i, ']input=', idx, itkn)
		
		# token probability vectors in the sequence
		# x = [batch][sequence][num_vocab]
		p = F.softmax(x, dim = 2)
		sampler = torch.distributions.Categorical(p)
		# probabilistically sampled index sequence
		sampled_idx_seq = sampler.sample()
		#print('[', seq_index, ']', 'x(out)=', x.size(), x)
		#print('[', seq_index, ']', 'p=', p.size(), p)
		#print('[', seq_index, ']', 'sampler=', type(sampler), sampler)
		#for i, t in enumerate(x[0]):
		#	idx = torch.argmax(t).item()
		#	otkn = voc_itos[idx]
		#	print('[', seq_index, '][', i, ']output max p=', idx, otkn)
		
		# an output token is from the probabilistically sampled index at the latest prediction
		idx = sampled_idx_seq[0][-1].item()
		otkn = voc_itos[idx]
		result.append(otkn)
		# update input token sequence
		input_indices.append(idx)
		#print('[', seq_index, ']', 'sampled=', idx, otkn)

		# update x
		set_1hot_without_clear(xInput, idx, seq_index)	
		# update seq_index
		seq_index += 1
		if idx == eos:
			# finish when the output token is eos
			break
		
	#print('seq_index=', seq_index, 'result=', result)
	print('seq_index=', seq_index, 'result=', ' '.join(result))
	
# measure execution time in seconds
tEnd = time.perf_counter()

print('------------------------------------')
print('execution time (s):', (tEnd - t0))




