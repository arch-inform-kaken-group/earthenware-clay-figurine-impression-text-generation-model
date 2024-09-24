import torch
import ec_utils
import time

vocfile = 'rokutanda_doki_n.txt.voc.pth'
dsfile = 'rokutanda_doki_n.txt.dataset.pth'

# measure execution time in seconds
t0 = time.perf_counter()

#############################################################################################
# Device
#############################################################################################
device = 'cuda'
#device = 'cpu'
#############################################################################################
# Max and batch size
#############################################################################################
max_batchsize = 1024
bchsize = 4
#############################################################################################
# Context length for input tokens of the model
#############################################################################################
context_token_length = 5
#max_token_length = 64 (deprecated)
#############################################################################################
# Vocab and Dataset
#############################################################################################
voc = torch.load(vocfile)
ds = torch.load(dsfile)
#############################################################################################
# 1-hot dimension for tokens
#############################################################################################
dim_token = len(voc)
#############################################################################################
# embedding dimension
#############################################################################################
embeddim = 512
#embeddim = 128
#############################################################################################
# number of epochs
#############################################################################################
#nepoch = 1000
nepoch = 0
#############################################################################################
# output model file name
#############################################################################################
modelfilename = f'tf1.pth.ctx{context_token_length}_em{embeddim}_ep{nepoch}_bc{bchsize}.latest'
#print(modelfilename)
#exit(-1)

# the index of special tokes in the Vocab
pad = voc['<pad>']
unk = voc['<unk>']
bos = voc['<bos>']
eos = voc['<eos>']

print(type(voc), 'num vocabulary=', len(voc))
print('<pad>', pad, '<unk>', unk, '<bos>', bos, '<eos>', eos)
#print(voc.get_itos())
#print()
#print(voc.get_stoi())
#print()

print(type(ds), 'num dataset=', len(ds))
#for i, tkns in enumerate(ds):
#	print(i, tkns)

#############################################################################################
# convert tokens in a batch to ids with padding and then into a tensor
#############################################################################################
def ids_collate(bch):
	#print('ids_collate: bch=', type(bch), bch)
	# max length in the batch
	maxlen = max(len(tkns) for tkns in bch)
	# to ids
	ret = [[voc[t] for t in ts] for ts in bch]
	# padding with the index of '<pad>'
	ret = [r if len(r) == maxlen else r + [pad] * (maxlen - len(r)) for r in ret]
	#print('ids_collate: ret=', type(ret), ret)
	#itos = voc.get_itos()
	#print(ret, [[itos[i] for i in ids] for ids in ret])
	#print('ids_collate=', [[itos[i] for i in ids] for ids in ret])

	# return a tensor
	return torch.tensor(ret, dtype = torch.long, device=device)
	
#############################################################################################
# THIS PROGRAM DOES NOT USE this typical function as collate_fn for the DataLoader caused of inefficiency.
# convert tokens in a batch to ids with padding and then into a tensor
#############################################################################################
#def ids_collate_old(bch):
#	#print(bch)
#	# to ids
#	ret = [[voc[t] for t in ts] for ts in bch]
#	# padding with the index of '<pad>'
#	ret = [r[:max_token_length] if len(r) >= max_token_length else r + [pad] * (max_token_length - len(r)) for r in ret]
#	print(ret)
#	#itos = voc.get_itos()
#	#print(ret, [[itos[i] for i in ids] for ids in ret])
#	#print('ids_collate=', [[itos[i] for i in ids] for ids in ret])
#
#	# return a tensor
#	return torch.tensor(ret, dtype = torch.int)

#############################################################################################
# generate input and target tensors for a leaning with a specified context length 
# from a DataLoader batch, inp (2-rank tensor [batch][sequence ID]).
# If the token sequence length of inp <= cntxt_len, return (None, None); else,
# return (torch.Tensor[batch][cntxt_len] for batch of input token ID sequences, torch.Tensor[batch] for batch of target token IDs)
# Learning data pairs of which each pair has the target <pad> will be removed from the generated batch.
# Notify the generated number of batchs will be determined dynamically, so possibly produces an error caused of
# an excess of max_batchsize.
#############################################################################################
def gen_batch(inp, cntxt_len):
	seqlen = inp.size()[1]
	numbchPerSeq = seqlen - cntxt_len
	if numbchPerSeq <= 0:
		return None, None
	else:
		#print('seqlen=', seqlen, 'numbchPerSeq=', numbchPerSeq, 'cntxt_len=', cntxt_len, 'inp=', inp.size(), inp)
		#print('seqlen=', seqlen, 'numbchPerSeq=', numbchPerSeq, 'cntxt_len=', cntxt_len, 'inp=', inp.size())
		# list of input, [batch][cntxt_len], from inp
		genbch = [inp[:, i:i + cntxt_len] for i in range(numbchPerSeq)]
		# list of target tokens, [batch][1] from inp
		# this results in [batch] because automatic squeeze()
		gentgt = [inp[:, i + cntxt_len] for i in range(numbchPerSeq)]
		# create pairs of batch inputs and targets.
		# 1. Flatten along batch axis.
		# 2. remove pairs that has a target <pad>
		binp = torch.cat(genbch, dim = 0)
		btgt = torch.cat(gentgt, dim = 0)
		indices = btgt != pad
		#print('gen_batch gentgt=', gentgt, 'btgt=', btgt.size(), btgt, 'indices=', type(indices), indices.size(), indices, 'indices.squeeze=', indices.squeeze().size())
		#print('gen_batch binp=', binp.size(), binp)
		# extract pairs of not <pad> inputs and target
		binp = binp[indices]
		btgt = btgt[indices]
		#print('gen_batch btgt2=', btgt.size(), btgt)
		#print('gen_batch binp2=', binp.size(), binp)
		genbatch_len = len(btgt)
		if genbatch_len > max_batchsize:
			print('ERROR: A generated batch data [', genbatch_len, '] exceeded max_batchsize,', max_batchsize, '. Change max_batchsize...')
			exit(-1)
			
		#print('gen_batch() btgt=', btgt.size(), 'binp=', binp.size())
		#print('btgt=', btgt, 'binp=', binp)
		return binp, btgt

#############################################################################################
# Batch tensor DataLoader with the Dataset and the batch size
#############################################################################################
ldr = torch.utils.data.DataLoader(ds, collate_fn = ids_collate, batch_size = bchsize, shuffle = True)

print(type(ldr), len(ldr))
#a = next(iter(ldr))
#print(a)
#for i, d in enumerate(ldr):
#	print(type(d), i, d.shape, d)

#############################################################################################
# xInput: a tensor buffer for inputs
#############################################################################################
# xInput = 1-hot vector sequence
# supposing [max_batchsize][num_vocab][context_token_length (sequence axis)] tensor for xInput
# global tensor for x, which is initialized to zeros.
#xInput = torch.zeros(max_batchsize, dim_token, max_token_length).to(device)
xInput = torch.zeros(max_batchsize, dim_token, context_token_length).to(device)
print('xInput', xInput.size())
def set_1hot(x, token_index_sequence):
#def set_1hot(x, cntxt_len, token_index_sequence):
	#x[0, [0,1], [0,1]] = 1
	#for i, tknidxseq in enumerate(token_index_sequence):
	#	x[i, tknidxseq, [j for j in range((token_index_sequence.size()[1]))]] = 1
	x.zero_()
	bchsize = len(token_index_sequence)
	seqlen = token_index_sequence.size()[1]
	b = torch.arange(bchsize).unsqueeze(dim = 1)
	s1 = torch.arange(seqlen)
	s2 = torch.empty(bchsize, seqlen, dtype = int)
	s2[:] = s1
	x2 = x[:bchsize, :, :seqlen]
	x2[b, token_index_sequence, s2] = 1
	#print('bchsize=', bchsize, 'seqlen=', seqlen, 'x=', x.size(), 'x2=', x2.size(), 'b=', b, 's1=', s1, 's2=', s2)
	#print('bchsize=', bchsize, 'seqlen=', seqlen, 'x=', x.size(), 'x2=', x2.size())

	#print('set_1hot() tkn_idx_seq=', len(token_index_sequence), token_index_sequence)
	#print('x=', x.size())
	#for i, b in enumerate(x):
	#	for j, d in enumerate(b):
	#		if d.sum() > 0.5:
	#			print(i, j, d)
	return x2

#############################################################################################
# model
#############################################################################################
print(type(voc), 'num vocabulary=', len(voc))
print(type(ds), 'num dataset=', len(ds))

model = ec_utils.TfDecoderMini1(len(voc), embeddim, context_token_length)
print(model)
print([(nm, p.size()) for nm, p in model.named_parameters()])
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num_parameters=', num_parameters)

model.to(device)

#exit(-1)

#############################################################################################
# optimizer
#############################################################################################
from torch import optim

optim = optim.Adam(model.parameters(), lr = 5e-5)
#optim = optim.AdamW(model.parameters(), lr = 0.001)
#optim = optim.SGD(model.parameters(), lr = 0.01)
#optim = optim.SGD(model.parameters(), lr = 0.1)

#############################################################################################
# loss
#############################################################################################
import torch.nn as nn
fLoss = nn.CrossEntropyLoss()

#############################################################################################
# learn
#############################################################################################
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
cpath = Path.cwd()
logdir = cpath / 'log'
logdir.mkdir(exist_ok = True)
print('log=', logdir)
wtr = SummaryWriter(log_dir = logdir)

model.train()

#wq1 = model.wq0[:,136].clone()
#wk1 = model.wk0[:,136].clone()
#wv1 = model.wv0[:,136].clone()
#print('深鉢', voc['深鉢'], wq1, wk1, wv1)

# エポックをnepoch回ループ
for iEp in range(nepoch):
	# 1エポック
	lossepc = 0
	accepc = 0
	num_bch_in_epc = 0
	for i, inpbch0 in enumerate(ldr):
		# generate pairs of context and a token as input tokens and a target, respectively.
		# context length is increases from 1 to the context_token_length
		num_eff_batch = 0
		num_nonebatch = 0
		lossbch = 0
		accbch = 0
#		accbch2 = 0
		for icntxtlen in range(context_token_length):
			cntxtlen = icntxtlen + 1
			inps, tgts = gen_batch(inpbch0, cntxtlen)
			if inps == None:
				#print('[', i, ']', 'None')
				num_nonebatch = num_nonebatch + 1
				continue
			#学習データinpsのシーケンス長はcntxtlenであり、xはset_1hotで自動的にinpsと同じシーケンス長に切り詰められる（コピーはしない）
			x = set_1hot(xInput, inps)
			#x = set_1hot(xInput, context_token_length, inps)
			#print('epc=', iEp, '[', i, '][', cntxtlen, ']inpbch0=', inpbch0.size(), 'inps=', inps.size(), 'tgts=', tgts.size(), 'x=', x.size())
			#for j, ii100 in enumerate(inpbch0):
			#	print('[', i, ']inpbch0[', j, ']=', ii100)
			#print('[', i, ']inps=', inps.size())
			#for j, ii100 in enumerate(inps):
			#	print('[', i, ']inps[', j, ']=', ii100)
			#print('[', i, ']tgts=', tgts.size(), tgts)
			#for j, tt100 in enumerate(tgts):
			#	print('[', i, ']tgts[', j, ']=', tt100)
			#print('[', i, ']x=', x.size(), x)
			#print()
			
			optim.zero_grad()
			# out = [batch][sequence][num_vocab]
			out = model(x)
			# out_latest_token = [batch][num_vocab]
			out_latest_token = out[:,-1,:]
			#print('out=', out.size(), 'out_latest_token', out_latest_token.size(), 'tgts=', tgts.size())
			loss = fLoss(out_latest_token, tgts)
			loss.backward()
			optim.step()
			#print('[', iEp, ']', 'loss', loss.item())
			
			# accumulate mean (among all the tgts) loss
			lossbch = lossbch + loss.item()

			# token probability vectors in the sequence
			# p = [batch][num_vocab]
			# softmax dim should be along [num_vocab] thus 1.
			p = F.softmax(out_latest_token, dim = 1)
			# torch.gather(p, 1, tgts.unsqueeze(1)); 1 means p[][select from this]; tgts.unsqueeze(1) means such as [[tkn_id1],[tkn_id2],...,[tkn_idN]]
			# torch.gather(p, 1, tgts.unsqueeze(1)) results in p[][1; probabilty of selected tkn_id],
			# and then sum() means the sum of probabilities at correct (target) token ids; Thus its maximum is len(tgts) = len(batch) = p.size(0).
			accbch = accbch + torch.gather(p, 1, tgts.unsqueeze(1)).sum().item()
			num_eff_batch = num_eff_batch + len(tgts)
			#for i100, idx in enumerate(tgts):
			#	accbch2 = accbch2 + p[i100][idx].item()
			#	#print('p=', p.size(), p[i100][idx])
			#break
		
		# lossbch should devided by context_token_length because each mean loss per loop are accumulated.
		lossbchavg = lossbch / context_token_length
		lossepc = lossepc + lossbchavg
		num_bch_in_epc = num_bch_in_epc + 1
		#print('epc=', iEp, '[', i, ']inpbch0=', inpbch0.size(), 'lossbchavg=', lossbchavg, 'num_eff_batch=', num_eff_batch)
		#break
		accbchavg = accbch / num_eff_batch
		accepc = accepc + accbchavg
		#print('accbch=', accbch, 'accbch2=', accbch2)
	
	lossepcavg = lossepc / num_bch_in_epc
	accepcavg = accepc / num_bch_in_epc
	wtr.add_scalar('Loss', lossepcavg, iEp)
	wtr.add_scalar('Accuracy', accepcavg, iEp)
	print('epc=', iEp, 'lossepcavg=', lossepcavg, 'accepcavg=', accepcavg, 'num_bch_in_epc=', bchsize, 'x', num_bch_in_epc)

wtr.close()

#############################################################################################
# save the model
#############################################################################################

print('vocabulary=', len(voc), 'dataset=', len(ds), 'epoch=', nepoch, ' DataLoader batch=', bchsize)

# measure execution time in seconds
tEnd = time.perf_counter()

print('------------------------------------')
print('execution time (s):', (tEnd - t0))

torch.save(model, modelfilename)

#wq2 = model.wq0[:,136]
#wk2 = model.wk0[:,136]
#wv2 = model.wv0[:,136]
#print('深鉢1', voc['深鉢'], wq1, wk1, wv1)
#print('深鉢2', voc['深鉢'], wq2, wk2, wv2)
#print('深鉢1-2', voc['深鉢'], wq1-wq2, wk1-wk2, wv1-wv2)





