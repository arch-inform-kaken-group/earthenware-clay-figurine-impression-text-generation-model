import time
import sys
import re
from pathlib import Path
from torch_geometric.loader import DataLoader as DataLoader
import torch.optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import ec_utils

#############################################################################################
# set_1hot: set 1-hot vectors from batch of sequences of token IDs.
# x = specify the xInput (or another tensor buffer)
# token_index_sequence = [batch][expanded batch][sequence of token IDs]
#############################################################################################unittest
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
# extract and expand input and target tensors for a leaning with a specified context length
# from a DataLoader batch, inp (2-rank tensor [batch][sequence IDs]).
#
# return:
# If the token sequence length of inp <= cntxt_len, return (None, None); else,
# return torch.Tensor[batch][expanded batch][cntxt_len] as training input tokens,
# and torch.Tensor[batch][expanded batch] for training target tokens.
#############################################################################################unittest
def expand_batch(inp, cntxt_len, td_max_batchsize, td_max_expansion_batchsize):
	dvc = inp.device
	#print('expand_batch(): dvc=', dvc)
	seqlen = inp.size(1)
	numbchPerSeq = seqlen - cntxt_len
	if numbchPerSeq <= 0:
		return None, None
	else:
		#print('seqlen=', seqlen, 'numbchPerSeq=', numbchPerSeq, 'cntxt_len=', cntxt_len, 'inp=', inp.size(), inp)
		#print('seqlen=', seqlen, 'numbchPerSeq=', numbchPerSeq, 'cntxt_len=', cntxt_len, 'inp=', inp.size())
		# extract inputs, [batch][expanded batch][cntxt_len], from inp[:, use of :]
		binp = torch.stack(tuple(inp[:, i:i + cntxt_len] for i in range(numbchPerSeq)), dim = 1)
		# extract target tokens, [batch][expanded batch], from inp[:, no use of :]
		# this results in [batch][expanded batch] because of no use of : in the dim = 1 axis of inp.
		btgt = torch.stack(tuple(inp[:, i + cntxt_len] for i in range(numbchPerSeq)), dim = 1)
		#print('expand_batch binp=', binp.size(), binp)
		#print('expand_batch btgt=', btgt.size(), btgt)
		# check buffer size
		batch_len = btgt.size(0)
		if batch_len > td_max_batchsize:
			print('ERROR: A generated batch data [', batch_len, '] exceeded td_max_batchsize,', td_max_batchsize, '. Change td_max_batchsize...')
			sys.exit(-1)
		expnd_batch_len = btgt.size(1)
		if expnd_batch_len > td_max_expansion_batchsize:
			print('ERROR: A generated batch data [', batch_len, '][', expnd_batch_len, '] exceeded td_max_expansion_batchsize,', td_max_expansion_batchsize, '. Change td_max_expansion_batchsize...')
			sys.exit(-1)
		#print('expand_batch btgt=', btgt.size(), 'binp=', binp.size())
		#print('expand_batch btgt=', btgt, 'binp=', binp)
		return binp.to(dvc), btgt.to(dvc)

###########################################################################
# generate a 2-rank tensor from two 1-rank tensors in which one has
# flatten token ids in a batch and another has corresponding lengths
###########################################################################unittest
def convertTokenIDsTensor(baty, battkns_len, pad_id):
	dvc = baty.device
	#print('convertTokenIDsTensor(): dvc=', dvc)

	# generate a [batch][token_ids with max length] tensor
	tkns_len_max = max(battkns_len).item()
	slice_idx = 0
	yys = []
	for len1 in battkns_len:
		yy = baty[slice_idx:slice_idx + len1]
		yy_padded = F.pad(yy, (0, tkns_len_max - len1), 'constant', pad_id)
		yys.append(yy_padded.unsqueeze(0))
		slice_idx = slice_idx + len1
	ret = torch.cat(yys, dim = 0).to(dvc)
	#print('convertTokenIDsTensor():input', baty.size(), baty, battkns_len.size(), battkns_len)
	#print('convertTokenIDsTensor():return', ret.size(), ret)
	return ret

###########################################################################
# display points for debug
###########################################################################
#import matplotlib.pyplot as plt
#fig100 = plt.figure()
#def dbg_display_points(bat):
#	for i in range(len(bat)):
#		#print(i, type(bat[i]), bat[i])
#		b = bat[i]
#		plt.gca().clear()
#		pn100 = b.pos.numpy()
#		mn100 = np.mean(pn100, axis = 0)
#		print(i, mn100)
#		ax100 = fig100.add_subplot(111, projection = '3d')
#		#print(type(fig100), type(ax100), mn100)
#		ax100.scatter(pn100[:, 0], pn100[:, 1], pn100[:, 2], c = 'b', marker = 'o')
#		plt.pause(0.1)


###########################################################################
# one model training per epoch:
# training if is_train = True else test
###########################################################################
def train_or_test_per_epoch(xbuff, model, optm, sched, fLoss, ldr, device, learn_log, logoutval, pad_id, epc, is_train, run_dict2):
	# context_token_length is supposed to be xbuff.size(3)
	context_token_length = xbuff.size(3)

	if is_train:
		######################
		# training
		######################
		model.train()
		gradcxt = torch.enable_grad()
	else:
		######################
		# test
		######################
		model.eval()
		gradcxt = torch.no_grad()

	with gradcxt:
		# accumulating loss in one epoch
		lossall = 0
		# accumulating accuracy in one epoch
		accrall = 0
		# counting number of mini-batches in one epoch
		batcnt = 0

		# loop for mini-bathces from Dataloader
		for i, bat in enumerate(ldr):
			# get bat, a mini-batch.
			bat.to(device)
			
			#dbg_display_points(bat)

			# type(bat) = torch_geometric.data.batch.DataBatch
			# bat.batch = [number of points]. Each a value means a corresponding batch ID.
			# (e.g., batch IDs are 0, 1, ..., 7 if batch size is 8, and then bat.batch is such as [0, 0, 0, 1, 1, 2, ...])
			# bat.pos = [number of points][xyz = 3]
			# bat.y = [flattened token Ids]. Those of batches in order.
			# bat.tkns_len = [a sequence of token lengths of batches]. Those of batches in order.
			#print('train', i, bat.batch.size(), bat.batch, bat.pos.size(), bat.pos, bat.y.size(), bat.y, bat.tkns_len.size(), bat.tkns_len)
			# generate a [batch][padded token_ids] tensor, baty. baty[i][j] is the j-th token id at i-th batch
			baty = convertTokenIDsTensor(bat.y, bat.tkns_len, pad_id)
			#print('%%%%%1', baty.device, 'device=', device)

			# generate pairs of context and a token as input tokens and a target, respectively.
			# context length is increases from 1 to the context_token_length
			num_eff_expbatch = 0
			num_none_expbatch = 0
			loss_expbch = 0
			acc_expbch = 0
	#		acc_expbch2 = 0
			for icntxtlen in range(context_token_length):
				cntxtlen = icntxtlen + 1
				# extract and expand batches from baty to training pairs of inputs, inps and targets, tgst.
				# input: baty = [batch][padded token_ids]
				# return: inps = [batch][expanded batch][token_ids; length of cntxtlen]
				# return: tgts = [batch][expanded batch]
				inps, tgts = expand_batch(baty, cntxtlen, run_dict2['td_max_batchsize'], run_dict2['td_max_expansion_batchsize'])
				if inps is None:
					#print('[', i, ']', 'None')
					num_none_expbatch = num_none_expbatch + 1
					continue
				#print('%%%%%2', inps.device, tgts.device)

				# The tensor buffer xInput is sliced (not copied) to x by set_1hot() from inps
				# xInput = [td_max_batchsize][td_max_expansion_batchsize][num_vocab][context_token_length (sequence axis)]
				# inps: [batch][expanded batch][token_ids; length of cntxtlen]
				# return: x = [batch][expanded batch][num_vocab][cntxtlen; 1 <= cntxtlen <= context_token_length]
				x = set_1hot(xbuff, inps)
				#print('epc=', epc, '[', i, '][', cntxtlen, ']baty=', baty.size(), 'inps=', inps.size(), 'tgts=', tgts.size(), 'x=', x.size())
				#for j, ii100 in enumerate(baty):
				#	print('[', i, ']baty[', j, ']=', ii100)
				#print('[', i, ']inps=', inps.size())
				#for j, ii100 in enumerate(inps):
				#	print('[', i, ']inps[', j, ']=', ii100)
				#print('[', i, ']tgts=', tgts.size(), tgts)
				#for j, tt100 in enumerate(tgts):
				#	print('[', i, ']tgts[', j, ']=', tt100)
				#print('[', i, ']x=', x.size(), x)
				#print()
				#print('%%%%%3', x.device, xInput.device)

				#if icntxtlen == 2:
				#	exit(-1)

				if is_train:
					optm.zero_grad()

				# bat.pos = [number of points][xyz = 3]
				# bat.batch = [number of points]. Each a value means a corresponding batch ID.
				# (e.g., batch IDs are 0, 1, ..., 7 if batch size is 8, and then bat.batch is such as [0, 0, 0, 1, 1, 2, ...])
				# x = [batch][expanded batch][num_vocab][sequence]
				# return: out = [batch][expanded batch][sequence][num_vocab]
				out = model(bat.pos, bat.batch, x)
				# out_latest_token = [batch][expanded batch][num_vocab]
				out_latest_token = out[:, :, -1, :]
				#print('out=', out.size(), 'out_latest_token', out_latest_token.size(), 'tgts=', tgts.size())
				#print('out=', out.size(), 'out_latest_token', out_latest_token.size(), 'tgts=', tgts.size(), tgts)
				#print('%%%%%4', out.device, out_latest_token.device)
				# nn.CrossEntropyLoss() is supposed to input [batch][num_vocab][expanded batch]
				# out_latest_token = [batch][expanded batch][num_vocab]
				# out_latest_token_t = [batch][num_vocab][expanded batch]
				# tgts = [batch][expanded batch]
				out_latest_token_t = out_latest_token.transpose(1, 2)
				loss = fLoss(out_latest_token_t, tgts)
				
				if is_train:
					loss.backward()
					optm.step()
					#print('[', iEp, ']', 'loss', loss.item())
					if sched is not None:
						sched.step()

				# accumulate mean (among all the tgts) loss
				loss_expbch = loss_expbch + loss.item()

				# token probability vectors in the sequence
				# p = [batch][expanded batch][num_vocab]
				# softmax dim should be along [num_vocab] thus 2.
				p = F.softmax(out_latest_token, dim = 2)
				# torch.gather(p, 2, tgts.unsqueeze(2)); 1st '2' means p[][][select from this]; tgts.unsqueeze(2) means
				# such as [[[tkn_id1_1],[tkn_id1_2]],[[tkn_id2_1],[tkn_id2_2]],...,[[tkn_idN_1],[tkn_idN_2]]]
				# torch.gather(p, 2, tgts.unsqueeze(2)) results in p[][][1; probabilty of selected tkn_id],
				# and then .squeeze() means p[][][1] -> p[][]
				# and then sum() means the sum of probabilities at correct (target) token ids in all [batch][expanded batch];
				# Thus its maximum is tgts.size(0) * tgts.size(1).
				acc_expbch = acc_expbch + torch.gather(p, 2, tgts.unsqueeze(2)).sum().item()
				# num_eff_expbatch should count [batch][expanded batch].
				num_eff_expbatch = num_eff_expbatch + tgts.size(0) * tgts.size(1)
				#print('p=', p.size())
				#print()
				#for i100, idx in enumerate(tgts):
				#	acc_expbch2 = acc_expbch2 + p[i100][idx].item()
				#	#print('p=', p.size(), p[i100][idx])
				#break

			# average loss over expanded batches (contexts)
			# loss_expbchavg should devided by context_token_length because each mean loss per loop are accumulated.
			loss_expbchavg = loss_expbch / context_token_length
			# accumlate loss over all the batches
			lossall = lossall + loss_expbchavg
			# number of mini-batch
			batcnt = batcnt + 1
			#print('epc=', iEp, '[', i, ']inpbch0=', inpbch0.size(), 'loss_expbchavg=', loss_expbchavg, 'num_eff_expbatch=', num_eff_expbatch)
			#break
			# average accuracy over expanded batches (contexts)
			acc_expbchavg = acc_expbch / num_eff_expbatch
			# accumlate accuracy over all the batches
			accrall = accrall + acc_expbchavg
			#print('acc_expbch=', acc_expbch, 'acc_expbch2=', acc_expbch2)

		# average on one epoch per one Data
		# lossall and accrall should devided by batcnt because simply accumulated one per loop
		lossavg = lossall / batcnt
		accravg = accrall / batcnt

		# for calculating average on an output interval
		lossavg_all = learn_log['lossavg_all']
		accravg_all = learn_log['accravg_all']
		lossaccravg_cnt = learn_log['lossaccravg_cnt']

		lossavg_all = lossavg_all + lossavg
		accravg_all = accravg_all + accravg
		lossaccravg_cnt = lossaccravg_cnt + 1
		#print('train', epc, lossavg, lossavg_all, accravg, accravg_all, lossaccravg_cnt)
		# output average losses and accuracies for the output interval
		if logoutval:
			lossavg_out = lossavg_all / lossaccravg_cnt
			accravg_out = accravg_all / lossaccravg_cnt
			learn_log['losses'].append({'epc': epc, 'lss': lossavg_out, 'acc': accravg_out})
			#print('out train', epc, lossavg_out, accravg_out)
			lossavg_all = 0.0
			accravg_all = 0.0
			lossaccravg_cnt = 0

		learn_log['lossavg_all'] = lossavg_all
		learn_log['accravg_all'] = accravg_all
		learn_log['lossaccravg_cnt'] = lossaccravg_cnt


	###########################################################################
	# return
	###########################################################################
	return

###########################################################################
# one model training and test routine:
###########################################################################
def train_test_run(xbuff, trial_id, grid_params, run_dict):
	device = run_dict['device']
	dstrn = run_dict['dstrn']
	dstst = run_dict['dstst']
	run_str_base = run_dict['run_str_base']
	voc = run_dict['voc']
	td_max_batchsize = run_dict['td_max_batchsize']
	td_max_expansion_batchsize = run_dict['td_max_expansion_batchsize']
	#aaa = run_dict['aaa']
	
	run_dict2 = {'td_max_batchsize':td_max_batchsize, 'td_max_expansion_batchsize':td_max_expansion_batchsize}

	# measure execution time in seconds
	t0 = time.perf_counter()
	print('--- trial of training and test', trial_id, grid_params)

	###########################################################################
	# set up this run
	###########################################################################
	# number of vocabulary is supposed to be xbuff.size(2)
	# context_token_length is supposed to be xbuff.size(3)
	# xbuff is supposed to be xInput = torch.zeros(td_max_batchsize, td_max_expansion_batchsize, num_vocab, context_token_length).to(device)
	num_vocab = xbuff.size(2)
	context_token_length = xbuff.size(3)

	# hyper parameters
	optm_label = grid_params['optm_label']
	lrn_rate = grid_params['lrn_rate']
	btcsize_from_hyp = grid_params['btcsize']

	# a string for addition for discreminating different calls of train_test_optimize()
	run_str_add = f'try{trial_id}.bc{btcsize_from_hyp}.{optm_label}{lrn_rate}'
	print(run_str_add, 'btcsize(train)=', btcsize_from_hyp, 'optm=', optm_label, 'lr=', lrn_rate)

	# number of epoch
	#num_epoch = 8000
	num_epoch = 0

	# check point interval
	# the model will be saved every ckpt_interval of epochs
	# though the final result will also be saved regardless of ckpt_interval.
	ckpt_interval = 10000

	# train batch size
	btc_size_trn = btcsize_from_hyp

	# test batch size
	btc_size_tst = 10
	print('epoch=', num_epoch, 'batch(train)=', btc_size_trn, 'batch(test)=', btc_size_tst, 'ckpt interval=', ckpt_interval)

	# embedding dimension PointNet
	pn_embeddim1 = 128
	pn_embeddim2 = 1024
	pn_embeddim3 = 512
	pn_embeddim4 = 256

	# embedding dimension Transformer decoder
	td_embeddim = 512

	# input model file name
	# if None, training from scratch, else, training continues from the file.
	input_modelfilename = None
	#input_modelfilename = 'pn1.pth.ep30000_bc32'
	print()
	print('input modelfilename=', input_modelfilename)

	# model
	if input_modelfilename is None:
		model = ec_utils.PointcloudcCaption1(num_vocab, td_embeddim, context_token_length,
			pn_embeddim1, pn_embeddim2, pn_embeddim3, pn_embeddim4)
	else:
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
	#exit(-1)
	#x = model(bt)
	#print(x.shape, x)

	# optimizer
	if optm_label == 'AdamW':
		optm = torch.optim.AdamW(lr = lrn_rate, params = model.parameters())
	elif optm_label == 'Adam':
		optm = torch.optim.Adam(lr = lrn_rate, params = model.parameters())
	elif optm_label == 'SGD':
		optm = torch.optim.SGD(lr = lrn_rate, params = model.parameters())
	else:
		print('error: suggested optimizer [', optm_label, '] is not implemented...')
		sys.exit(-1)
	#sched = torch.optim.lr_scheduler.StepLR(optm, step_size = num_epoch // 10, gamma = 0.5)
	sched = None

	# loss function
	fLoss = nn.CrossEntropyLoss()
	#exit(-1)

	# torch_geometric.loader.DataLoader
	trnldr = DataLoader(dstrn, batch_size = btc_size_trn, shuffle = True)
	tstldr = DataLoader(dstst, batch_size = btc_size_tst, shuffle = True)
	#bt = next(iter(trnldr))
	#print('num_graphs=', bt.num_graphs, bt)

	# tensorboard logdir and run base name
	tnsrbd_logdir_name = 'logpccap'
	tnsrbd_run_dirname = f'{run_str_base}.{run_str_add}'
	cpath = Path.cwd()
	#print(cpath, type(cpath))
	logdirbase = cpath / tnsrbd_logdir_name
	logdirbase.mkdir(exist_ok = True)

	# output dir and ckpt file names
	output_run_str = tnsrbd_run_dirname
	outputdirbaseOfBase = cpath / 'out.d'
	outputdirbaseOfBase.mkdir(exist_ok = True)
	outputdirbase = outputdirbaseOfBase / f'out.{output_run_str}'
	outputdirbase.mkdir(exist_ok = True)
	print('output dir=', outputdirbase)
	out_modelfilename = f'{modelclassname}.{output_run_str}.pth'

	# number of output losses and accuracy
	num_output_values = 2000
	epoch_per_outval = num_epoch / num_output_values
	print(' num_output_values=', num_output_values, 'num_epoch=', num_epoch, 'epoch_per_outval=', epoch_per_outval)

	# the index of special tokes in the Vocab
	pad = voc['<pad>']
#	unk = voc['<unk>']
#	bos = voc['<bos>']
#	eos = voc['<eos>']


	###########################################################################
	# training and test
	###########################################################################

	# learning logs such as losses and accuracies over one run
	# finally learn_logs will be written to tensorboard logdir
	learn_logs = {'train':{}, 'test':{}}
	for trn_tst, logdict in learn_logs.items():
		# for calculating interval values
		logdict['lossavg_all'] = 0.0
		logdict['accravg_all'] = 0.0
		logdict['lossaccravg_cnt'] = 0
		# recording for output values (tensorboard)
		logdict['losses'] = []
		tnsrbd_run_rundirpath = logdirbase / str(trn_tst + '.' + tnsrbd_run_dirname)
		tnsrbd_run_rundirpath.mkdir(exist_ok = True)
		logdict['wtr'] = SummaryWriter(log_dir = tnsrbd_run_rundirpath)
		print('log train', trn_tst, '=', tnsrbd_run_rundirpath)
	print()

	# count for check point
	ickpt = 0
	# count for output values
	outval_next_epc = epoch_per_outval

	# traing and test num_epoch times
	for epc in tqdm(range(num_epoch), total = num_epoch):
		# save check point
		if ickpt == ckpt_interval:
			torch.save(model, outputdirbase / str(out_modelfilename + '.ckpt' + str(epc)))
			ickpt = 0
		ickpt = ickpt + 1

		# log output values or not
		logoutval = False
		if epc + 1 >= outval_next_epc:
			logoutval = True
			outval_next_epc = outval_next_epc + epoch_per_outval
		elif epc == 0 or epc == num_epoch - 1:
			# always output the first and last epc
			logoutval = True

		# training
		# def train_or_test_per_epoch(xbuff, model, optm, sched, fLoss, ldr, device, learn_log, logoutval, pad_id, epc, is_train):
		is_train = True
		train_or_test_per_epoch(xbuff, model, optm, sched, fLoss, trnldr, device, learn_logs['train'], logoutval, pad, epc, is_train, run_dict2)

		# test
		is_train = False
		train_or_test_per_epoch(xbuff, model, optm, sched, fLoss, tstldr, device, learn_logs['test'], logoutval, pad, epc, is_train, run_dict2)


	# save model
	torch.save(model, outputdirbase / out_modelfilename)

	# log to Tensorboard
	for trn_tst, logdict in learn_logs.items():
		wtr = logdict['wtr']
		for v in logdict['losses']:
			wtr.add_scalar('Loss', v['lss'], v['epc'])
			wtr.add_scalar('Accuracy', v['acc'], v['epc'])
			print(trn_tst, v['epc'], v['lss'], v['acc'])
		wtr.close()

	# measure execution time in seconds
	tEnd = time.perf_counter()
	print('run', trial_id, 'tnsrbd_run_dirname=', tnsrbd_run_dirname)
	print('execution time (s):', (tEnd - t0))
	print()

	###########################################################################
	# return
	###########################################################################
	return

###########################################################################
# The main routine
###########################################################################
def main():
	#print(len(sys.argv), sys.argv)
	#exit(-1)

	if len(sys.argv) != 3:
		print(f'Usage: python {sys.argv[0]} runname gpu_id\n\trunname= any string for output file base name. Do not use the same string with other runs.\n\tgpu_id = a GPU ID such as 0, 1, 2, or 3.')
		sys.exit(-1)

	print('-------------------------------------------------')
	print('Prototype of Jomon pottery point cloud-Japanese')
	print('language captioning model')
	print('(c)2024 NUIS')
	print('-------------------------------------------------')

	###########################################################################
	# hyper parameters
	###########################################################################
#	hyp_optm = {'name':'optimizer', 'params':['AdamW', 'Adam', 'SGD']}
#	hyp_optm = {'name':'optimizer', 'params':['Adam', 'SGD']}
	#hyp_optm = {'name':'optimizer', 'params':['AdamW', 'Adam']}
	hyp_optm = {'name':'optimizer', 'params':['Adam']}
	#hyp_lrnrate = {'name':'learning_rate', 'params':[1e-3, 1e-4]}
	#hyp_lrnrate = {'name':'learning_rate', 'params':[1e-5, 1e-6]}
	hyp_lrnrate = {'name':'learning_rate', 'params':[1e-5]}
#	hyp_btcsize = {'name':'batch_size', 'params':[2, 4, 8, 16, 32]}
	hyp_btcsize = {'name':'batch_size', 'params':[4]}

	###########################################################################
	# run string
	# several output files and tensorboard logs are named based on this string
	###########################################################################
	# replace special characters to _ for a file system
	run_str_base = re.sub(r'[\<\>\:\"\/\\\|\?\*]', '_', sys.argv[1])
	print('Output files and tensorboard logs are named based on: ' + run_str_base)
	#exit(-1)

	###########################################################################
	# device
	###########################################################################
	gpuid = 0
	try:
		gpuid = int(sys.argv[2])
	except ValueError:
		print('error: gpu_id must be int >= 0...')
		sys.exit(-1)
	device = torch.device('cuda:' + str(gpuid)) if torch.cuda.is_available() else torch.devce('cpu')
	#device = 'cpu'
	#print('device=', device)
	#exit(-1)

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
	#exit(-1)

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
	# Context length for input tokens of the model
	#############################################################################################
	context_token_length = 6

	#############################################################################################
	# xInput: a tensor buffer for Transformer decoder inputs
	#############################################################################################
	td_max_batchsize = 32
	td_max_expansion_batchsize = 1024
	# xInput = tensor buffer for 1-hot vector sequence
	# supposing [td_max_batchsize][td_max_expansion_batchsize][num_vocab][context_token_length (sequence axis)] tensor for xInput
	# this tensor is repeatedly initialized to zeros before reuse.
	xInput = torch.zeros(td_max_batchsize, td_max_expansion_batchsize, num_vocab, context_token_length).to(device)
	print('xInput', xInput.size())

	# a dict for the main
	run_dict = {
		'device':device, 'dstrn':dstrn, 'dstst':dstst, 'run_str_base':run_str_base, 'voc':voc,
		'td_max_batchsize':td_max_batchsize, 'td_max_expansion_batchsize':td_max_expansion_batchsize,
	}

	###########################################################################
	# grid search hyper parameters in model trainings and tests
	###########################################################################
	num_trials = len(hyp_optm['params']) * len(hyp_lrnrate['params']) * len(hyp_btcsize['params'])
	print('-------------------------------------------------')
	print('grid search for hyper parameters started...')
	print('number of trials', num_trials) 
	print('optimizer', hyp_optm['params'])
	print('learning rate', hyp_lrnrate['params'])
	print('batch size', hyp_btcsize['params'])
	print('-------------------------------------------------')
	print()

	# grid search hyper parameters
	trial_id = 0
	for i, btcsize in enumerate(hyp_btcsize['params']):
		for j, optmzr in enumerate(hyp_optm['params']):
			for k, lrnrt in enumerate(hyp_lrnrate['params']):
				grid_params = {}
				grid_params['btcsize'] = btcsize
				grid_params['optm_label'] = optmzr
				grid_params['lrn_rate'] = lrnrt
				#print(trial_id, grid_params)
				train_test_run(xInput, trial_id, grid_params, run_dict)
				trial_id = trial_id + 1

	###########################################################################
	# the optimization results
	###########################################################################
	print('-------------------------------------------------')
	print('grid search was finished...')
	print('number of trials', num_trials) 
	print('-------------------------------------------------')

###########################################################################
# main() will be called from a command line only
###########################################################################
if __name__ == '__main__':
	main()







