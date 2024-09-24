from torch_geometric.loader import DataLoader as DataLoader
import torch_geometric.transforms as T
import torch.optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ec_utils
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import time
import sys
import re

#print(len(sys.argv), sys.argv)
#exit(-1)

if len(sys.argv) != 3:
	print(f'Usage: python {sys.argv[0]} runname gpu_id\n\trunname= any string for output file base name. Do not use the same string with other runs.\n\tgpu_id = a GPU ID such as 0, 1, 2, or 3.')
	exit(-1)

print('-------------------------------------------------')
print('Jomon pottery point cloud model')
print('(c)2024 NUIS')
print('-------------------------------------------------')

###########################################################################
# hyper parameters
###########################################################################
#hyp_optm = {'name':'optimizer', 'params':['AdamW', 'Adam', 'SGD']}
hyp_optm = {'name':'optimizer', 'params':['AdamW', 'Adam']}
#hyp_lrnrate = {'name':'learning_rate', 'params':[1e-3, 1e-4, 1e-5]}
hyp_lrnrate = {'name':'learning_rate', 'params':[1e-5, 1e-6]}
hyp_btcsize = {'name':'batch_size', 'params':[2, 4, 8, 16, 32]}

###########################################################################
# number of trials
###########################################################################
num_trials = 10
#num_trials = len(hyp_optm['params']) * len(hyp_lrnrate['params']) * len(hyp_btcsize['params']) # for grid search

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
	exit(-1)
device = torch.device('cuda:' + str(gpuid)) if torch.cuda.is_available() else torch.devce('cpu')
#device = 'cpu'
print('device=', device)
#exit(-1)

###########################################################################
# Dataset source
###########################################################################
#dstdir = '/home/chika/db/torch_geometric_datasets/jomon1/rokutanda1_pc1024'
dstdir = '/home/nuis1/db/torch_geometric_datasets/jomon1/rokutanda1_pc1024'
print('dataset=', dstdir)

###########################################################################
# transform for data augmentation
###########################################################################
import torch_geometric.transforms
from scipy.spatial.transform import Rotation
# Random 3D rotation
class DataAug1(torch_geometric.transforms.BaseTransform):
	def __call__(self, data):
		rot = Rotation.random().as_matrix()
		#print(type(data.pos), data.pos.size())
		data.pos = torch.matmul(data.pos, torch.tensor(rot, dtype = torch.float).t())
		return data

trnsf_trn = DataAug1()

###########################################################################
# torch_geometric.data.Dataset
###########################################################################
dstrn = ec_utils.Jomon_Rokutanda_1024_Dataset1(dstdir, train = True, transform = trnsf_trn)
dstst = ec_utils.Jomon_Rokutanda_1024_Dataset1(dstdir, train = False, transform = trnsf_trn)
lentrn = len(dstrn)
lentst = len(dstst)
print('train=', len(dstrn), type(dstrn))
print('test=', len(dstst), type(dstst))
print('target_labels', dstrn.itos, dstrn.stoi)
#print(dstrn[0])
#print(dstrn[0]['pos'][:10])
#print(dstrn[0]['normal'][:10])
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


###########################################################################
# one model training and test routine: the objective function for hyper parameters optimization
###########################################################################
def train_test_optimize(trial):
	print('--- trial of training and test', trial.number)
	# suggenstion from optuna
	optm_label = trial.suggest_categorical(hyp_optm['name'], hyp_optm['params'])
	lrn_rate = trial.suggest_categorical(hyp_lrnrate['name'], hyp_lrnrate['params'])
	btcsize_from_hyp = trial.suggest_categorical(hyp_btcsize['name'], hyp_btcsize['params'])
	# a string for addition for discreminating different calls of train_test_optimize()
	run_str_add = f'try{trial.number}.bc{btcsize_from_hyp}.{optm_label}{lrn_rate}'
	print(run_str_add, 'btcsize(train)=', btcsize_from_hyp, 'optm=', optm_label, 'lr=', lrn_rate)

	#----------------------------------------------------------------------------------------------------------
	# measure execution time in seconds
	t0 = time.perf_counter()

	###########################################################################
	# input model file name
	# if None, training from scratch, else, training continues from the file.
	###########################################################################
	input_modelfilename = None
	#input_modelfilename = 'pn1.pth.ep30000_bc32'
	print('input modelfilename=', input_modelfilename)

	###########################################################################
	# number of epoch
	###########################################################################
	num_epoch = 300000

	###########################################################################
	# check point interval
	# the model will be saved every ckpt_interval of epochs
	# though the final result will also be saved regardless of ckpt_interval.
	###########################################################################
	ckpt_interval = 10000

	###########################################################################
	# train batch size
	###########################################################################
	btc_size_trn = btcsize_from_hyp
	#btc_size_trn = 4
	#btc_size_trn = 32

	###########################################################################
	# test batch size
	###########################################################################
	btc_size_tst = 10
	print('epoch=', num_epoch, 'batch(train)=', btc_size_trn, 'batch(test)=', btc_size_tst, 'ckpt interval=', ckpt_interval)

	###########################################################################
	# model
	###########################################################################
	if input_modelfilename == None:
	#	model = ec_utils.PointNetMini1(len(dstrn.itos))
		model = ec_utils.PointNetMini0(len(dstrn.itos))
	else:
		model = torch.load(input_modelfilename)

	modelclassname = model.__class__.__name__
	print(type(model), modelclassname)
	print(model)
	num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('parameters=', num_parameters, [n for n, p in model.named_parameters() if p.requires_grad])
	model = model.to(device)
	#exit(-1)
	#x = model(bt)
	#print(x.shape, x)

	###########################################################################
	# optimizer
	###########################################################################
	if optm_label == 'AdamW':
		optm = torch.optim.AdamW(lr = lrn_rate, params = model.parameters())
		#optm = torch.optim.AdamW(lr = 0.00001/2, params = model.parameters())
		#optm = torch.optim.AdamW(lr = 1e-5, params = model.parameters())
		#optm = torch.optim.AdamW(lr = 0.0001/100, params = model.parameters())
	elif optm_label == 'Adam':
		optm = torch.optim.Adam(lr = lrn_rate, params = model.parameters())
	elif optm_label == 'SGD':
		optm = torch.optim.SGD(lr = lrn_rate, params = model.parameters())
	else:
		print('error: suggested optimizer [', optm_label, '] is not implemented...')
		exit(-1)

	#sched = torch.optim.lr_scheduler.StepLR(optm, step_size = num_epoch // 10, gamma = 0.5)
	sched = None

	###########################################################################
	# loss function
	###########################################################################
	fLoss = nn.CrossEntropyLoss()
	#exit(-1)

	###########################################################################
	# tensorboard logdir and run base name
	###########################################################################
	tnsrbd_logdir_name = 'logpn'
	tnsrbd_run_dirname = f'{run_str_base}.{run_str_add}'

	cpath = Path.cwd()
	logdirbase = cpath / tnsrbd_logdir_name
	run1 = logdirbase / str('train.' + tnsrbd_run_dirname)
	run2 = logdirbase / str('test.' + tnsrbd_run_dirname)
	logdirbase.mkdir(exist_ok = True)
	run1.mkdir(exist_ok = True)
	run2.mkdir(exist_ok = True)
	wtr1 = SummaryWriter(log_dir = run1)
	wtr2 = SummaryWriter(log_dir = run2)
	print('log train=', run1)
	print('log test=', run2)
	#print(cpath, type(cpath))

	###########################################################################
	# torch_geometric.loader.DataLoader
	###########################################################################
	trnldr = DataLoader(dstrn, batch_size = btc_size_trn, shuffle = True)
	tstldr = DataLoader(dstst, batch_size = btc_size_tst, shuffle = True)
	#bt = next(iter(trnldr))
	#print('num_graphs=', bt.num_graphs, bt)

	#############################################################################################
	# output dir and ckpt file names
	#############################################################################################
	output_run_str = tnsrbd_run_dirname
	outputdirbaseOfBase = cpath / 'out.d'
	outputdirbaseOfBase.mkdir(exist_ok = True)
	outputdirbase = outputdirbaseOfBase / f'out.{output_run_str}'
	outputdirbase.mkdir(exist_ok = True)
	print('output dir=', outputdirbase)
	out_modelfilename = f'{modelclassname}.{output_run_str}.pth'

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

	#############################################################################################
	# number of output losses and accuracy
	#############################################################################################
	num_output_values = 2000
	epoch_per_outval = num_epoch / num_output_values
	print('num_output_values=', num_output_values, 'num_epoch=', num_epoch, 'epoch_per_outval=', epoch_per_outval)

	###########################################################################
	# training and test
	###########################################################################
	# recording for output values (tensorboard)
	losses = []
	lossestst = []
	# for calculating interval values
	lossavg_all = 0.0
	accravg_all = 0.0
	lossaccravg_cnt = 0
	losststavg_all = 0.0
	accrtstavg_all = 0.0
	lossaccrtstavg_cnt = 0
	# count for check point
	ickpt = 0
	# count for output values
	outval_next_epc = epoch_per_outval
	for epc in tqdm(range(num_epoch), total = num_epoch):
		######################	
		# save check point
		######################	
		if ickpt == ckpt_interval:
			torch.save(model, outputdirbase / str(out_modelfilename + '.ckpt' + str(epc)))
			ickpt = 0
		ickpt = ickpt + 1

		######################	
		# log output values or not
		######################	
		logoutval = False
		if epc + 1 >= outval_next_epc:
			logoutval = True
			outval_next_epc = outval_next_epc + epoch_per_outval
		elif epc == 0 or epc == num_epoch - 1:
			# always output the first and last epc
			logoutval = True

		######################	
		# training
		######################	
		model.train()

		lossall = 0
		accrall = 0
		batcnt = 0
		#for bat in tqdm(trnldr, total = len(trnldr)):
		for i, bat in enumerate(trnldr):
			#dbg_display_points(bat)

			# type(bat) = torch_geometric.data.batch.DataBatch
			# bat.batch = [number of points]. Each a value means the corresponding batch ID.
			# (e.g., batch IDs are 0, 1, ..., 7 if batch size is 8.)
			# bat.pos = [number of points][xyz = 3]
			# bat.y = [batch size]. Each value means a target label corresponding to one batch data.
			bat.to(device)
			#print(i, bat.batch.size(), bat.batch, bat.pos.size(), bat.pos, bat.y.size(), bat.y)
			optm.zero_grad()
			out1 = model(bat)
			loss = fLoss(out1, bat.y)
			loss.backward()
			optm.step()
			if sched != None:
				sched.step()

			# accumlate loss over all the batches
			lossall = lossall + loss.item()
			# predicted labels
			out2 = out1.argmax(dim = 1)
			# count correct predicted labels with target labels
			accr = int((out2 == bat.y).sum())
			# accumlate the above count over all the batches
			accrall = accrall + accr
			# count number of batches in an epoch
			batcnt = batcnt + 1

			#print(epc, i, 'train', len(bat), batcnt, loss.item(), accr, 'out=', out2, 'y=', bat.y)
			
		# average on one epoch per one Data
		# lossall should devided by batcnt because of accumulated mean per Data values
		lossavg = float(lossall) / batcnt
		# accrall should devided by the number of Data
		accravg = float(accrall) / lentrn

		# for calculating average on an output interval
		lossavg_all = lossavg_all + lossavg
		accravg_all = accravg_all + accravg
		lossaccravg_cnt = lossaccravg_cnt + 1
		#print('train', epc, lossavg, lossavg_all, accravg, accravg_all, lossaccravg_cnt)

		# output average losses and accuracies for the output interval
		if logoutval:
			lossavg_out = lossavg_all / lossaccravg_cnt
			accravg_out = accravg_all / lossaccravg_cnt
			losses.append({'epc': epc, 'lss': lossavg_out, 'acc': accravg_out})
			#print('out train', epc, lossavg_out, accravg_out)
			lossavg_all = 0.0
			accravg_all = 0.0
			lossaccravg_cnt = 0

		######################	
		# test
		######################	
		model.eval()
		with torch.no_grad():

			lossalltst = 0
			accralltst = 0
			batcnttst = 0
			#for bat in tqdm(tstldr, total = len(tstldr)):
			for i, bat in enumerate(tstldr):
				#dbg_display_points(bat)
				
				# type(bat) = torch_geometric.data.batch.DataBatch
				# bat.batch = [number of points]. Each a value means the corresponding batch ID.
				# (e.g., batch IDs are 0, 1, ..., 7 if batch size is 8.)
				# bat.pos = [number of points][xyz = 3]
				# bat.y = [batch size]. Each value means a target label corresponding to one batch data.
				bat.to(device)
				#print(i, bat.batch.size(), bat.batch, bat.pos.size(), bat.pos, bat.y.size(), bat.y)
				out1 = model(bat)
				loss = fLoss(out1, bat.y)

				# accumlate loss over all the batches
				lossalltst = lossalltst + loss.item()
				# predicted labels
				out2 = out1.argmax(dim = 1)
				# count correct predicted labels with target labels
				accr = int((out2 == bat.y).sum())
				# accumlate the above count over all the batches
				accralltst = accralltst + accr
				# count number of batches in an epoch
				batcnttst = batcnttst + 1

				#print(epc, i, 'test', len(bat), batcnttst, loss.item(), accr, 'out=', out2, 'y=', bat.y)
				
			# average on one epoch per one Data
			# lossalltst should devided by batcnttst because of accumulated mean per Data values
			lossavg = float(lossalltst) / batcnttst
			# accralltst should devided by the number of Data
			accravg = float(accralltst) / lentst

			# for calculating average on an output interval
			losststavg_all = losststavg_all + lossavg
			accrtstavg_all = accrtstavg_all + accravg
			lossaccrtstavg_cnt = lossaccrtstavg_cnt + 1
			#print('test', epc, lossavg, losststavg_all, accravg, accrtstavg_all, lossaccrtstavg_cnt)

			# output average losses and accuracies for the output interval
			if logoutval:
				lossavg_out = losststavg_all / lossaccrtstavg_cnt
				accravg_out = accrtstavg_all / lossaccrtstavg_cnt
				lossestst.append({'epc': epc, 'lss': lossavg_out, 'acc': accravg_out})
				#print('out test', epc, lossavg_out, accravg_out)
				losststavg_all = 0.0
				accrtstavg_all = 0.0
				lossaccrtstavg_cnt = 0


	###########################################################################
	# save model
	###########################################################################
	torch.save(model, outputdirbase / out_modelfilename)

	###########################################################################
	# log to Tensorboard
	###########################################################################
	for v in losses:
		wtr1.add_scalar('Loss', v['lss'], v['epc'])
		wtr1.add_scalar('Accuracy', v['acc'], v['epc'])
		print('train', v['epc'], v['lss'], v['acc'])
	for v in lossestst:
		wtr2.add_scalar('Loss', v['lss'], v['epc'])
		wtr2.add_scalar('Accuracy', v['acc'], v['epc'])
		print('test', v['epc'], v['lss'], v['acc'])
	wtr1.close()
	wtr2.close()

	###########################################################################
	# calculate an objective function score to optimize
	###########################################################################
	scoreall = 0
	# targets are last 100 values within num_output_values (2000) accuracies.
	scoretargets = lossestst[-100:]
	for v in scoretargets:
		scoreall = scoreall + v['acc']
	score = scoreall / len(scoretargets)
	print('score=', score, 'scoreall=', scoreall, 'len=', len(scoretargets))
	#print('score', score)

	# measure execution time in seconds
	tEnd = time.perf_counter()
	print('run', trial.number, 'tnsrbd_run_dirname=', tnsrbd_run_dirname)
	print('execution time (s):', (tEnd - t0))
	print()

	###########################################################################
	# return an objective function score
	###########################################################################
	return score



###########################################################################
# optimize hyper parameters in model trainings and tests
###########################################################################
import optuna

print('-------------------------------------------------')
print('hyper parameter optimization started...')
print('number of trials', num_trials) 
print('optimizer', hyp_optm['params'])
print('learning rate', hyp_lrnrate['params'])
print('batch size', hyp_btcsize['params'])
print('-------------------------------------------------')
print()

optunafile = 'sqlite:///optuna.db'
studyname = f'{run_str_base}'
try:
	# delete the study if exists
	optuna.delete_study(study_name = studyname, storage = optunafile)
	print('delete', studyname, 'from', optunafile)
except KeyError:
	pass
# create study
study = optuna.create_study(study_name = studyname, storage = optunafile, direction = 'maximize')
study.optimize(train_test_optimize, n_trials = num_trials)

###########################################################################
# the optimization results
###########################################################################
trial = study.best_trial
print('-------------------------------------------------')
print('hyper parameter optimization was finished...')
print('number of trials', num_trials) 
print('-------------------------------------------------')
print(f'value: {trial.value}')
print('params:')
for key, value in trial.params.items():
    print(f'{key}: {value}')



