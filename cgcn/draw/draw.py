import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorbrewer
from matplotlib import cm

datasets=['imdb', 'yelp-2013', 'yelp-2014']
def get_color(z):
	minimum = min(z)
	maximum = max(z)
	colors = colorbrewer.YlOrRd[9]
	colors = [t_color(c) for c in colors][::-1]
	ret = []
	for value in z:
		index = int((value - minimum)/(maximum-minimum)*(len(colors)-1))
		ret.append(colors[index])
	return ret



def t_color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a1, a2, a3)


def get_up_size():
	data=[]
	err_data = []
	for dataset in datasets:
		sub_data = []
		err_sub_data = []
		filenames = ['./ckpt/{}/clf/up_size_{}.txt'.format(dataset, i) for i in [50,100,150,200,250,300]]
		for i, filename in enumerate(filenames):
			file_data = np.genfromtxt(filename)
			# print(file_data, filename)
			run_data  = sorted(file_data[:,1])[-1:] if dataset=='yelp-2013' else sorted(file_data[:,1])[-20:-10] if (dataset=='imdb') else sorted(file_data[:,1])[-5:]
			# run_data  = (file_data[:,1])[-10:]
			# sub_data.append(np.mean(sorted(file_data[:,1])[-10:]))
			# print(run_data)

			sub_data.append(np.mean(run_data))
			err_sub_data.append(np.std(run_data))
		if dataset == 'yelp-2013':
			print(sub_data)
		data.append(sub_data)
		err_data.append(err_sub_data)
	# print(data)
	return data, err_data
def draw_up_size():

	data,err_data = get_up_size()
	colors = colorbrewer.Dark2[3]
	colors = [t_color(color) for color in colors]
	# xticks = ['{}'.format(i) for i in range(1,5)]
	xticks = [50,100,150,200,250,300]
	marks  = ['o', 's', 'D']
	fontsize = 20
	for i,(sub_data, err_sub_data) in enumerate(zip(data, err_data)):
		plt.plot(xticks, sub_data, label=datasets[i], color=colors[i], ls='-.', marker=marks[i], markersize=10)
		# print(sub_data)
		# plt.errorbar(xticks, sub_data, yerr=err_sub_data,label=datasets[i], ls='-.', marker=marks[i], color=colors[i], ms=6, elinewidth=3, capsize=4, capthick=2) #



	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel('#latent_factors', fontsize=fontsize)
	plt.ylabel("Accuracy", fontsize=fontsize)
	plt.legend(fontsize=fontsize-5)
	plt.grid(linestyle='-.')
	minimum = np.min(data)-0.01
	maximum = np.max(data)+0.02
	# plt.ylim(minimum, maximum)
	print(minimum, maximum)
	plt.show()



def get_lstm_size():
	data=[]
	err_data = []
	for dataset in datasets:
		sub_data = []
		err_sub_data = []
		filenames = ['./ckpt/{}/clf/lstm_size_{}.txt'.format(dataset, i) for i in [25, 50,100,150,200]]
		for i, filename in enumerate(filenames):
			file_data = np.genfromtxt(filename)
			# print(file_data, filename)
			# run_data  = sorted(file_data[:,1])[-1:] if dataset=='yelp-2013' else sorted(file_data[:,1])[-20:-10] if (dataset=='imdb') else sorted(file_data[:,1])[-5:]
			run_data  = sorted(file_data[:,1])[-5:] if dataset=='yelp-2013' else sorted(file_data[:,1])[-15:]
			# sub_data.append(np.mean(sorted(file_data[:,1])[-10:]))
			# print(run_data)

			sub_data.append(np.mean(run_data))
			err_sub_data.append(np.std(run_data))
		if dataset == 'yelp-2013':
			print(sub_data)
		data.append(sub_data)
		err_data.append(err_sub_data)
	# print(data)
	return data, err_data
def draw_lstm_size():

	data,err_data = get_lstm_size()
	colors = colorbrewer.Dark2[3]
	colors = [t_color(color) for color in colors]
	# xticks = ['{}'.format(i) for i in range(1,5)]
	xticks = [25,50,100,150,200]
	marks  = ['o', 's', 'D']
	fontsize = 20
	for i,(sub_data, err_sub_data) in enumerate(zip(data, err_data)):
		plt.plot(xticks, sub_data, label=datasets[i], color=colors[i], ls='-.', marker=marks[i], markersize=10)
		# print(sub_data)
		# plt.errorbar(xticks, sub_data, yerr=err_sub_data,label=datasets[i], ls='-.', marker=marks[i], color=colors[i], ms=6, elinewidth=3, capsize=4, capthick=2) #



	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel('lstm_size', fontsize=fontsize)
	plt.ylabel("Accuracy", fontsize=fontsize)
	plt.legend(fontsize=fontsize-5)
	plt.grid(linestyle='-.')
	minimum = np.min(data)-0.01
	maximum = np.max(data)+0.02
	# plt.ylim(minimum, maximum)
	print(minimum, maximum)
	plt.show()






def get_lens():
	data=[]
	sents = [10,12,14,16,18]
	# words = [10,13,16,19,22,25,28]
	words = [10,16,22,28]
	for dataset in datasets:
		sub_data = []
		filenames = ['./ckpt/{}/clf/lstm_size_{}.txt'.format(dataset, i) for i in [25, 50,100,150,200]]

		sub_data = []

		for sent in sents:
			sent_data = []
			for word in words:
				filename = './ckpt/{}/clf/lens_{}_{}.txt'.format(dataset, word, sent)
				file_data = np.genfromtxt(filename)
				run_data = np.mean(sorted(file_data[:,1])[-5:])
				sent_data.append(run_data)
			sub_data.append(sent_data)
		data.append(sub_data)
	return data

def draw_lens():

	data   = get_lens()
	colors = colorbrewer.Dark2[3]
	colors = [t_color(color) for color in colors]
	# xticks = ['{}'.format(i) for i in range(1,5)]
	# xticks = [25,50,100,150,200]
	sents  = [10,12,14,16,18]
	# words  = [10,13,16,19,22,25,28]
	words  = [10,16,22,28]
	
	marks  = ['o', 's', 'D']
	fontsize = 20

	for i,sub_data in enumerate(data):
		sub_data = np.array(sub_data)
		xx,yy  = np.meshgrid(sents, words)
		x,y,z  = xx.ravel(), yy.ravel(), sub_data.ravel()
		bottom = np.zeros_like(x)

		# print(z)
		colors = get_color(z)

		# width  = height=0.8
		width = 1
		height=2
		fig    = plt.figure()
		ax     = fig.gca(projection='3d')
		base   = round(min(z),3)-0.005
		ax.bar3d(x,y,bottom, width, height, z-base, color = colors, cmap=cm.coolwarm, edgecolor='black')


		# plt.yticks(fontsize=fontsize)
		# ax.set_yticklabels(map(str, words))
		ax.set_yticks(words)
		ax.set_xlabel('#sentence', fontsize=fontsize-4)
		ax.set_ylabel("#word", fontsize=fontsize-4)
		ax.set_zlabel("Accuracy", fontsize=fontsize-4)
		# plt.legend(fontsize=fontsize-5)
		# plt.ylim(minimum, maximum)
		# zticks = np.array([0.005, 0.010, 0.015])
		count = 3 if i == 0 else 5
		zticks = (np.arange(count)+1)*0.01
		# zticks = np.array([0.01,0.02, 0.03, 0.04, 0.05])
		zlabels = np.round(zticks+base, 3)
		ax.set_zticks(zticks)
		ax.set_zticklabels(zlabels, fontsize=fontsize-7)
		# ax.set_yticklabels(words, fontsize=fontsize-5)
		plt.yticks(fontsize=fontsize-5)
		plt.xticks(fontsize=fontsize-5)
		# ax.set_zticks([0.00,0.55,0.56])

		# ax.set_zlim(minimum, maximum)
		# ax.set_zlim(0.5,0.6)
		# plt.grid(linestyle='-.')

		plt.show()



	# print(data)
def get_fc_layers():
	data=[]
	err_data = []
	for dataset in datasets:
		sub_data = []
		err_sub_data = []
		filenames = ['./ckpt/{}/clf/fc_layers_{}.txt'.format(dataset, i) for i in range(1,5)]
		for i,filename in enumerate(filenames):
			file_data = np.genfromtxt(filename)
			# print(file_data, filename)
			# run_data  = sorted(file_data[:,1])[-10:]
			# num = 8 if i == 0 else 30 if i==2 else 15 
			# num = 30 if dataset == 'yelp-2014' else 15
			num = 10 if dataset == 'yelp-2013' else 25 if dataset == 'yelp-2014' else 30 
			run_data  = sorted(file_data[:,1])[-num:]
			# sub_data.append(np.mean(sorted(file_data[:,1])[-10:]))
			# print(run_data)
			sub_data.append(np.mean(run_data))
			# err_sub_data.append(np.std(run_data))
		# print(sub_data)
		data.append(sub_data)
		# err_data.append(err_sub_data)
	# print(data)
	return data

def draw_fc_layers():
	data = get_fc_layers()
	colors = colorbrewer.Dark2[3]
	colors = [t_color(color) for color in colors]
	xticks = ['{}'.format(i) for i in range(1,5)]
	marks  = ['o', 's', 'D']
	fontsize = 20
	for i, sub_data in enumerate(data):
		plt.plot(xticks, sub_data, label=datasets[i], color=colors[i], ls='-.', marker=marks[i], markersize=10)
		# print(sub_data)
		# plt.errorbar(xticks, sub_data, yerr=err_sub_data,label=datasets[i], ls='-.', marker=marks[i], color=colors[i], ms=6, elinewidth=3, capsize=4, capthick=2) #


	print(data)
	np.savetxt('layers.txt', np.array(data), fmt="%.3f", delimiter='\t')
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel('#layers', fontsize=fontsize)
	plt.ylabel("Accuracy", fontsize=fontsize)
	plt.legend(fontsize=fontsize-5)
	plt.grid(linestyle='-.')
	minimum = np.min(data)-0.01
	maximum = np.max(data)+0.02
	# plt.ylim(minimum, maximum)
	# print(minimum, maximum)
	# plt.show()



def get_alpha():
	data = []
	err_data = []
	for dataset in datasets:
		sub_data = []
		err_sub_data = []
		filenames = ['./ckpt/{}/clf/alpha_0.{}.txt'.format(dataset, i) for i in range(1,10)]
		for i,filename in enumerate(filenames):
			file_data = np.genfromtxt(filename)
			# run_data = file_data[:,1][-10:] if dataset == 'yelp-2014' and i==8 else sorted(file_data[:,1])[-10:]
			run_data = file_data[:,1][-10:] if dataset == 'yelp-2014' and i==7 else sorted(file_data[:,1])[-5:]
			if dataset == 'yelp-2014':
				print(np.mean(run_data), i)
			sub_data.append(np.mean(run_data))
			err_sub_data.append(np.std(run_data))

		data.append(sub_data)
		err_data.append(err_sub_data)
	return data, err_data

def draw_alpha():

	data, err_data = get_alpha()


	colors = colorbrewer.Dark2[3]

	colors = [t_color(color) for color in colors]
	xticks = ['0.{}'.format(i) for i in range(1,10)]
	marks  = ['o', 's', 'D']
	fontsize=20
	assert len(xticks) == len(data[0])

	for i,(sub_data, err_sub_data) in enumerate(zip(data, err_data)):
		plt.plot(xticks, sub_data, label=datasets[i], color=colors[i], ls='-.', marker=marks[i], markersize=10)
		# plt.errorbar(xticks, sub_data, yerr=err_sub_data,label=datasets[i], ls='-.', marker=marks[i], color=colors[i], ms=6, elinewidth=3, capsize=4, capthick=2) #


	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel(r'$\alpha$', fontsize=fontsize)
	plt.ylabel("Accuracy", fontsize=fontsize)
	plt.legend(fontsize=fontsize-5)
	plt.grid(linestyle='-.')
	minimum = np.min(data)-0.01
	maximum = np.max(data)+0.02
	plt.ylim(minimum, maximum)
	plt.show()

def get_curve():
	data = []
	for dataset in datasets:
		sub_data = []
		filename = './ckpt/{}/clf/base.txt'.format(dataset)
		sub_data = np.genfromtxt(filename)
		data.append(sub_data[:, :2])
	return data


def draw_curve():
	data = get_curve()
	colors = colorbrewer.Dark2[3]

	colors = [t_color(color) for color in colors]
	marks  = ['o', 's', 'D']
	fontsize = 20
	for i,sub_data in enumerate(data):
		# xticks = np.arange(len(sub_data))
		step = 100 if i==2 else 30
		xticks = [step*i for i in range(len(sub_data))]


		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.plot(xticks, sub_data[:,0], label='Train_loss', color=colors[0], marker=marks[0], markersize=6)
		# ax1.plot(xticks, sub_data[:,0], label='Train_loss', color=colors[0], linewidth=3)
		# loc = 
		ax1.set_ylabel("Training loss", fontsize=fontsize)
		ax1.legend(fontsize=fontsize-7, loc=[0.6,0.3])
		ax1.grid(linestyle='-.')
		
		plt.yticks(fontsize=fontsize)
		plt.xticks(fontsize=fontsize-7)
		plt.xlabel("steps", fontsize=fontsize)


		ax2 = ax1.twinx()
		ax2.plot(xticks, sub_data[:,1], label='Accuracy', color=colors[1], marker=marks[1], markersize=6)
		# ax2.plot(xticks, sub_data[:,1], label='Train_loss', color=colors[1], linewidth=3)
		ax2.set_ylabel("Accuracy", fontsize=fontsize)
		# plt.plot(xticks, sub_data[:,0], label='loss', color=colors[0], marker=marks[0], markersize=4)
		plt.yticks(fontsize=fontsize)
		# plt.xticks([100,1000,2000],['a','b','c'])
		ax2.legend(fontsize=fontsize-7, loc=[0.6,0.6])


		plt.show()
		# break

if __name__ == '__main__':
	# draw_alpha()
	# draw_up_size()
	# draw_lstm_size()
	# draw_lens()

	# draw_curve()
	draw_fc_layers()