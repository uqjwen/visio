import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorbrewer
from matplotlib import cm
def get_color(z):
	minimum = min(z)
	maximum = max(z)
	colors = colorbrewer.YlOrRd[9]
	colors = [color(c) for c in colors][::-1]
	ret = []
	for value in z:
		index = int((value - minimum)/(maximum-minimum)*(len(colors)-1))
		ret.append(colors[index])
	return ret



def color(value):
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


def get_lens():
	data=[]
	sents = [10,12,14,16,18]
	words = [10,13,16,19,22,25,28]
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
datasets=['imdb', 'yelp-2013', 'yelp-2014']
def main():
	#构造需要显示的值
	data = np.array(get_lens()[0]).ravel()

	X=np.arange(0, 5, step=1)#X轴的坐标
	Y=np.arange(0, 9, step=1)#Y轴的坐标
	#设置每一个（X，Y）坐标所对应的Z轴的值，在这边Z（X，Y）=X+Y
	Z=np.zeros(shape=(5, 9))
	for i in range(5):
	    for j in range(9):
	        # Z[i, j]=(i+j)/100.+0.5
	        Z[i,j] = data[(i*9+j)%len(data)]-0.54
	# colors = colorbrewer.PuOr[5]
	# colors = [color(c) for c in colors]
	# colors = colors*9
	print(Z, type(Z))

	xx, yy=np.meshgrid(X, Y)#网格化坐标
	X, Y=xx.ravel(), yy.ravel()#矩阵扁平化
	bottom=np.zeros_like(X)#设置柱状图的底端位值
	Z=Z.ravel()#扁平化矩阵

	colors = get_color(Z)

	width=height=0.5#每一个柱子的长和宽
	width=0.3
	height=0.5

	#绘图设置
	fig=plt.figure()
	# ax=fig.gca(projection='3d')#三维坐标轴
	ax = Axes3D(fig)

	ax.bar3d(X, Y, bottom, width, height, Z, color=colors, cmap=cm.coolwarm, edgecolors='black')#

	#坐标轴设置
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z(value)')
	# ax.set_zticklabels([0.5,0.6])
	# ax.set_zlim(2,10)
	plt.show()


def main2():

	#生成绘图数据
	N = 100
	x, y = np.mgrid[:100, :100]
	Z = np.cos(x*0.05+np.random.rand()) + np.sin(y*0.05+np.random.rand())+2*np.random.rand()-1

	# mask out the negative and positive values, respectively
	Zpos = np.ma.masked_less(Z, 0)   #小于零的部分
	Zneg = np.ma.masked_greater(Z, 0)  #大于零的部分

	fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

	pos = ax1.imshow(Zpos, cmap='Reds', interpolation='none')
	fig.colorbar(pos, ax=ax1)  #这里使用colorbar来制定需要画颜色棒的图的轴，以及对应的cmap，与pos对应

	neg = ax2.imshow(Zneg, cmap='Blues_r', interpolation='none')
	fig.colorbar(neg, ax=ax2)

	pos_neg_clipped = ax3.imshow(Z, cmap='jet', vmin=-2, vmax=2,interpolation='none')  #-2,2的区间
	fig.colorbar(pos_neg_clipped, ax=ax3)
	plt.show()	
if __name__ == '__main__':
	# main()
	main()
	# print(color((251,180,174)))
