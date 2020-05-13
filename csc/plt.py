import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np 

# color: orange green teal 
# color: darkgoldenrod limegreen c

# marker: ^ p s h D H > < 

# linestyle: -- :
def sample(data):
	x_data = []
	y_data = []
	for i in range(9,100,10):
		x_data.append(data[i,0])
		y_data.append(data[i,1])
	return x_data, y_data


def main():
	fontsize = 15
	filename = ['res_'+str(i+1)+'.txt' for i in range(3)]
	val      = [0.578, 0.681, 0.690]
	for i,file in enumerate(filename):

		data = np.genfromtxt(file)


		x_data, y_data = sample(data)
		# x_data, y_data = data1[:,0], data1[:,1]

		# print(x_data)

		plt.plot(x_data, y_data, label = 'fix', color='c', ls = '-.', marker = 'o', markersize = 10)
		plt.plot(x_data, [val[i]]*len(x_data), label = 'automatic', ls = '--',color = 'green', marker = 'p', markersize = 10)

		x_major_locator=MultipleLocator(0.1)
		y_major_locator=MultipleLocator(0.02)

		ax = plt.gca()

		ax.xaxis.set_major_locator(x_major_locator)
		ax.yaxis.set_major_locator(y_major_locator)

		plt.xticks(fontsize = fontsize)
		plt.yticks(fontsize = fontsize)
		plt.xlabel(r'$\theta_s$', fontsize = fontsize)
		plt.ylabel("Accuracy", fontsize = fontsize)
		plt.legend(loc = 'lower center', fontsize = fontsize)
		minimum = np.min(y_data)-0.05
		maximum = np.max(y_data)+0.05
		plt.ylim(minimum, maximum)
		# plt.xlim(0.1,1)

		plt.show()



if __name__ == '__main__':
	main()