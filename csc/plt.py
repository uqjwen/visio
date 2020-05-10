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
	data1 = np.genfromtxt('res_1.txt')

	x_data, y_data = sample(data1)
	# x_data, y_data = data1[:,0], data1[:,1]

	print(x_data)

	plt.plot(x_data, y_data, label = 'fix', color='c', ls = '-.', marker = 'o', markersize = 8)
	plt.plot(x_data, [0.578]*len(x_data), label = 'automatic', ls = '--',color = 'green', marker = 'p', markersize = 8)

	x_major_locator=MultipleLocator(0.1)
	y_major_locator=MultipleLocator(0.02)

	ax = plt.gca()

	ax.xaxis.set_major_locator(x_major_locator)
	ax.yaxis.set_major_locator(y_major_locator)

	plt.xticks(fontsize = fontsize)
	plt.yticks(fontsize = fontsize)
	plt.xlabel(r'$\theta_s$', fontsize = fontsize)
	plt.ylabel("Accuracy", fontsize = fontsize)
	plt.legend(fontsize = fontsize)
	plt.ylim(0.4,0.6)
	# plt.xlim(0.1,1)

	plt.show()



if __name__ == '__main__':
	main()