# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random as rd
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal

N=10
M=[0,1,3,9]

def create_dataset(num):
	dataset = DataFrame(columns=["x","y"])
	# print dataset

	for i in range(num):
		x = float(i)/float(num-1)
		y = np.sin(2*np.pi*x) + normal(scale=0.3)
		dataset = dataset.append(Series([x,y],index=['x','y']) ,ignore_index=True)



	return dataset
	# data = [rd.randint(-10,10) +60+12*i -14*i**2 + i**3 + i for i in range(0,size)]

# 平方根平均二乗誤差（Root mean square error）を計算
def rms_error(dataset, f):
	err = 0.0

	for index, line in dataset.iterrows():
		x, y = line.x, line.y
		err += 0.5 * (y - f(x))**2    # 誤差E

	return np.sqrt(2 * err / len(dataset))

# 最小二乗法で解を求める
def resolve(dataset, m):
	t = dataset.y
	phi = DataFrame()

	#公式Φ
	for i in range(0,m+1):
        	p = dataset.x**i   #get x data   ?? p.type
        	p.name="x**%d" % i
        	phi = pd.concat([phi,p], axis=1) # 右にデータを結合する。

	tmp = np.linalg.inv(np.dot(phi.T, phi)) #逆行列計算
	ws = np.dot(np.dot(tmp, phi.T), t)


	def f(x):
		y = 0

		#f(x) = sigma (w*x^M)
		#enumerate:  ループする際にインデックスつきで要素を得ることができる。
		for i, w in enumerate(ws):
			y += w * (x ** i)

		return y


	return (f, ws) # return function !!

# fx = w0 + w1*x + w2*x**2 + w3*x**3

# Main

if __name__ == '__main__':
	train_set = create_dataset(10)
	test_set = create_dataset(10)
	df_ws = DataFrame()     #空のDS

	# 多項式近似の曲線を求めて表示
	fig = plt.figure()

	#それぞれの次元の関数を作成
	for c, m in enumerate([1,4]):

		f, ws = resolve(train_set, m)

	#
	df_ws = df_ws.append(Series(ws,name="M=%d" % m))


	#グラフ作成 左上から順に
	subplot = fig.add_subplot(2,1,c+1)
	subplot.set_xlim(-0.05,1.05)
	subplot.set_ylim(-1.5,1.5)
	subplot.set_title("M=%d" % m)

	# トレーニングセットを表示
	subplot.scatter(train_set.x, train_set.y, marker='o', color='blue')



	# 多項式近似の曲線を表示
	linex = np.linspace(0,1,101)
	liney = f(linex)
	label = "E(RMS)=%.2f" % rms_error(train_set, f)
	subplot.plot(linex, liney, color='red', label=label)
	subplot.legend(loc=1)

	# 係数の値を表示
	print "Table of the coefficients"
	print df_ws.transpose()
	fig.show()

	# トレーニングセットとテストセットでの誤差の変化を表示
	df = DataFrame(columns=['Training set','Test set'])
	for m in range(0,10):   # 多項式の次数
		f, ws = resolve(train_set, m)
		train_error = rms_error(train_set, f)
		test_error = rms_error(test_set, f)
		df = df.append(
		Series([train_error, test_error],
			index=['Training set','Test set']),
			ignore_index=True)
	df.plot(title='RMS Error', style=['-','--'], grid=True, ylim=(0,0.9))
	plt.show()

