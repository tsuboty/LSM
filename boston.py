#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def main():

	# boston data sets
	boston = datasets.load_boston()

	# 部屋数
	rooms = boston.data[:,5]

	# 家の値段
	house_prices = boston.target

	# 部屋の数と家の値段の関係をプロットする。
	plt.scatter(rooms, house_prices, color='r')


	# 最小二乗法で誤差が最も少なくなる直線を得る
	# x = np.array([rooms],np.one(len(rooms))).T
	x = np.array([[v, 1] for v in rooms])  # バイアス項を追加する
	y = house_prices

	# print np.ones_like(rooms)

	# 最小二乗法で誤差が最も少なくなる直線を得る
	(slope,bias), total_error, _, _ = np.linalg.lstsq(x, y)

	# 得られた直線をプロットする
	plt.plot(x[:, 0], slope * x[:, 0] + bias)
	# plt.xlabel('部屋の数')
	# plt.ylabel('家の値段 (単位: 1000 ドル)')
	plt.grid()
	plt.xlabel('rooms')
	plt.ylabel('price')
	plt.show()


if __name__ == '__main__':
	main()