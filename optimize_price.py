# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv
import pandas as pd
from saishoujijou import *

def readFile():
	file_name = "./2016-03-14_price.csv"
	data = pd.read_csv(file_name)

	return data


#最小二乗法をライブラリを使って表現
if __name__ == '__main__':
	train_set = readFile()
	df_ws = DataFrame()     #空のDS 係数Wが入る


	f ,ws = resolve(train_set,[4])



