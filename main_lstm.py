#!/usr/bin/env python
#coding=utf8
"""
====================================
 :mod:`main` Main
====================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

설명
=====

This is for analyzing energy consumption of drone,
and propose the neural network model for drone's energy consumption

참고
====
 * https://github.com/rhoowd/energy_drone

관련 작업자
===========

본 모듈은 다음과 같은 사람들이 관여했습니다:
 * Daewoo Kim
 * Se-eun Yoon

 CUDA_VISIBLE_DEVICES=0

"""
import config
import log_result as log
import models
import graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from keras import backend as K

from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import time

import input_preprocess as ip

FLAGS = config.flags.FLAGS


if __name__ == '__main__':

	Tk().withdraw()
	filelist = askopenfilenames()

	# df = pd.read_csv(FLAGS.f_dir+FLAGS.f_n)

	# x_labels: 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy'
	x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
	y_label = ['power']

	x_data_list = []
	y_data_list = []

	sc = StandardScaler()

	for f in filelist:
		df = pd.read_csv(f)
		x_data = sc.fit_transform(df[x_labels])
		x_data = pd.DataFrame(data=x_data, columns=x_labels)
		df = x_data.join(df[y_label])

		x_data, y_data = ip.reshape_for_rnn(df, time_window=10)

		x_data_list.append(x_data)
		y_data_list.append(y_data)

	x_data = np.concatenate(x_data_list)
	y_data = np.concatenate(y_data_list)

	# input normalization
	# sc = StandardScaler()
	# x_data = sc.fit_transform(df[x_labels])
	# x_data = pd.DataFrame(data=x_data, columns=x_labels)
	# df = x_data.join(df[y_label])

	# x_data, y_data = ip.reshape_for_rnn(df)


	log.logger.info("x_shape: " + str(x_data.shape) + ", y_shape:" + str(y_data.shape))


	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=FLAGS.test_size, random_state=FLAGS.seed)

	# Create model
	model = models.lstm(input_shape=(x_data.shape[1], x_data.shape[2]))

	# Start training
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=FLAGS.n_e,
			  batch_size=FLAGS.b_s, verbose=FLAGS.verbose)

	# Evaluate the model
	scores = model.evaluate(x_test, y_test)
	log.logger.info("ACC(test):\t" + str(scores[2] * 100) + "%\t" + log.filename + " s" + str(FLAGS.seed) + "\t")
	log.logger.info("MSE(test):\t" + str(scores[1]) + "\t" + log.filename + " s"+ str(FLAGS.seed) + "\t")
	scores = model.evaluate(x_data, y_data)
	log.logger.info("ACC(all):\t" + str(scores[2] * 100) + "%\t" + log.filename + " s" + str(FLAGS.seed) + "\t")
	log.logger.info("MSE(all):\t" + str(scores[1]) + "\t" + log.filename + " s" + str(FLAGS.seed) + "\t")


	# save prediction result
	predictions = model.predict(x_test)
	y_test_t = y_test.reshape((-1, 1))
	result = np.concatenate((y_test_t,predictions),axis=1)
	now = time.localtime()
	s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
	result_file_name = "pred_result/"+str(FLAGS.depth)+ "-" + str(FLAGS.h_size)+"-"+str(FLAGS.b_s)+"-"+str(FLAGS.dropout_rate)+"-"+s_time+".csv"
	np.savetxt(result_file_name, result, delimiter=",")


	# Save model
	model_json = model.to_json()
	with open("result/model/"+log.filename+".json", "w") as json_file:
		json_file.write(model_json)  # serialize model to JSON
	model.save_weights("result/model/"+log.filename+".h5")  # weight
	print("Save model ... done")

	# Make graph
	if FLAGS.graph == 1:
		est = model.predict(x_data)
		graph.draw_graph(x_data[:, -1], y_data, est)
		print("Save graph ... done")

	K.clear_session()
