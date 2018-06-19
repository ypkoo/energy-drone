import pandas as pd
import config
from pykalman import KalmanFilter
from math import sqrt

FLAGS = config.flags.FLAGS

def get_df(filename=None):

	if filename:
		df = pd.read_csv(FLAGS.f_dir+filename)
	else:
		df = pd.read_csv(FLAGS.f_dir+FLAGS.f_n)

	return df

def make_history(data, index_list, history_num):

		history = pd.DataFrame()

		for i in index_list:
			for n in range(history_num):
				data[i+'_shifted_by_'+str(n+1)] = data[i].shift(n+1)

		return data[history_num:]
		# data.drop(data.index[])
		# df.drop(df.index[[1,3]], inplace=True)
		# return history

def make_input_for_cnn(data, index_list, output_index, history_num):

	inputs = []

	for i in index_list:
		temp = []
		for n in range(data.shape[0] - history_num):
			# print (data[i][n:n+history_num-1].values.shape)
			temp.append(data[i][n:n+history_num].values)
		inputs.append(pd.DataFrame(temp))

	return inputs



def shift(data, index_list, shift_num):
	for i in index_list:
		data[i] = data[i+'_shifted_by_'+str(shift_num)].shift(shift_num)

	return data[shift_num:]

def get_current(data):
	data['cur'] = ((data['cur_raw'] / 1024.0)*5000 - 2500) / 100
	data['power'] = data['cur'] * data['vol']

	return data

def get_moving_average(data, index_list, window):

	for i in index_list:
		data[i] = data[i].rolling(window=window).mean()

	return data[window-1:]

def get_act_vel(data):

	data['act_v'] = (data['act_vx']**2 + data['act_vy']**2 + data['act_vz']**2)**.5

	return data

def get_vel(data):
	data['vel'] = (data['vel_x']**2 + data['vel_y']**2 + data['vel_z']**2)**.5

	return data

def get_dot_product(data):

	data['dot_p'] = data['act_vx'] * data['vel_x'] + data['act_vy'] * data['vel_y'] + data['act_vz'] * data['vel_z']

	return data

def get_act_acc(data):

	data['act_acc'] = ((data['act_vx']-data['vel_x'])**2 + (data['act_vy']-data['vel_y'])**2 + (data['act_vz']-data['vel_z'])**2)**.5
	# data['act_acc'] = ((data['act_vx']-data['vel_x'])**2 + (data['act_vy']-data['vel_y'])**2)**.5

	return data

def get_acc(data):

	data['acc'] = (data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)**.5

	return data

def concat(filename_prefix, file_num):
	df_list = []

	for i in range(file_num):
		df = pd.read_csv(filename_prefix + str(i+1) + '.csv')
		df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
		df_list.append(df)

	df_concat = pd.concat(df_list)

	return df_concat

def save_csv(df, postfix):
	fn = FLAGS.f_dir+FLAGS.f_n
	fn_split = fn.split('.')
	df.to_csv(fn_split[0]+'_'+postfix+'.'+fn_split[1], index=False, sep=",")

if __name__ == '__main__':
	x = pd.read_csv("0415233753_log_mod2.csv")

	ip = InputPreprocessor()

	x = ip.make_history(x, ['vel_x', 'vel_y'], 2)

	# x = x[1:]
	print (x)
	# print y