import tensorflow as tf

FIELD_NUM = 17
TIMESTAMP, VEL_X, VEL_Y, VEL_Z, ACC_X, ACC_Y, ACC_Z, ROLL, PITCH, YAW, RC0, RC1, RC2, RC3, VOL, CUR, POWER = range(17)

flags = tf.flags

# flags for setting
flags.DEFINE_integer("n_s", -1, "Number of sample (-1: read all data from file)")
flags.DEFINE_integer("n_h", 0, "Number of histroy we are interesed in from data")
flags.DEFINE_integer("n_e", 50, "Number of epoch for training")
flags.DEFINE_integer("b_s", 40, "Size of batch for each training epoch")
# flags.DEFINE_string("f_n", "0415233753_log_mod2.csv", "filename of data file")
flags.DEFINE_string("f_n", "exp_power_actv_history.csv", "filename of data file")
flags.DEFINE_string("f_dir", "data/", "data file directory")


# For flexible model
flags.DEFINE_integer("depth", 5, "Depth of network (>1)")
flags.DEFINE_integer("h_size", 150, "Hidden node size")
flags.DEFINE_float("dropout_rate", 0.00, "Dropout rate")

# For CNN-1D model
flags.DEFINE_integer("n_f", 7, "Number of Filter")
flags.DEFINE_integer("l_f", 3, "Length of filter")

# For data split for cross validation
flags.DEFINE_integer("seed", 1, "Random seed for split data set")
flags.DEFINE_float("test_size", 0.33, "Test data size")


flags.DEFINE_integer("verbose", 1, "Print during fit (1) or not (0)")
flags.DEFINE_integer("graph", 0, "Save graph (1) or not (0)")
