from input_preprocess import *

fn = FLAGS.f_dir+FLAGS.f_n
fn_split = fn.split('.')

df = get_df()
# df = get_act_acc(df)

# df = get_act_vel(df)

df = get_vel(df)

# df = get_dot_product(df)

df = get_acc(df)

df.to_csv(fn_split[0]+'_actv.'+fn_split[1], index=False, sep=",")