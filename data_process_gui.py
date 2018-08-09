from tkinter import *
from tkinter.scrolledtext import ScrolledText
from input_preprocess import *
from tkinter.filedialog import askopenfilenames


function_dict = {
	# "get velocity": get_vel_xy,
	# "get action velocity": get_act_vel,
	# "get vel and acc": get_vel_and_acc_xy,
	# "get distance": get_distance,
	# "get accumulated power": get_accum_power,
	# "magic": get_accum_power_by_distance,
	"delete useless power": delete_useless_power,
	"delete useless power shift": delete_useless_power_shift,
	"concat": concat,
	"shift power": shift_power,
	"remove abnormal data": remove_abnormal_data,
	"shuffle power": shuffle_power,
	"make history": make_history,
	# "reshape_for_rnn": reshape_for_rnn,
	"trim_shift_concat": trim_shift_concat,

}

class DataPreProcessingWindow(Frame):

	# Define settings upon initialization. Here you can specify
	def __init__(self, master=None):
		
		Frame.__init__(self, master)   

		#reference to the master widget, which is the tk window                 
		self.master = master

		self.file_list = []

		#with that, we want to then run init_window, which doesn't yet exist
		self.init_window()

	#Creation of init_window
	def init_window(self):

		# changing the title of our master widget      
		self.master.title("GUI")

		self.function_frame = Frame(self.master)
		self.function_list_frame = Frame(self.function_frame)
		self.file_select_frame = Frame(self.master)
		self.file_list_frame = Frame(self.file_select_frame)

		self.status_bar_label = Label(self.master, text="")

		function_label = Label(self.function_frame, text="Functions")

		function_label.grid(row=0, column=0, sticky=N)
		self.function_list_frame.grid(row=1, column=0)

		i = 0
		self.selected_function_var = StringVar()
		for fn, f in function_dict.items():
			rb = Radiobutton(self.function_list_frame, indicatoron=0, text=fn, variable=self.selected_function_var, value=fn)
			rb.pack(anchor="w", fill=BOTH)

			i = i+1

		self.file_list_text = ScrolledText(self.file_select_frame, width=50)
		file_select_btn = Button(self.file_select_frame, text="File Select", width=15, command=self.file_select_btn_clicked)
		apply_btn = Button(self.file_select_frame, width=15, text="Apply", command=self.apply_btn_clicked)
		clear_btn = Button(self.file_select_frame, width=15, text="Clear", command=self.clear_btn_clicked)
		postfix_label = Label(self.file_select_frame, text="Postfix")
		self.postfix_text = Entry(self.file_select_frame)

		self.file_list_text.grid(row=0, column=0, columnspan=2)
		file_select_btn.grid(row=1, column=0, columnspan=2, sticky=E+W)
		apply_btn.grid(row=2, column=0, columnspan=2, sticky=E+W)
		clear_btn.grid(row=3, column=0, columnspan=2, sticky=E+W)
		postfix_label.grid(row=4, column=0)
		self.postfix_text.grid(row=4, column=1)

		self.result_text = ScrolledText(self.master)
		self.result_text.tag_configure("bold", font="Helvetica 12 bold")
		self.result_text.grid(row=1, column=0, columnspan=2)

		self.function_frame.grid(row=0, column=0, sticky=N)
		self.file_select_frame.grid(row=0, column=1)
		self.status_bar_label.grid(row=2, column=0, columnspan=2)


	def file_select_btn_clicked(self):
		self.file_list = askopenfilenames()

		for f in self.file_list:
			self.file_list_text.insert(END, f.split("/")[-1]+"\n")

	def apply_btn_clicked(self):

		func_key = self.selected_function_var.get()

		func = function_dict[func_key]

		self.status_bar_label.config(text="Processing...")

		if func_key == "concat":
			ret = func(self.file_list)
			postfix = self.postfix_text.get()
			save_csv(ret, filename=self.file_list[0], postfix=postfix)
		elif func_key == "trim_shift_concat":
			ret = func(self.file_list)
			postfix = self.postfix_text.get()
			save_csv(ret, filename=self.file_list[0], postfix=postfix)
		elif func_key == "make history":
			for file in self.file_list:
				df = get_df(file)
				ret = func(df, ['roll', 'pitch', 'yaw', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'act_vx', 'act_vy'], 3)

				postfix = self.postfix_text.get()
				save_csv(ret, filename=file, postfix=postfix)
		elif func_key == "reshape_for_rnn":
			for file in self.file_list:
				df = get_df(file)
				func(df)
		else:
			for file in self.file_list:
				df = get_df(file)
				ret = func(df)

				
				# if ret:
				if isinstance(ret, str):
					header = "{} {}\n".format(func_key, file.split("/")[-1])
					ret = ret + "\n\n"
					self.result_text.insert(END, header, "bold")
					self.result_text.insert(END, ret)
				else:
					postfix = self.postfix_text.get()
					save_csv(ret, filename=file, postfix=postfix)

		

		self.status_bar_label.config(text="Done!")

	def clear_btn_clicked(self):
		self.file_list = []
		self.file_list_text.delete(1.0, END)
		self.result_text.delete(1.0, END)
		self.postfix_text.delete(0, END)


if __name__ == "__main__":
	# root window created. Here, that would be the only window, but
	# you can later have windows within windows.
	root = Tk()

	# root.geometry("400x300")

	#creation of an instance
	app = DataPreProcessingWindow(root)

	#mainloop 
	root.mainloop() 