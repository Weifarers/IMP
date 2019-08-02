from tkinter import filedialog
from tkinter import ttk
import tkinter as tk
from IMP import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class GUI:
    def __init__(self, master):
        # Creates the title and size of the window.
        self.master = master
        master.title('Iterative Matrix Pencil Tool')
        master.geometry('950x600')
        self.tab(master)

    def tab(self, master):
        # Enables usage of tabs in the GUI.
        tab_control = ttk.Notebook(master)
        # Initializes the tabs, and defines the window names for each tab.
        # Main Tab, where users input files and choose options, along with seeing the solution.
        main_tab = ttk.Frame(tab_control)
        tab_control.add(main_tab, text='Main Page')
        self.main_window(main_tab)
        tab_control.pack(expand=1, fill='both')

    def main_window(self, tab):
        # Sets the IMP options to global variables.
        global sig, svd_entry, iter_entry
        # Sets the timer variables to global variables.
        global pre_process_time, iter_choices, mp_time, \
            mode_shape_time, signal_recon_time, cf_time, total_time, def_iter
        # Sets the cost function statistics variables to global variables.
        global cf_max, cf_min, cf_avg
        # Sets the list of signals and the signal plots to global variables, for access while plotting.
        global sig_choices, def_sig, sig_plot, sig_canvas
        # Sets the plot and canvas of the modes to global variables for access while plotting.
        global mode_plot, mode_canvas, mode_val
        # Creates a display for users to know what file they are using.
        sig_frame = tk.Frame(tab)
        signal_name = tk.StringVar()
        # Creates the empty box for which the file name will show up.
        sig = tk.Entry(sig_frame, relief=tk.GROOVE, textvariable=signal_name, width=30)
        sig.bind("<Key>", lambda e: "break")
        sig.grid(column=0, row=0)
        # Gets the "Open File" button to prompt users to find their file.
        open_btn = tk.Button(sig_frame, text='Open File', command=openfile)
        open_btn.grid(column=1, row=0)
        # Builds the frame.
        sig_frame.grid(column=0, row=0)

        # Creates the title of the IMP Method Options.
        iter_opt = tk.Label(tab, text='Iterative Matrix Pencil Options', font='Helvetica 10 bold')
        iter_opt.grid(column=0, row=1)

        # Creates the frame for users to input a SVD threshold.
        svd_frame = tk.Frame(tab)
        # Creates the SVD Threshold label.
        svd_lbl = tk.Label(svd_frame, text='SVD Threshold:')
        svd_lbl.grid(column=0, row=0)
        # Defaults the SVD threshold to 0.025.
        svd_entry = tk.Entry(svd_frame, width=10, textvariable=tk.StringVar(tab, value='0.025'))
        svd_entry.grid(column=1, row=0)
        # Builds the frame.
        svd_frame.grid(column=0, row=2)

        # Creates the frame for users to input a number of iterations.
        iter_frame = tk.Frame(tab)
        # Creates the "# of Iterations" label.
        iter_lbl = tk.Label(iter_frame, text='# of Iterations:')
        iter_lbl.grid(column=0, row=0)
        # Defaults the number of iterations to 10.
        iter_entry = tk.Entry(iter_frame, width=10, textvariable=tk.StringVar(tab, value='10'))
        iter_entry.grid(column=1, row=0)
        # Builds the frame.
        iter_frame.grid(column=0, row=3)

        # Creates the Frame for users to run the method.
        run_frame = tk.Frame(tab)
        # Defines the run button.
        run_btn = tk.Button(run_frame, text='Run', command=run_imp)
        run_btn.grid(column=0, row=0)
        # Builds the frame.
        run_frame.grid(column=0, row=4)

        # Calls the function to make the frame for displaying time results.
        pre_process_time, iter_choices, mp_time, mode_shape_time, signal_recon_time, cf_time, total_time, def_iter\
            = self.timer_frame(tab)

        # Calls the function to make the frame for displaying the cost function statistics.
        cf_max, cf_min, cf_avg = self.cf_stat_frame(tab)

        # Calls the function to make the frame for displaying the signal plot.
        sig_choices, def_sig, sig_plot, sig_canvas = self.signal_frame(tab)

        # Calls the function to make the frame for displaying the mode plot.
        mode_plot, mode_canvas = self.mode_frame(tab)

        # Calls the function to make the frame for displaying the mode values.
        mode_val = self.mode_results(tab)

    @staticmethod
    def timer_frame(tab):
        # Creates the computation times title.
        time_title = tk.Label(tab, text='Computation Times', font='Helvetica 10 bold')
        time_title.grid(column=0, row=5)

        # Creates a frame for data pre-processing.
        timer_frame = tk.Frame(tab)
        # Creates the data pre-processing time display.
        pre_process_lbl = tk.Label(timer_frame, text='Data Pre-processing:')
        pre_process_lbl.grid(column=0, row=0)
        # Initializes the time as empty.
        pre_process_txt = tk.StringVar()
        pre_process_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=pre_process_txt)
        # Locks the entry from being typed in, since it's display only.
        pre_process_entry.bind("<Key>", lambda e: "break")
        pre_process_entry.grid(column=1, row=0)

        # Creates the iteration number display.
        iter_num_lbl = tk.Label(timer_frame, text='Iteration #:')
        iter_num_lbl.grid(column=0, row=1)
        # Creates the drop down menu.
        iter_default = tk.StringVar()
        iter_default.set('Choose an Iteration')
        # Trace is used to detect when the value inside the combo box has changed.
        iter_default.trace('w', update_times)
        iter_num_combo = ttk.Combobox(timer_frame, textvariable=iter_default, state='readonly')
        # Sets the values of the drop down menu, and the current value (by index).
        iter_num_combo['values'] = ()
        iter_num_combo.current()
        iter_num_combo.grid(column=1, row=1)
        iter_num_combo.bind("<<ComboboxSelected>>", iter_chooser)

        # Creates the matrix pencil method time display.
        mp_lbl = tk.Label(timer_frame, text='Matrix Pencil Method:')
        mp_lbl.grid(column=0, row=2)
        # Initializes the time as empty.
        mp_txt = tk.StringVar()
        mp_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=mp_txt)
        # Locks the entry from being typed in, since it's display only.
        mp_entry.bind("<Key>", lambda e: "break")
        mp_entry.grid(column=1, row=2)

        # Creates the mode shape calculation time display.
        mode_shape_lbl = tk.Label(timer_frame, text='Mode Shape Calculation:')
        mode_shape_lbl.grid(column=0, row=3)
        # Initializes the time as empty.
        mode_shape_txt = tk.StringVar()
        mode_shape_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=mode_shape_txt)
        # Locks the entry from being typed in, since it's display only.
        mode_shape_entry.bind("<Key>", lambda e: "break")
        mode_shape_entry.grid(column=1, row=3)

        # Creates the signal reconstruction time display.
        signal_recon_lbl = tk.Label(timer_frame, text='Signal Reconstruction:')
        signal_recon_lbl.grid(column=0, row=4)
        # Initializes the time as empty.
        signal_recon_txt = tk.StringVar()
        signal_recon_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=signal_recon_txt)
        # Locks the entry from being typed in, since it's display only.
        signal_recon_entry.bind("<Key>", lambda e: "break")
        signal_recon_entry.grid(column=1, row=4)

        # Creates the cost function time display.
        cf_lbl = tk.Label(timer_frame, text='Cost Function Calculation:')
        cf_lbl.grid(column=0, row=5)
        # Initializes the time as empty.
        cf_txt = tk.StringVar()
        cf_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=cf_txt)
        # Locks the entry from being typed in, since it's display only.
        cf_entry.bind("<Key>", lambda e: "break")
        cf_entry.grid(column=1, row=5)

        # Creates the total time display.
        total_lbl = tk.Label(timer_frame, text='Total Run Time:')
        total_lbl.grid(column=0, row=6)
        # Initializes the time as empty.
        total_txt = tk.StringVar()
        total_entry = tk.Entry(timer_frame, relief=tk.GROOVE, text=total_txt)
        # Locks the entry from being typed in, since it's display only.
        total_entry.bind("<Key>", lambda e: "break")
        total_entry.grid(column=1, row=6)

        timer_frame.grid(column=0, row=6)

        return pre_process_entry, iter_num_combo, mp_entry, mode_shape_entry,\
            signal_recon_entry, cf_entry, total_entry, iter_default

    @staticmethod
    def cf_stat_frame(tab):
        # Defines the cost function statistics label.
        cf_title = tk.Label(tab, text='Cost Function Statistics', font='Helvetica 10 bold')
        cf_title.grid(column=0, row=7)

        # Builds the frame to store the cost function information..
        cf_frame = tk.Frame(tab)
        # Builds the maximum cost function display.
        cf_max_lbl = tk.Label(cf_frame, text='Maximum Cost Function:')
        cf_max_lbl.grid(column=0, row=0)
        # Initializes the entry as empty.
        cf_max_txt = tk.StringVar()
        cf_max_entry = tk.Entry(cf_frame, relief=tk.GROOVE, text=cf_max_txt)
        # Locks the entry from being typed in, since it's display only.
        cf_max_entry.bind("<Key>", lambda e: "break")
        cf_max_entry.grid(column=1, row=0)

        # Builds the minimum cost function display.
        cf_min_lbl = tk.Label(cf_frame, text='Minimum Cost Function:')
        cf_min_lbl.grid(column=0, row=1)
        # Initializes the entry as empty.
        cf_min_txt = tk.StringVar()
        cf_min_entry = tk.Entry(cf_frame, relief=tk.GROOVE, text=cf_min_txt)
        # Locks the entry from being typed in, since it's display only.
        cf_min_entry.bind("<Key>", lambda e: "break")
        cf_min_entry.grid(column=1, row=1)

        # Builds the average cost function display.
        cf_avg_lbl = tk.Label(cf_frame, text='Average Cost Function:')
        cf_avg_lbl.grid(column=0, row=2)
        # Initializes the entry as empty.
        cf_avg_txt = tk.StringVar()
        cf_avg_entry = tk.Entry(cf_frame, relief=tk.GROOVE, text=cf_avg_txt)
        # Locks the entry from being typed in, since it's display only.
        cf_avg_entry.bind("<Key>", lambda e: "break")
        cf_avg_entry.grid(column=1, row=2)
        # Builds the frame.
        cf_frame.grid(column=0, row=8)

        return cf_max_entry, cf_min_entry, cf_avg_entry

    @staticmethod
    def signal_frame(tab):
        # Builds the signal plot label.
        sig_plot_frame = tk.Frame(tab)
        # Creates the title of the plot.
        sig_plot_lbl = tk.Label(sig_plot_frame, text='Plot of Signal:', font='Helvetica 10 bold')
        sig_plot_lbl.grid(column=0, row=0)
        # Creates the default option for signal select.
        default_sig = tk.StringVar()
        default_sig.set('Choose a Signal')
        # Trace is used to detect when the value inside the combo box has changed.
        default_sig.trace('w', update_signal)
        # Creates the drop down menu for the signal selector.
        sig_combo = ttk.Combobox(sig_plot_frame, textvariable=default_sig, state='readonly')
        # Sets the values of the drop down menu, and the current value (by index).
        sig_combo['values'] = ()
        sig_combo.current()
        sig_combo.grid(column=1, row=0)
        sig_combo.bind("<<ComboboxSelected>>", signal_chooser)
        sig_plot_frame.grid(column=1, row=0)

        # Creates a frame for the plot.
        plot_frame = tk.Frame(tab)
        # Creates a blank figure, and defines the size in inches.
        main_sig_fig = plt.figure(figsize=(6.25, 2.75))
        # We generate a subplot so that we can create axes labels.
        signal_plot = main_sig_fig.add_subplot(111, frameon=True)
        # Sets the options for the plot.
        signal_plot.grid()
        signal_plot.set_xlabel('Time (s)')
        signal_plot.set_ylabel('Frequency (Hz)')
        signal_plot.set_title('Reconstructed Data vs Original Data')
        plt.tight_layout()
        # Creates a canvas to place the figure, and draws it.
        signal_canvas = FigureCanvasTkAgg(main_sig_fig, master=plot_frame)
        signal_canvas.draw()
        signal_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Draws the frame.
        plot_frame.grid(column=1, row=1, rowspan=6, padx=(1, 1))

        return sig_combo, default_sig, signal_plot, signal_canvas

    @staticmethod
    def mode_frame(tab):
        # Building the frame for the mode label.
        mode_lbl_frame = tk.Frame(tab)
        mode_lbl = tk.Label(mode_lbl_frame, text='Plot of Modes', font='Helvetica 10 bold')
        mode_lbl.grid(column=0, row=0)
        mode_lbl_frame.grid(column=1, row=7)

        # Building the frame for the mode plot.
        mode_plot_frame = tk.Frame(tab)
        # Creates the figure for the plot, and sets the size in inches.
        mode_plot_fig = plt.figure(figsize=(6.25, 2))
        # Creates a subplot so we can set the axes labels.
        mode_plot_static = mode_plot_fig.add_subplot()
        # Sets all the options for the plot.
        mode_plot_static.grid()
        mode_plot_static.set_xlabel('Frequency (Hz)')
        mode_plot_static.set_ylabel('Damping (%)')
        mode_plot_static.set_ylim(-110, 110)
        mode_plot_static.set_xlim(-0.1, 1)
        # We use tight_layout to contain the labels inside the figure.
        plt.tight_layout()
        # Creates the canvas, and puts it inside the frame.
        mode_canvas_static = FigureCanvasTkAgg(mode_plot_fig, master=mode_plot_frame)
        mode_canvas_static.draw()
        mode_canvas_static.get_tk_widget().pack()
        # Draws the frame.
        mode_plot_frame.grid(column=1, row=8, rowspan=8, padx=(1, 1))

        return mode_plot_static, mode_canvas_static

    @staticmethod
    def mode_results(tab):
        # Creates a frame for labeling the mode table.
        mode_lbl_frame = tk.Frame(tab)
        f_lbl = tk.Label(mode_lbl_frame, text='Frequency')
        f_lbl.grid(column=0, row=1, ipadx=30)
        damp_lbl = tk.Label(mode_lbl_frame, text='Damping %')
        damp_lbl.grid(column=1, row=1, ipadx=40)
        mode_lbl = tk.Label(mode_lbl_frame, text='Modes', font='Helvetica 10 bold')
        mode_lbl.grid(column=0, row=0, columnspan=2)
        mode_lbl_frame.grid(column=0, row=9)

        # Makes a scrollable version of the modes, in the event we need more space.
        mode_frame = tk.Frame(tab, width=200, height=200)
        mode_frame.grid(column=0, row=10)
        mode_val_canvas = tk.Canvas(mode_frame, width=200, height=200, scrollregion=(0, 0, 0, 1000))
        # Initializes the list to show one blank space for modes.
        for j in range(2):
            tab_entry = tk.Entry(mode_val_canvas, text="")
            tab_entry.grid(row=2, column=j)
            tab_entry.bind("<Key>", lambda e: "break")
        # Creates a vertical scrollbar, in the event we have many modes.
        vbar = tk.Scrollbar(mode_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=mode_val_canvas.yview)
        # Creates the canvas.
        mode_val_canvas.config(width=200, height=200)
        mode_val_canvas.config(yscrollcommand=vbar.set)
        mode_val_canvas.pack(side=tk.LEFT)

        return mode_val_canvas


def openfile():
    # Opens a CSV file for use.
    global file
    # Prompts users to get a CSV file of data.
    filepath = filedialog.askopenfilename(initialdir='C:/', title="Select file",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    # Gets the name of the file to display to users.
    if len(filepath) > 0:
        file = pd.read_csv(filepath, sep=',')
        sig_name = filepath.split('/')[-1]
        sig.delete(0, tk.END)
        sig.insert(0, sig_name)
    else:
        exit()


def run_imp():
    # We set a few of the variables to be global, so that they can be accessed by functions which can't take in any
    # arguments.
    global iter_time, y_hat_data, val_data, time_data
    f_list, b_per, y_hat_data, time_data, val_data, detrend_data, cost_list, detrend_time, \
        imp_time, iter_time, start_time = imp(file, float(svd_entry.get()), int(iter_entry.get()))
    display_result(f_list, b_per, start_time, detrend_time, cost_list, imp_time, iter_time)


def update_times(*args):
    # This function triggers whenever the comobobox is changed, and sets the times to the corresponding iteration.
    curr_iter = def_iter.get()
    if curr_iter != 'Total':
        curr_iter = int(def_iter.get())

    mp = mp_time
    mp.delete(0, tk.END)
    mp.insert(0, '{0:0.4f} s'.format(iter_time[curr_iter]['Matrix Pencil']))

    ms = mode_shape_time
    ms.delete(0, tk.END)
    ms.insert(0, '{0:0.4f} s'.format(iter_time[curr_iter]['Mode Shape Calculation']))

    sr = signal_recon_time
    sr.delete(0, tk.END)
    sr.insert(0, '{0:0.4f} s'.format(iter_time[curr_iter]['Reconstruction']))

    cf = cf_time
    cf.delete(0, tk.END)
    cf.insert(0, '{0:0.4f} s'.format(iter_time[curr_iter]['Cost Function Calculation']))


def update_signal(*args):
    # Gets the currently selected signal.
    curr_signal = def_sig.get()
    # Clears any old data.
    sig_plot.clear()
    # Plots the data.
    sig_plot.plot(time_data, val_data[curr_signal], 'b', time_data, y_hat_data[curr_signal], 'r--')
    # Sets the options for the plot.
    sig_plot.grid()
    sig_plot.set_xlabel('Time (s)')
    sig_plot.set_ylabel('Frequency (Hz)')
    sig_plot.set_title('Reconstructed Data vs Original Data')
    sig_plot.legend(['Original Data', 'Reconstructed Data'])
    # Redraws the plot.
    sig_canvas.draw()


def iter_chooser(event_object):
    global iter_choice
    # This function is used to get the choice from the drop down menu of the iteration # chooser.
    iter_choice = iter_choices.get()
    print(iter_choice)


def signal_chooser(event_object):
    global sig_choice
    # THis function is used to get hte choice from the drop down menu of the signal chooser.
    sig_choice = sig_choices.get()
    print(sig_choice)


def display_result(f, b, start_time, detrend_time, cost_list, imp_time, iter_time_dict):
    # Formats the displays in the results tab to show the results.

    # Updates the iteration choices. Note that we add a 'Total' option, to show the totals for times.
    iter_list = list(iter_time_dict.keys())
    iter_choices['values'] = iter_list
    # Also sets the iteration choice to the last one, to display some default results.
    iter_choices.set(iter_choices['values'][-1])
    # Updates the signal choices.
    sig_choices['values'] = list(cost_list.keys())
    # Also sets the signal choice to the first one, to display some default results.
    sig_choices.set(sig_choices['values'][0])

    # Sets the data pre-processing time.
    detrend = pre_process_time
    detrend.delete(0, tk.END)
    detrend.insert(0, '{0:0.4f} s'.format(detrend_time - start_time))

    # Sets the total run time.
    total = total_time
    total.delete(0, tk.END)
    total.insert(0, '{0:0.4f} s'.format(imp_time - start_time))

    # Sets the maximum cost function.
    max_cost = cf_max
    max_cost.delete(0, tk.END)
    max_cost.insert(0, '{0:0.4f}'.format(np.max(list(cost_list.values()))))

    # Sets the minimum cost function.
    min_cost = cf_min
    min_cost.delete(0, tk.END)
    min_cost.insert(0, '{0:0.4f}'.format(np.min(list(cost_list.values()))))

    # Sets the average cost function.
    avg_cost = cf_avg
    avg_cost.delete(0, tk.END)
    avg_cost.insert(0, '{0:0.4f}'.format(np.mean(list(cost_list.values()))))

    # Plots the modes.
    mode_plot.plot(f, b, 'o')
    mode_canvas.draw()

    # Displays the modes by creating more frames inside the mode canvas.
    for i in range(len(f)):
        for j in range(2):
            if j == 0:
                f_entry = tk.Entry(mode_val, textvariable=tk.StringVar(value='{0:0.4f}'.format(f[i])))
                f_entry.grid(row=i, column=j)
                f_entry.bind("<Key>", lambda e: "break")
            else:
                b_entry = tk.Entry(mode_val, textvariable=tk.StringVar(value='{0:0.4f}'.format(b[i])))
                b_entry.grid(row=i, column=j)
                b_entry.bind("<Key>", lambda e: "break")


def main():
    # Initialization of the window.
    window = tk.Tk()
    GUI(window)
    window.mainloop()


if __name__ == '__main__':
    main()
