from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import matplotlib.pyplot as plt
import manualpicking as manual
import numpy as np
import os
import firstbreak as fb
import picoscopereader as psdata
import plug_parameter as plug
import fit_tanh as fit
from scipy.fftpack import rfft, rfftfreq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import linregress

class programa():
    def __init__(self,root):
        root.title("PyRockPhysics")

        self.titulo=Label(root, text="PyRockPhysics", font=("Verdana", "24", "bold"))
            
        self.man_pick = Button(root, text="Manual Picking", command=self.manual_picking, background="white")
        self.auto_pick = Button(root, text="Automatic Picking", command=self.auto_picking, background="white")
        self.attenuation = Button(root, text="Attenuation Estimate", command=self.attenuation, background="white")

        self.titulo.grid(row=0, column=0, columnspan=3)

        self.man_pick.grid(row=4, column=0, rowspan=5, padx=3, sticky="EW")
        self.auto_pick.grid(row=4, column=1, rowspan=5, padx=3, sticky="EW")
        self.attenuation.grid(row=4, column=2, rowspan=5, padx=3, sticky="EW")
    
    def manual_picking(self):
        manual_picking()

    def auto_picking(self):
        auto_picking()

    def attenuation(self):
        attenuation()

class manual_picking():
    
    def __init__(self):

        self.manual = Toplevel()
        self.manual.title("Manual Picking Module")

        self.titulo=Label(self.manual, text="Manual Picking Module", font=("Verdana", "24", "bold"))

        self.file = Button(self.manual, text="File", command=self.file, background="white")
        self.label_file = Label(self.manual, text="", wraplength=400)
        self.label_length = Label(self.manual,text="Length (mm):")
        self.length = Entry(self.manual)
        self.run = Button(self.manual, text="Run", command=self.run, background="white")

        self.titulo.grid(row=0, column=0, columnspan=3)

        self.file.grid(row=4, column=1, sticky="EW")
        self.label_file.grid(row=5, column=0, columnspan=3, rowspan=3, sticky="W")
        self.label_length.grid(row=8, column=0, sticky="WE")
        self.length.grid(row=8, column=1, sticky="WE")
        self.run.grid(row=8, column=2, sticky="WE")

    def file(self):       
        
        self.file_name = filedialog.askopenfilename()
        self.time, self.waves = psdata.load_psdata_bufferized(self.file_name, False, "buffer")
        
        self.label_file["text"] = ("Arquivo: %s" %self.file_name)


    def run(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def print_pick_data(pick_position):
            print("Pick = {}; Velocity = {}".format(pick_position, float(self.length.get())/(pick_position)))
     
        manual.main(self.time, self.waves[0], float(self.length.get()), ax, print_pick_data)

        ax.set_xlim(0, np.nanmax(self.time))
        ymax = np.percentile(np.abs(self.waves[0][(~np.isnan(self.waves[0])) & (self.time >= 0)]), 99.95)
        ax.set_ylim(-1.2*ymax, 1.2*ymax)

        plt.show()

class auto_picking():

    def __init__(self):

        self.auto = Toplevel()
        self.auto.title("Automatic Picking Module")

        self.titulo=Label(self.auto, text="Automatic Picking Module", font=("Verdana", "24", "bold"))

        self.file_ftf = Button(self.auto, text="Face-to-face File", command=self.file_ftf, background="white")
        self.label_file_ftf = Label(self.auto, text="", wraplength=200)

        self.file_sample = Button(self.auto, text="Sample File", command=self.file_sample, background="white")
        self.label_file_sample = Label(self.auto, text="", wraplength=200)
        self.label_sample_type = Label(self.auto, text = "Sample Type", font=("Verdana","10"))
        self.sample_type = ttk.Combobox(self.auto, value = ("Aluminium", "Dolomite", "Limestone", "Sandstone", "Shale", "Titanium"), font=("Verdana","10"))
        self.label_length = Label(self.auto,text="Length (mm):",font=("Verdana","10"))
        self.length = Entry(self.auto)

        self.run = Button(self.auto, text="Run", command=self.run, background="white")

        self.titulo.grid(row=0, column=0, columnspan=6)

        self.file_ftf.grid(row=4, column=0, columnspan=3, padx=6, sticky="EW")
        self.label_file_ftf.grid(row=5, column=0, columnspan=3, rowspan=3, sticky="EW")

        self.file_sample.grid(row=4, column=3, columnspan=3, sticky="EW")
        self.label_file_sample.grid(row=5, column=3, columnspan=3, rowspan=3, sticky="EW")
        self.label_sample_type.grid(row=8, column=0, columnspan=2, pady=10, sticky="EW")
        self.sample_type.grid(row=8, column=2, pady=10, padx=6, sticky="EW")
        self.label_length.grid(row=8, column=3, pady=10, sticky="EW")
        self.length.grid(row=8, column=4, pady=10, sticky="EW")

        self.run.grid(row=9, column=4, columnspan=2, sticky="EW")

    def file_ftf(self):       
        
        self.file_name_ftf = filedialog.askopenfilename()
        self.time_ftf, self.waves_ftf = psdata.load_psdata_bufferized(self.file_name_ftf, False, "buffer")
        
        self.label_file_ftf["text"] = ("Arquivo: %s" %self.file_name_ftf)

    def file_sample(self):

        self.file_name_sample = filedialog.askopenfilename()
        self.time_sample, self.waves_sample = psdata.load_psdata_bufferized(self.file_name_sample, False, "buffer")
        
        self.label_file_sample["text"] = ("Arquivo: %s" %self.file_name_sample)


    def run(self):

        ##################### Face-to-Face First Break ###########################

        results_ftf = fb.first_break(self.time_ftf, self.waves_ftf[0])
        t0 = results_ftf['arrival']
        p05_ftf = results_ftf['p05']
        p95_ftf = results_ftf['p95']

        label = 'Face to face time = {:.2f} us'.format(t0)

        plt.figure()
        plt.suptitle(os.path.basename(self.file_name_ftf))
        plt.subplot(1, 1, 1)
        plt.plot(self.time_ftf, self.waves_ftf[0], "C7")
        plt.vlines(t0, np.nanmin(self.waves_ftf[0]), np.nanmax(self.waves_ftf[0]), colors='C0', label=label)
        plt.fill_betweenx([np.nanmin(self.waves_ftf[0]), np.nanmax(self.waves_ftf[0])], p05_ftf, p95_ftf, color="C3", alpha=0.5, label="90% confidence interval")
        plt.xlim(0.0, np.nanmax(self.time_ftf))
        plt.xlabel('Time (us)')
        plt.ylabel('Amplitude')
        plt.legend()

        print("Face to face file: {}\nFirst break: {:.2f} us\n".format(os.path.basename(self.file_name_ftf), t0))

        ################### Sample First Break ##################################

        length = float(self.length.get())
        sample_type = self.sample_type.get()

        velocities = plug.standard_velocities(sample_type, 'material_velocities.csv')
        vmin = velocities[0]/1000.0
        vmax = velocities[1]/1000.0

        results_sample = fb.first_break(self.time_sample, self.waves_sample[0], t0, length=length, vmin=vmin, vmax=vmax, run_twice=True)

        t_first_break = results_sample['arrival']
        p05_sample = results_sample['p05']
        p95_sample = results_sample['p95']

        V = length/t_first_break
        V *= 1000.0
        label = 'First break time = {:.2f} us\nVelocity = {:.2f} m/s'.format(t_first_break, V)

        plt.figure()
        plt.suptitle(os.path.basename(self.file_name_sample))
        plt.subplot(1, 1, 1)
        plt.plot(self.time_sample, self.waves_sample[0], "C7")
        plt.vlines(t_first_break, np.nanmin(self.waves_sample[0]), np.nanmax(self.waves_sample[0]), colors='C0', label=label)
        plt.fill_betweenx([np.nanmin(self.waves_sample[0]), np.nanmax(self.waves_sample[0])], p05_sample, p95_sample, color="C3", alpha=0.5, label="90% confidence interval")
        plt.xlim(0.0, np.nanmax(self.time_sample))
        plt.xlabel('Time (us)')
        plt.ylabel('Amplitude (V)')
        plt.legend()

        print("Sample file: {}\nFirst break time = {:.2f} us\nVelocity = {:.2f} m/s\n".format(os.path.basename(self.file_name_sample), t_first_break, V))
        print("")
    
        plt.show()

class attenuation():

    def __init__(self):
        
        self.atten = Toplevel()
        self.atten.title("Attenuation Estimate Module")

        self.titulo=Label(self.atten, text="Attenuation Estimate Module", font=("Verdana", "24", "bold"))

        self.ftf_fb_time = Entry(self.atten)
        self.label_ftf_fb_time = Label(self.atten, text="Face-to-Face time (us):", font=("Verdana","10"))

        self.subtitulo_sample = Label(self.atten, text="Sample", font=("Verdana", "10", "bold", ))
        self.file_sample = Button(self.atten, text="File", command=self.file_sample, background="white")
        self.label_file_sample = Label(self.atten, text="", wraplength=200)
        self.label_length_sample = Label(self.atten,text="Length (mm):",font=("Verdana","10"))
        self.length_sample = Entry(self.atten)
        self.fb_time_sample = Entry(self.atten)
        self.label_fb_time_sample = Label(self.atten,text="First Break time (us):",font=("Verdana","10"))

        self.subtitulo_reference = Label(self.atten, text="Reference Sample", font=("Verdana", "10", "bold", ))
        self.file_sample_reference = Button(self.atten, text="File", command=self.file_sample_reference, background="white")
        self.label_file_sample_reference = Label(self.atten, text="", wraplength=200)
        self.label_length_sample_reference = Label(self.atten,text="Length (mm):",font=("Verdana","10"))
        self.length_sample_reference = Entry(self.atten)
        self.fb_time_sample_reference = Entry(self.atten)
        self.label_fb_time_sample_reference = Label(self.atten,text="First Break time (us):",font=("Verdana","10"))

        self.run_frequency_domain = Button(self.atten, text="Frequency Domain", command=self.frequency_domain, background="white")

        self.min_frequency_label = Label(self.atten, text="Minimum Frequency:",font=("Verdana","10"))
        self.min_frequency = Entry(self.atten)

        self.max_frequency_label = Label(self.atten, text="Maximum Frequency:",font=("Verdana","10"))
        self.max_frequency = Entry(self.atten)

        self.run = Button(self.atten, text="Estimate Attenution", command=self.run, background="white")
        
        self.titulo.grid(row=0, column=0, columnspan=6)

        self.label_ftf_fb_time.grid(row=4, column=0, columnspan=3, sticky="EW")
        self.ftf_fb_time.grid(row=4, column=3, columnspan=3, sticky="EW")

        self.subtitulo_sample.grid(row=5, column=0, columnspan=3, padx= 6, sticky="EW")
        self.file_sample.grid(row=6, column=0, columnspan=3, padx= 6, sticky="EW")
        self.label_file_sample.grid(row=7, column=0, columnspan=3, padx= 6, sticky="EW")
        self.label_length_sample.grid(row=8, column=0, columnspan=2, sticky="EW")
        self.length_sample.grid(row=8, column=2, padx= 6, sticky="EW")
        self.fb_time_sample.grid(row=9, column=2, padx= 6, pady=6, sticky="EW")
        self.label_fb_time_sample.grid(row=9, column=0, columnspan=2, pady=6, sticky="EW")

        self.subtitulo_reference.grid(row=5, column=3, columnspan=3, sticky="EW")
        self.file_sample_reference.grid(row=6, column=3, columnspan=3, sticky="EW")
        self.label_file_sample_reference.grid(row=7, column=3, columnspan=3, sticky="EW")
        self.label_length_sample_reference.grid(row=8, column=3, columnspan=2, sticky="EW")
        self.length_sample_reference.grid(row=8, column=5, sticky="EW")
        self.fb_time_sample_reference.grid(row=9, column=5, pady=6, sticky="EW")
        self.label_fb_time_sample_reference.grid(row=9, column=3, columnspan=2, pady=6, sticky="EW")

        self.run_frequency_domain.grid(row=11, column=4, columnspan=2, pady=6, sticky="EW")

        self.min_frequency_label.grid(row=12, column=0, columnspan=2, pady=6, sticky="EW")
        self.min_frequency.grid(row=12, column=2, padx= 6, pady=6, sticky="EW")

        self.max_frequency_label.grid(row=12, column=3, columnspan=2, pady=6, sticky="EW")
        self.max_frequency.grid(row=12, column=5, pady=6, sticky="EW")

        self.run.grid(row=13, column=4, columnspan=2, sticky="EW")
    
    def file_sample(self):
        
        self.file_name_sample = filedialog.askopenfilename()
        self.time_sample, self.wave_sample = psdata.load_psdata_bufferized(self.file_name_sample, False, "buffer")
        
        self.label_file_sample["text"] = ("Arquivo: %s" %self.file_name_sample)

    def file_sample_reference(self):
        
        self.file_name_sample_reference = filedialog.askopenfilename()
        self.time_sample_reference, self.wave_sample_reference = psdata.load_psdata_bufferized(self.file_name_sample_reference, False, "buffer")
        
        self.label_file_sample_reference["text"] = ("Arquivo: %s" %self.file_name_sample_reference)
    
    def frequency_domain(self):

        global min_length

        if float(self.length_sample.get()) < float(self.length_sample_reference.get()):
            min_length = float(self.length_sample.get())
        else:
            min_length = float(self.length_sample_reference.get())

        f2f = float(self.ftf_fb_time.get())
        f2f *= 0.001

        ######################## Pass Sample to Frequency Domain ###############################

        time_sample, wave_sample = self.time_sample, self.wave_sample[0]
        time_sample *= 0.001
        fb_time_sample = float(self.fb_time_sample.get())
        fb_time_sample *= 0.001

        global v_sample
        v_sample = float(self.length_sample.get())/(fb_time_sample - f2f)

        t1_sample = fb_time_sample + min_length/v_sample

        where_sample = (time_sample >= fb_time_sample) & (time_sample <= t1_sample)

        global wave_sample_f
        wave_sample_f = np.fft.rfft(wave_sample[where_sample])
        global freq_sample
        freq_sample = np.fft.rfftfreq(np.sum(where_sample), time_sample[1] - time_sample[0])

        ylim_sample = np.max(np.abs(wave_sample))*1.1
        ylim_sample_f = np.max(np.abs(wave_sample_f))*1.1

        ####################### Pass Sample Refrence to Frequency Domain ###############################

        time_sample_reference, wave_sample_reference = self.time_sample_reference, self.wave_sample_reference[0]
        time_sample_reference *= 0.001
        fb_time_sample_reference = float(self.fb_time_sample_reference.get())
        fb_time_sample_reference *= 0.001

        v_sample_reference = float(self.length_sample_reference.get())/(fb_time_sample_reference - f2f)

        t1_sample_reference = fb_time_sample_reference + min_length/v_sample_reference

        where_sample_reference = (time_sample_reference >= fb_time_sample_reference) & (time_sample_reference <= t1_sample_reference)

        global wave_sample_reference_f
        wave_sample_reference_f = np.fft.rfft(wave_sample_reference[where_sample_reference])
        global freq_sample_reference
        freq_sample_reference = np.fft.rfftfreq(np.sum(where_sample_reference), time_sample_reference[1] - time_sample_reference[0])

        ylim_sample_reference = np.max(np.abs(wave_sample_reference))*1.1
        ylim_sample_reference_f = np.max(np.abs(wave_sample_reference_f))*1.1

        plt.figure()
        plt.suptitle(os.path.basename(self.file_name_sample))
        plt.subplot(1, 1, 1)
        plt.plot(freq_sample, np.abs(wave_sample_f))
        plt.xlim(0.0, np.max(freq_sample))
        plt.ylim(0, ylim_sample_f)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Amplitude')

        plt.figure()
        plt.suptitle(os.path.basename(self.file_name_sample_reference))
        plt.subplot(1, 1, 1)
        plt.plot(freq_sample_reference, np.abs(wave_sample_reference_f))
        plt.xlim(0.0, np.max(freq_sample_reference))
        plt.ylim(0, ylim_sample_reference_f)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Amplitude')

        plt.show()

    def run(self):

        ref_data_f_resamp = np.interp(freq_sample, freq_sample_reference, np.abs(wave_sample_reference_f))

        where_f = (freq_sample >= float(self.min_frequency.get())) & (freq_sample <= float(self.max_frequency.get()))

        x = freq_sample[where_f]
        y = np.log(np.abs(ref_data_f_resamp)/np.abs(wave_sample_f))[where_f]
        a, b, c, d = fit.fit_tanh(x, y)
        y_ = a*np.tanh(c*x + d) + b

        y0 = -a + b
        y1 = a + b
        xc = -d/c
        x0 = (x[0] + xc)/2.0
        x1 = (x[-1] + xc)/2.0

        dydx = (y1 - y0)/(x1 - x0)

        Q = np.pi*min_length/dydx/v_sample

        plt.subplot(1, 1, 1)
        plt.plot(x, y)
        plt.plot(x, y_)
        plt.plot([x0, x1], [y0, y1], label="Q = {:.2f}".format(Q))
        plt.legend()
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Ln (A_ref/A_sample)')

        plt.show()

root = Tk()
programa(root)
root.mainloop()