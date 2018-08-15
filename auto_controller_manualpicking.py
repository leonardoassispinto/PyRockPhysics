import picoscopereader as psdata
import matplotlib.pyplot as plt
import sys
import numpy as np
import json

class ClickHandler:
    def __init__(self, line, line_color="r", line_style="solid", callback=None):
        self.line = line
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.vline_extent = None
        self.vlines = []

        self.line_color = line_color
        self.line_style = line_style

        self.callback = callback

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        
        if event.button == 1:
            return
        
        if event.button == 2:
            for i in reversed(range(len(self.vlines))):
                vline = self.vlines.pop()
                vline.remove()
                del vline
            self.line.figure.canvas.draw()
            return
        
        if self.vline_extent is None:
            self.vline_extent = self.line.axes.get_ylim()
        
        ylim = self.line.axes.get_ylim()
        vlines = self.line.axes.vlines(event.xdata, self.vline_extent[0], self.vline_extent[1], colors=self.line_color, linestyles=self.line_style)
        self.line.axes.set_ylim(ylim)
        
        if self.callback is not None:
            self.callback(event.xdata)

        self.vlines.append(vlines)

        self.line.figure.canvas.draw()

def main(time, wave, ax, callback, t0=0, line_color="r", line_style="solid"):
    line, = ax.plot(time, wave)
    click_handler = ClickHandler(line, line_color=line_color, line_style=line_style, callback=callback)
    return line, click_handler

if __name__ == '__main__':

    data = open('measure_information_manualpicking.json', 'r')
    data_json = json.load(data)
    experiment_information = data_json['experiment_information']

    for experiment in experiment_information:
        
        file_name=experiment["file_name"]

        t0=0

        time, waves = psdata.load_psdata_bufferized(file_name, False, "buffer")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def print_pick_data(pick_position):
            print("Pick = {:.2f}".format(pick_position))

        main(time, waves[0], ax, print_pick_data)

        ax.set_xlim(t0, np.nanmax(time))
        ymax = np.percentile(np.abs(waves[0][(~np.isnan(waves[0])) & (time >= t0)]), 99.95)
        ax.set_ylim(-1.2*ymax, 1.2*ymax)

        plt.show()