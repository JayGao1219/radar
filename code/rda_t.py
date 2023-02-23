import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pprint
from scipy import constants

from ifxAvian import Avian
from internal.fft_spectrum import *
from DBF import DBF
from doppler import DopplerAlgo,linear_to_dB

def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask
    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c

def get_max_intensity_row(arr):
    max_row_idx = np.argmax(arr.sum(axis=1))
    max_row = arr[max_row_idx, :]
    return max_row

class RealTimePlotter:
    def __init__(self, num_plots, plot_names, plot_scales):
        # 1. 需要把数据整理成对应的热力图格式
        # 2. 需要把数据的范围找到（range，Doppler，angle）
        self.num_plots = num_plots
        self.plot_names = plot_names
        self.plot_scales = plot_scales
        self.fig, self.axs = plt.subplots(num_plots, 1, figsize=(10, 8))
        self._h = []
        self.fig.canvas.manager.set_window_title("Real Time Plotter")
        self.plot_scales = plot_scales
        self.plot_names = plot_names
        self._is_window_open=True
        
    def _draw_first_time(self, data_all_antennas):
        # First time draw
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i in range(len(data_all_antennas)):
            data=data_all_antennas[i]
            h=self.axs[i].imshow(
                data,
                vmin=minmin, vmax=maxmax,
                cmap='hot',
                extent=(self.plot_scales[i][0],
                    self.plot_scales[i][1],
                    self.plot_scales[i][2],
                    self.plot_scales[i][3]),
                origin='lower')
            self._h.append(h)
            self.axs[i].set_title=self.plot_names[i]
            self.axs[i].set_xlabel('Time')
            self.axs[i].set_ylabel(self.plot_names[i])

        # 设置热力图颜色
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar =self.fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")
        # plt.tight_layout()

    def _draw_next_time(self, data_all_antennas):
        # Update data for each antenna
        for i in range(self.num_plots):
            data = data_all_antennas[i]
            self._h[i].set_data(data)

    def draw(self, data_all_antennas):
        # Draw data for all antenna
        if self._is_window_open:
            first_run = len(self._h) == 0
            if first_run:
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            plt.draw()
            plt.pause(1e-3)

    def close(self, event = None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')
            exit(0)

    def is_open(self):
        return self._is_window_open

if __name__ == '__main__':
    num_beams = 27         # number of beams
    max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees

    config = Avian.DeviceConfig(
        sample_rate_Hz = 1_000_000,       # 1MHZ
        rx_mask = 5,                      # activate RX1 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 33,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 58e9,        # 60GHz 
        end_frequency_Hz = 63e9,        # 61.5GHz
        num_chirps_per_frame = 64,       # 128 chirps per frame
        num_samples_per_chirp = 256,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0005, # 0.5ms
        frame_repetition_time_s = 0.15,   # 0.15s, frame_Rate = 6.667Hz
        mimo_mode = 'off'                 # MIMO disabled
    )

    with Avian.Device() as device:
        # set configuration
        device.set_config(config)

        # get metrics and print them
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)

        # get maximum range
        max_range_m = metrics.max_range_m

        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)

        # Create objects for Range-Doppler, DBF, and plotting.
        # distanceExample = DistanceAlgo(config)
        doppler = DopplerAlgo(config, num_rx_antennas, 0.9)
        dbf = DBF(num_rx_antennas, num_beams = num_beams, max_angle_degrees = max_angle_degrees)

        # Define the names and scales of the data to be plotted
        plot_names = ['range', 'doppler', 'angle']

        # 20个frame凑成1张图,3s 生成一张图
        plot_scales = [(0,max_range_m,0, max_range_m), (-metrics.max_speed_m_s,metrics.max_speed_m_s ,-metrics.max_speed_m_s,metrics.max_speed_m_s), (-40, 40,-40,40)]
    
        # Create an instance of the RealTimePlotter class
        num_plots = len(plot_names)
        plotter = RealTimePlotter(num_plots, plot_names, plot_scales)

        # Continuously update the plots
        tot=0
        # data是一个三维数组，['range','doppler','angle']
        data = [[],[],[]]
        while True:
            tot+=1
            # frame has dimension num_rx_antennas x num_samples_per_chirp x num_chirps_per_frame
            frame = device.get_next_frame()

            rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros((config.num_samples_per_chirp, num_beams))

            for i_ant in range(num_rx_antennas): # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                mat = frame[i_ant, :, :]

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:,:,i_ant] = dfft_dbfs

            dfft_dbfs=linear_to_dB(dfft_dbfs)
            # Find dominat doppler of target
            _, idx = np.unravel_index(dfft_dbfs.argmax(),dfft_dbfs.shape)


            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:,:,i_beam]
                beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            # doppler: row correspond to the maximum pixel intensity from the range-doppler map
            cur_doppler=get_max_intensity_row(dfft_dbfs)

            # range: columns correspond to the maximum pixel intensity from the doppler-range map
            cur_range=get_max_intensity_row(dfft_dbfs.T)

            # Maximum energy in Range-Angle map
            # angle
            cur_angle=get_max_intensity_row(beam_range_energy)
            # max_columns_range_angle=get_max_intensity_row(beam_range_energy.T)

            data[0].append(cur_range)
            data[1].append(cur_doppler)
            data[2].append(cur_angle)

            if tot%20==0:
                for item in range(data):
                    data[item]=np.array(data[item])
                # update ploting data
                plotter.update_plots(data)
                data=[[],[],[]]