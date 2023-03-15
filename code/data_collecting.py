import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pprint
from scipy import constants
import os

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
    def __init__(self, num_plots, plot_names, plot_scales,filepath,image_index):
        # 1. 需要把数据整理成对应的热力图格式
        # 2. 需要把数据的范围找到（range，Doppler，angle）
        self.num_plots = num_plots
        self.plot_names = plot_names
        self.plot_scales = plot_scales
        self.fig, self.axs = plt.subplots(num_plots, 1, figsize=(10, 15))
        self._h = []
        self.fig.canvas.manager.set_window_title("Real Time Plotter")
        self.plot_scales = plot_scales
        self.plot_names = plot_names
        self._is_window_open=True
        self.filepath=filepath
        self.image_index=image_index
        
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
                    self.plot_scales[i][3]))
            self._h.append(h)
            self.axs[i].set_title=self.plot_names[i]
            self.axs[i].set_xlabel('Time')
            self.axs[i].set_ylabel(self.plot_names[i])
            self.axs[i].set_aspect('auto')

    def _draw_next_time(self, data_all_antennas):
        # Update data for each antenna
        for i in range(self.num_plots):
            data = data_all_antennas[i]
            self._h[i].set_data(data)

    def draw(self, data_all_antennas):
        self.image_index+=1
        # Draw data for all antenna
        if self._is_window_open:
            first_run = len(self._h) == 0
            if first_run:
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            plt.draw()
            plt.pause(1e-3)

            plt.savefig("%s%d.png"%(self.filepath,self.image_index))
            

    def close(self, event = None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')
            exit(0)

    def is_open(self):
        return self._is_window_open

def data_collect_in_real_time(root,gesture,number):
    filepath='%s%s/'%(root,gesture)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        begin=0
    else:
        begin=len(os.listdir(filepath))

    num_beams = 27         # number of beams
    max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees

    # config = Avian.DeviceConfig(
    #     sample_rate_Hz = 1_000_000,       # 1MHZ
    #     rx_mask = 7,                      # activate RX1 RX2 RX3
    #     tx_mask = 1,                      # activate TX1
    #     if_gain_dB = 33,                  # gain of 33dB
    #     tx_power_level = 31,              # TX power level of 31
    #     start_frequency_Hz = 58e9,        # bandwith 5GHz 
    #     end_frequency_Hz = 63e9,        # 61.5GHz
    #     num_chirps_per_frame = 32,       # 128 chirps per frame
    #     num_samples_per_chirp = 64,       # 64 samples per chirp
    #     chirp_repetition_time_s = 0.0005, # 0.5ms
    #     frame_repetition_time_s = 0.15,   # 0.15s, frame_Rate = 6.667Hz
    #     mimo_mode = 'off'                 # MIMO disabled
    # )

    metrics = Avian.DeviceMetrics(
        sample_rate_Hz =           1_000_000,
        range_resolution_m =       0.1,
        max_range_m =              1.5,
        max_speed_m_s =            1.9,
        speed_resolution_m_s =     0.2,
        frame_repetition_time_s =  0.15,
        center_frequency_Hz =      60_500_000_000,
        rx_mask =                  7,
        tx_mask =                  1,
        tx_power_level =           31,
        if_gain_dB =               33
    )

    config = device.metrics_to_config(metrics)
    with Avian.Device() as device:
        # set configuration
        device.set_config(config)

        # get metrics and print them
        # metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)

        # Create frame handle
        num_rx_antennas = 2
        
        # get maximum range
        max_range_m = metrics.max_range_m

        #get maximum speed
        max_speed_m_s = metrics.max_speed_m_s

        # Create objects for Range-Doppler, DBF, and plotting.
        # distanceExample = DistanceAlgo(config)
        doppler = DopplerAlgo(config, num_rx_antennas, 0.9)
        dbf = DBF(num_rx_antennas, num_beams = num_beams, max_angle_degrees = max_angle_degrees)

        # Define the names and scales of the data to be plotted
        plot_names = ['range', 'doppler', 'angle']

        # 20个frame凑成1张图,3s 生成一张图
        plot_scales = [(0,20,0,max_range_m), (0,20,-max_speed_m_s,max_speed_m_s), (0,20,-max_angle_degrees,max_angle_degrees)]
    
        # Create an instance of the RealTimePlotter class
        num_plots = len(plot_names)
        plotter = RealTimePlotter(num_plots, plot_names, plot_scales, filepath, begin)

        # Continuously update the plots
        tot=0
        # data是一个三维数组，['range','doppler','angle']
        data = [[],[],[]]
        print("开始收集数据")
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

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Rescale map to better capture the peak The rescaling is done in a
            # way such that the maximum always has the same value, independent
            # on the original input peak. A proper peak search can greatly
            # improve this algorithm.
            scale = 150
            beam_range_energy = scale*(beam_range_energy/max_energy - 1)

            # doppler: row correspond to the maximum pixel intensity from the range-doppler map
            cur_doppler=get_max_intensity_row(dfft_dbfs)

            # range: columns correspond to the maximum pixel intensity from the doppler-range map
            cur_range=get_max_intensity_row(dfft_dbfs.T)
            # reverse cur_range
            cur_range=cur_range[::-1]

            # Maximum energy in Range-Angle map
            # angle
            cur_angle=get_max_intensity_row(beam_range_energy)
            max_columns_range_angle=get_max_intensity_row(beam_range_energy.T)

            data[0].append(cur_range)
            data[1].append(cur_doppler)
            data[2].append(cur_angle)

            if tot>=20:
                cur_data=[[],[],[]]
                for item in range(3):
                    cur_data[item]=data[item][-20:]
                    cur_data[item]=np.array(cur_data[item])
                    cur_data[item]=cur_data[item].T
                    # print(data[item].shape)
                # update ploting data
                plotter.draw(cur_data)

if __name__=='__main__':
    gesture=input("请输入想要录入的手势:")
    number=input("请输入录入手势的数目：")
    number=int(number)
    root='../image/'
    data_collect_in_real_time(root,gesture,number)

