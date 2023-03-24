# 用于测试校准位置
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import random
import time
import pprint
from scipy import constants
import os

from ifxAvian import Avian
from internal.fft_spectrum import *
from DBF import DBF
from doppler import DopplerAlgo,linear_to_dB
from data_collecting import num_rx_antennas_from_config,get_max_intensity_row
from radar_config import trace_config

def store_position_data(root_path, tot_time, position, distance):
    result=[]
    index = len(os.listdir(root_path))

    '''
    metrics = Avian.DeviceMetrics(
        sample_rate_Hz =           1_000_000,
        range_resolution_m =       0.05,
        max_range_m =              0.59,
        max_speed_m_s =            3,
        speed_resolution_m_s =     0.2,
        frame_repetition_time_s =  0.15,
        center_frequency_Hz =      60_750_000_000,
        rx_mask =                  5, # activate RX1 and RX3
        tx_mask =                  1,
        tx_power_level =           31,
        if_gain_dB =               33
    )
    '''

    # set config as the Radar SNN
    config = Avian.DeviceConfig(
        sample_rate_Hz = 1_000_000,       # 1MHZ
        rx_mask = 7,                      # activate RX1 RX2 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 33,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 58e9,        # 58.9GHz 
        end_frequency_Hz = 63e9,        # 63.9GHz
        num_chirps_per_frame = 32,       # 32 chirps per frame
        num_samples_per_chirp = 64,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0005, # 0.5ms
        frame_repetition_time_s = 0.075476,   # 75.476ms, frame_Rate = 13.24Hz
    )
    device = Avian.Device()
    # config = device.metrics_to_config(metrics)

    # get metrics and print them
    metrics = device.metrics_from_config(config)
    pprint.pprint(metrics)

    # set configuration
    device.set_config(config)
    # get maximum range
    max_range_m = metrics.max_range_m
    num_rx_antennas=2
    # Create objects to perform range-Doppler and DBF 
    # get the angle of arrival for azimuth and elevation
    azimuth_doppler = DopplerAlgo(config, num_rx_antennas, 0.9) #RX1, RX3
    azimuth_antennas = [0,2]
    elevation_doppler = DopplerAlgo(config, num_rx_antennas, 0.9) #RX2, RX3
    elevation_antennas = [1,2]
    azimuth_dbf = DBF(num_rx_antennas, num_beams = trace_config.num_azimuth_beam, max_angle_degrees = trace_config.max_Azimuth_degress)
    elevation_dbf = DBF(num_rx_antennas, num_beams = trace_config.num_elevation_beam, max_angle_degrees = trace_config.max_Elevation_degress)
    # Continuously calculate the angle of arrival and plot the results
    print("开始收集数据")
    # 计时器
    start_time=time.time()
    while True:
        now = time.time()
        if now-start_time>tot_time:
            break

        frame = device.get_next_frame()
        azimuth_rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)
        elevation_rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)
        azimuth_beam_range_energy = np.zeros((config.num_samples_per_chirp, trace_config.num_azimuth_beam))
        elevation_beam_range_energy = np.zeros((config.num_samples_per_chirp, trace_config.num_elevation_beam))
        for i_ant in range(num_rx_antennas):
            # azimuth and elevation
            # get current RX antenna
            azimuth_mat = frame[azimuth_antennas[i_ant], :, :]
            elevation_mat = frame[elevation_antennas[i_ant], :, :]
            # Compute Doppler spectrum
            azimuth_dfft_dbfs = azimuth_doppler.compute_doppler_map(azimuth_mat, i_ant)
            azimuth_rd_spectrum[:,:,i_ant] = azimuth_dfft_dbfs
            elevation_dfft_dbfs = elevation_doppler.compute_doppler_map(elevation_mat, i_ant)
            elevation_rd_spectrum[:,:,i_ant] = elevation_dfft_dbfs            
        # Compute the range-Angle map
        azimuth_beam_formed = azimuth_dbf.run(azimuth_rd_spectrum)
        for i_beam in range(trace_config.num_azimuth_beam):
            doppler_i = azimuth_beam_formed[:,:,i_beam]
            azimuth_beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(trace_config.num_azimuth_beam)
        elevation_beam_formed = elevation_dbf.run(elevation_rd_spectrum)
        for i_beam in range(trace_config.num_elevation_beam):
            doppler_i = elevation_beam_formed[:,:,i_beam]
            elevation_beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(trace_config.num_elevation_beam)            
        # maximum energy in range-Angle map
        azimuth_max_energy = np.max(azimuth_beam_range_energy)
        elevation_max_energy = np.max(elevation_beam_range_energy)
        # Rescale map to better capture the peak The rescaling is done in a
        # way such that the maximum always has the same value, independent
        # on the original input peak. A proper peak search can greatly
        # improve this algorithm.
        azimuth_beam_range_energy = trace_config.azimuth_scale *(azimuth_beam_range_energy / azimuth_max_energy -1)
        elevation_beam_range_energy = trace_config.elevation_scale *(elevation_beam_range_energy / elevation_max_energy -1)
        # find dominant angle of target
        _, azimuth_idx = np.unravel_index(azimuth_beam_range_energy.argmax(), azimuth_beam_range_energy.shape)
        _, elevation_idx = np.unravel_index(elevation_beam_range_energy.argmax(), elevation_beam_range_energy.shape)
        azimuth_angle = np.linspace(-trace_config.max_Azimuth_degress, trace_config.max_Azimuth_degress, trace_config.num_azimuth_beam)[azimuth_idx]
        elevation_angle = np.linspace(-trace_config.max_Elevation_degress, trace_config.max_Elevation_degress, trace_config.num_elevation_beam)[elevation_idx]
        print("azimuth_angle\t%f\televation_angle\t%f" % (azimuth_angle, elevation_angle))            
        # get the range of target
        azimuth_dfft_dbfs = linear_to_dB(azimuth_dfft_dbfs)
        cur_range = get_max_intensity_row(azimuth_dfft_dbfs.T)
        cur_range = cur_range[::-1]
        range_idx = np.argmax(cur_range)
        ranges = np.linspace(0, max_range_m, cur_range.shape[0])[range_idx]
        # get the coordinates in space of target
        x = ranges * np.cos(azimuth_angle * np.pi / 180) * np.cos(elevation_angle * np.pi / 180)
        y = ranges * np.sin(azimuth_angle * np.pi / 180) * np.cos(elevation_angle * np.pi / 180)
        z = ranges * np.sin(elevation_angle * np.pi / 180)
        result.append([x,y,z])

    with open("%s%d.txt"%(root_path, index), "w") as f:
        f.write("%s\n%s\n"%(str(config),str(metrics)))
        f.write("position:(%d,%d)\tdistance:%d\ttime:%d\n"%( position[0], position[1], distance, tot_time ))
        f.write(str(result))

if __name__=='__main__':
    x=input("请输入横坐标x")
    x=int(x)
    y=input("请输入纵坐标y")
    y=int(y)
    distance=input("请输入距离")
    distance=int(distance)
    root='../data/position'
    # tot=input("请输入测试时长")
    # tot=int(tot)
    store_position_data(root,20,(x,y),distance)