# 用于测试校准角度
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

def test_angel():
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

    # 创建Qt应用程序
    app = pg.mkQApp()           # App Setup
    # 创建一个窗口
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1000,600)
    win.setWindowTitle('Real-time data plot')

    # 创建两个绘图区域
    p1 = win.addPlot(title="azimuth")
    p2 = win.addPlot(title="elevation")

    # 创建两个数据缓冲区
    data1 = np.zeros(30)
    data2 = np.zeros(30)

    # 创建两个曲线对象
    curve1 = p1.plot(data1, pen='r') # azimuth
    curve2 = p2.plot(data2, pen='b') # elevation

    device = Avian.Device()
    config = device.metrics_to_config(metrics)

    print("config")
    pprint(config)

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

    # 定义更新函数
    def update():
        nonlocal data1, data2

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

        data1[:-1]=data1[1:]
        data1[-1]=azimuth_angle
        data2[:-1]=data2[1:]
        data2[-1]=elevation_angle

        curve1.setData(data1)
        curve2.setData(data2)
        p1.enableAutoRange('y', True)
        p2.enableAutoRange('y', True)
    
    # 定义定时器
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50) # 50ms刷新一次
    
    # 运行应用程序
    app.exec_()

def test_azimuth():
    num_beams = 27         # number of beams
    max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees

    metrics = Avian.DeviceMetrics(
        sample_rate_Hz =           1_000_000,
        range_resolution_m =       0.1,
        max_range_m =              1,
        max_speed_m_s =            0.9,
        speed_resolution_m_s =     0.2,
        frame_repetition_time_s =  0.15,
        center_frequency_Hz =      60_500_000_000,
        rx_mask =                  5, # activate RX1 and RX3
        tx_mask =                  1,
        tx_power_level =           31,
        if_gain_dB =               33
    )
    # 创建Qt应用程序
    app = pg.mkQApp()           # App Setup
    # 创建一个窗口
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1000,600)
    win.setWindowTitle('Real-time data plot')

    # 创建1个绘图区域
    p1 = win.addPlot(title="azimuth")

    # 创建1个数据缓冲区
    data1 = np.zeros(30)

    # 创建两个曲线对象
    curve1 = p1.plot(data1, pen='r') # azimuth

    device = Avian.Device()
    config = device.metrics_to_config(metrics)
    device.set_config(config)

    print("config")
    # get maximum range
    max_range_m = metrics.max_range_m

    # Create frame handle
    num_rx_antennas = 2

    # Create objects for Range-Doppler, DBF, and plotting.
    doppler = DopplerAlgo(config, num_rx_antennas, 0.9)
    dbf = DBF(num_rx_antennas, num_beams = num_beams, max_angle_degrees = max_angle_degrees)

    def update():
        nonlocal data1
        frame = device.get_next_frame()

        rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

        beam_range_energy = np.zeros((config.num_samples_per_chirp, num_beams))

        for i_ant in range(num_rx_antennas): # For each antenna
            # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
            mat = frame[i_ant, :, :]

            # Compute Doppler spectrum
            dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
            rd_spectrum[:,:,i_ant] = dfft_dbfs

        # Compute Range-Angle map
        rd_beam_formed = dbf.run(rd_spectrum)
        for i_beam in range(num_beams):
            doppler_i = rd_beam_formed[:,:,i_beam]
            beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

        # Maximum energy in Range-Angle map
        max_energy = np.max(beam_range_energy)

        scale = 150
        beam_range_energy = scale*(beam_range_energy/max_energy - 1)
        # Find dominant angle of target
        _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
        angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]

        data1[:-1]=data1[1:]
        data1[-1]=angle_degrees
        curve1.setData(data1)
        p1.enableAutoRange('y', True)
    
    # 定义定时器
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50) # 50ms刷新一次
    
    # 运行应用程序
    app.exec_()


def store_angle_data(root_path, angle_data, tot_time, distance):
    filepath='%s%d/'%(root_path, angle_data)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        index=0
    else:
        index=len(os.listdir(filepath))

    result=[]


    num_beams = 27         # number of beams
    max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees
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
        rx_mask = 6,                      # activate RX2 and RX3
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

    with Avian.Device() as device:
        # set configuration
        device.set_config(config)

        # get metrics and print them
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)
        print("**************")

        # get maximum range
        max_range_m = metrics.max_range_m

        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)

        # Create objects for Range-Doppler, DBF, and plotting.
        doppler = DopplerAlgo(config, num_rx_antennas, 0.9)
        dbf = DBF(num_rx_antennas, num_beams = num_beams, max_angle_degrees = max_angle_degrees)

        # 计时器
        start_time=time.time()
        while True:
            now = time.time()
            if now-start_time>tot_time:
                break
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

            # Find dominant angle of target
            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]
            result.append(angle_degrees)

    with open("%s%d.txt"%(filepath, index), "w") as f:
        f.write("%s\n%s\n"%(str(config),str(metrics)))
        f.write("%d\t%d\t%d\n"%(angle_data, tot_time, distance))
        f.write(str(result))

if __name__=="__main__":
    angle=input("请输入想要测试的角度:")
    angle=int(angle)
    # tot=input("请输入测试时长")
    # tot=int(tot)
    distance=input("请输入距离")
    distance=int(distance)
    root='../../data/'
    store_angle_data(root, angle, 20, distance)