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
from data_collecting import num_rx_antennas_from_config,get_max_intensity_row
from radar_config import trace_config

def get_trace_in_real_time():
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

    with Avian.Device() as device:
        config = device.metrics_to_config(metrics)
        # set configuration
        device.set_config(config)

        num_rx_antennas=2

        # Create objects to perform range-Doppler and DBF 
        # get the angle of arrival for azimuth and elevation
        azimuth_doppler = DopplerAlgo(config, num_rx_antennas, 0.9) #RX1, RX3
        azimuth_antennas = [0,2]

        elevation_doppler = DopplerAlgo(config, num_rx_antennas, 0.9) #RX2, RX3
        elevation_antennas = [0,1]

        azimuth_dbf = DBF(num_rx_antennas, num_beams = trace_config.num_azimuth_beam, max_angle_degrees = trace_config.max_Azimuth_degress)
        elevation_dbf = DBF(num_rx_antennas, num_beams = trace_config.num_elevation_beam, max_angle_degrees = trace_config.max_Elevation_degress)

        # Continuously calculate the angle of arrival and plot the results
        print("开始收集数据")
        tot=0
        while True:
            tot+=1
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

            print("azimuth_angle: ", azimuth_angle, "elevation_angle: ", elevation_angle)

if __name__=="__main__":
    get_trace_in_real_time()