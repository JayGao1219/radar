# write config file
class data_collect_config:
    max_len=20

class trace_config:
    max_Azimuth_degress=40
    num_azimuth_beam=27
    azimuth_scale = 150
    azimuth_threshold = 1

    max_Elevation_degress=40
    num_elevation_beam=27
    elevation_scale = 150
    elevation_threshold = 1

class angle_correct_config:
    azimuth_a= 1.13
    azimuth_b= -5.49
    elevation_a= 0.81
    elevation_b= 19.43

class position_cofig:
    dx=15
    dy=15
    distance=40
    root="../data/position/"