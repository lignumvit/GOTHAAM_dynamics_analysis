from scipy.optimize import curve_fit
import math
import copy
import os
import netCDF4
import pandas as pd
from datetime import datetime, timedelta
from fnmatch import fnmatch
import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models import Title, CustomJS, Select, TextInput, Button, LinearAxis, Range1d, HoverTool, ColumnDataSource
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.tickers import DatetimeTicker
from bokeh.palettes import Category20
import itertools
from scipy import signal
from scipy.stats import skew
from global_land_mask import globe
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.palettes import Viridis256, Category20b


# Thresholds for determining what data can be considered in AoA coef determination (e.g. straight-and-level).
# Used by mask_straight_and_level.
max_vspd = 8 # m/s
max_roll = 5 # deg
min_tas = 85 # m/s, used to eliminate slow, Hudson passes
# min vspd used to isolate climb situations
min_vspd = 2.5 

# constants from Nimbus
mol_wgt_dry_air = 28.9637 # kg/kmol
R0 = 8314.462618 # J/kmol/K
Rd = R0/mol_wgt_dry_air
Cpd = 7.0/2.0*Rd
Cvd = 5.0/2.0*Rd

read_vars = ['ADIFR', 'BDIFR', 'ADIFRTEMP', 'BDIFRTEMP',
             'QCF', 'QCR', 'QC_A', 'QC_A2',
             'GGLAT','GGLON','GGNSAT','GGQUAL','GGSPD','GGTRK',
             'PSFRD', 'PSXC',
             'VEW', 'VNS', 'VSPD', 'GGVEW', 'GGVNS', 'GGVSPD', 'VEWC', 'VNSC',
             'UI', 'UIC', 'VI', 'VIC', 'WI', 'WIC',
             'PALTF', 'PALT', 'GGALT', 'ALT',
             'TASF', 'TASR', 'TAS_A', 'TAS_A2', 'MACHX',
             'PITCH', 'ROLL', 'THDG','AKRD', 'SSLIP',
             'RHUM', 'RICE', 'ATX', 'BNORMA', 'BLATA', 'BLONGA', 'WDC',
             'THETA','EW_VXL','WSC','WDC','VMR_VXL'
            ]

def is_land_profile(df,st,sp):
    t_sfm = df['Time']
    mask = mask_in_times(df,st,sp)
    land_mask = globe.is_land(df['GGLAT'][mask],df['GGLON'][mask])
    return True if np.mean(land_mask) > 0.5 else False

def hms_to_sfm(hms_str: str):
    hh = int(hms_str[0:2])
    mm = int(hms_str[3:5])
    ss = int(hms_str[6:8])
    return hh*3600 + mm*60 + ss

def to_sfm(beg_end_times: list):
    sfm_list = []
    for pair in beg_end_times:
        sfm_list.append([hms_to_sfm(pair[0]), hms_to_sfm(pair[1])])
    return sfm_list

def hms_dict_to_sfm(beg_end_times: dict):
    sfm_dict = {}
    for flight, pairs in beg_end_times.items():
        sfm_dict[flight] = to_sfm(pairs)
        #print(f"flight: {flight}, pairs: {pairs}, list_sfm: {list_sfm}")
    return sfm_dict

def mask_out_times(df, beg_time=-1, end_time=-1):
    if beg_time != -1 and end_time != -1:
        mask = np.logical_or(df["Time"].to_numpy() <= beg_time,
                             df["Time"].to_numpy() >= end_time)
    else:
        n_pts = len(nc["Time"].to_numpy().squeeze())
        mask = np.ones(n_pts)
    return mask

def mask_in_times(df, beg_time=-1, end_time=-1):
    if beg_time != -1 and end_time != -1:
        mask = np.logical_and(df["Time"].to_numpy() >= beg_time,
                              df["Time"].to_numpy() <= end_time)
    else:
        n_pts = len(nc["Time"].to_numpy().squeeze())
        mask = np.ones(n_pts)
    return mask

def mask_straight_and_level(df,max_roll=max_roll,max_vspd=max_vspd,min_tas=min_tas,min_alt=0,max_tas=600,max_alt=100000):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASF']
    alt = df['GGALT']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, np.abs(vspd) < max_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    mask = np.logical_and(mask, tas < max_tas)
    mask = np.logical_and(mask, alt > min_alt)
    mask = np.logical_and(mask, alt < max_alt)
    nan_check_vars = ['QCF', 'PSFRD', 'ADIFR', 'GGVSPD', 'TASF', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_ascent(df):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASFR']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, vspd > min_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    nan_check_vars = ['QCF', 'PSFD', 'ADIFR', 'GGVSPD', 'TASFR', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_descent(df):
    roll = df['ROLL']
    vspd = df['GGVSPD']
    tas = df['TASFR']
    mask = np.abs(roll) < max_roll
    mask = np.logical_and(mask, vspd < -min_vspd)
    mask = np.logical_and(mask, tas > min_tas)
    nan_check_vars = ['QCF', 'PSFD', 'ADIFR', 'GGVSPD', 'TASFR', 'PITCH', 'AKRD']
    for var in nan_check_vars:
        mask = np.logical_and(mask, np.isfinite(df[var]))
    return mask.to_numpy()

def mask_flying(df):
    tas = df['TASFR']
    mask = tas > 50.
    return mask.to_numpy()

def open_nc(data_dir, freq='1Hz'):
    """
    Opens NetCDF files in the specified directory. 
    freq: '1Hz' looks for standard files (*rf20.nc), '25Hz' looks for high-rate files (*rf20h.nc)
    """
    suffix = "h.nc" if freq == '25Hz' else ".nc"
        
    ppnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, f"*PP??{suffix}")])
    ffnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, f"*ff??{suffix}")])
    tfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, f"*tf??{suffix}")])
    rfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, f"*rf??{suffix}")])
    allfnames = sorted(ppnames + ffnames + tfnames + rfnames)
    
    print(f"Found {len(ppnames)} proficiency, {len(ffnames)} ferry, {len(tfnames)} test, and {len(rfnames)} research flights")
    print(f"Opening all {freq} flight NetCDF Files")
    
    nc_dict = {}
    for fname in allfnames:
        stem = fname.split('.')[0]
        if freq == '25Hz' and stem[-1] == "h":
            flname = stem[-5:-1]
        elif freq == '1Hz' and stem[-1] != "h":
            flname = stem[-4:]
        else:
            flname = stem[-5:-1] if stem[-1] == 'h' else stem[-4:]
    
        try:
            nc_dict[flname] = netCDF4.Dataset(os.path.join(data_dir, fname))
        except Exception as e:
            print(f"Could not read {fname} netcdf: {e}")

    # Try to grab global attributes from the first available file
    if len(allfnames) > 0:
        first_key = list(nc_dict.keys())[0]
        try:
            print('Processing Date & Time: ' + nc_dict[first_key].getncattr('date_created'))
        except Exception:
            pass

    return nc_dict

def read_nc(nc: netCDF4.Dataset, freq='1Hz'):
    """
    Reads the NetCDF file into a Pandas DataFrame.
    If freq='25Hz', flattens the high-rate (sps25) 2D arrays and upsamples 1D arrays to match.
    Creates an absolute datetime column based on the NetCDF time units.
    """
    data = {}
    
    for i in read_vars:
        if i not in nc.variables:
            continue
        try:
            var = nc.variables[i]
            output = var[:]
            
            if freq == '25Hz':
                # If variable is sampled at 25Hz, flatten it from (Time, 25) to a 1D array
                if len(var.dimensions) == 2 and var.dimensions[1] == 'sps25':
                    output = output.flatten()
                # If variable is 1Hz (like ambient temp), repeat the value 25 times per second
                elif len(var.dimensions) == 1 and var.dimensions[0] == 'Time':
                    output = np.repeat(output, 25)
                else:
                    output = np.repeat(output[:, 0], 25) if len(var.dimensions) == 2 else np.repeat(output, 25)
            
            data[i] = pd.Series(output)
        except Exception as e:
            print(f"Error reading {i}: {e}")

    # Calculate accurate timestamps and parse base date
    try:
        time_var = nc.variables['Time']
        output = time_var[:]
        if freq == '25Hz':
            # Add fractional 0.04 second intervals to the base timestamp array
            output = np.repeat(output, 25) + np.tile(np.arange(25)/25.0, len(output))
        data['Time'] = pd.Series(output)
        
        # NEW: Extract the actual flight date from the NetCDF Time attribute
        # e.g., "seconds since 2025-08-27 00:00:00 +0000" -> "2025-08-27"
        time_units = time_var.units
        base_date_str = time_units.split('since')[-1].strip().split(' ')[0]
        base_date = datetime.strptime(base_date_str, '%Y-%m-%d')
        
    except Exception as e:
        print(f"Error reading Time: {e}")
        # Fallback to an arbitrary date if attribute is missing
        base_date = datetime(1970, 1, 1)

    df = pd.DataFrame(data)

    # NEW: Convert to actual Pandas Timestamps (absolute datetime)
    df['datetime'] = base_date + pd.to_timedelta(df['Time'], unit='s')

    return df


def plot_track(df: pd.DataFrame, land_profiles, marine_profiles, title: str =''):
    # get latitude and longitude from dataframe
    latitude = df["GGLAT"].to_numpy().squeeze()
    longitude = df["GGLON"].to_numpy().squeeze()
    
    # update to mercator projection
    k = 6378137
    longitude = longitude * (k * np.pi/180.0)
    latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k

    # create the plot layout and add axis labels
    try:
        plot1 = figure(width=500, height=350, title=f"{title} Land Profiles", x_axis_type="mercator", y_axis_type="mercator") 
        plot1.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        plot1.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        
        plot1.line(longitude,latitude, color="black")
        colors = itertools.cycle(Category20[20])
        for pair in land_profiles:
            mask = mask_in_times(df,pair[0],pair[1])
            plot1.line(longitude[mask],latitude[mask], color=next(colors),line_width=5,line_cap='round')
        plot1.add_tile("CARTODBPOSITRON_RETINA_NOLABELS", retina=True)

        plot2 = figure(width=500, height=350, title=f"{title} Marine Profiles", x_axis_type="mercator", y_axis_type="mercator") 
        plot2.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        plot2.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        
        plot2.line(longitude,latitude, color="black")
        colors = itertools.cycle(Category20[20])
        for pair in marine_profiles:
            mask = mask_in_times(df,pair[0],pair[1])
            plot2.line(longitude[mask],latitude[mask], color=next(colors),line_width=5,line_dash='dashed',line_cap='round')
        plot2.add_tile("CARTODBPOSITRON_RETINA_NOLABELS", retina=True)
    
        p = gridplot([[plot1, plot2]])
        show(p)
    except Exception as e:
        print(e)

from bokeh.models.formatters import DatetimeTickFormatter

def format_ticks(plot):
    """
    Formats the x-axis of a Bokeh plot to show Time in HH:MM:SS
    """
    plot.xaxis.formatter = DatetimeTickFormatter(
        days="%H:%M:%S",
        hours="%H:%M:%S",
        hourmin="%H:%M:%S",
        minutes="%H:%M:%S",
        minsec="%H:%M:%S",
        seconds="%H:%M:%S",
        milliseconds="%H:%M:%S"
    )
    # Optional: Rotate the labels slightly if they overlap when you zoom out
    # plot.xaxis.major_label_orientation = 0.785 # 45 degrees in radians

def plot_profile_ts(df: pd.DataFrame, land_profiles, marine_profiles, all_ts=False):
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    n_ts = len(df['datetime'])
    w_var_ts = np.zeros(n_ts)
    tas_var_ts = np.zeros(n_ts)
    avg_n_sec = 10
    half_avg = round(avg_n_sec/2)
    w = df['WIC'].to_numpy()
    tas = df['TASF'].to_numpy()
    for i in range(half_avg,n_ts-half_avg):
        w_var_ts[i] = np.var(w[i-half_avg:i+half_avg])
        tas_var_ts[i] = np.var(tas[i-half_avg:i+half_avg])

    # generate the altitude, heading and gps quality plots
    height = 200
    width = 1000

    p1 = figure(width=width, height=height)
    p1.add_layout(Title(text="Pres. Alt. [ft]", align="center"), "left")
    p1.line(df['datetime'], df['PALTF'], legend_label='PSFD')
    colors = itertools.cycle(Category20[20])
    for pair in land_profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p1.line(df['datetime'][mask], df['PALTF'][mask], color=next(colors), line_cap='round', line_width=5)
    colors = itertools.cycle(Category20[20])
    for pair in marine_profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p1.line(df['datetime'][mask], df['PALTF'][mask], color=next(colors), line_cap='round',line_dash='dashed', line_width=5)
  
    p1.legend.location = 'bottom_right'
    p1.add_tools(ht)
    format_ticks(p1)

    colors = itertools.cycle(Category20[20])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Roll [deg]", align="center"), "left")
    p2.line(df['datetime'], df['ROLL'], color=next(colors), legend_label='Roll')
    colors = itertools.cycle(Category20[20])
    for pair in land_profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p2.line(df['datetime'][mask], df['ROLL'][mask], color=next(colors), line_cap='round',line_width=2)
    colors = itertools.cycle(Category20[20])
    for pair in marine_profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p2.line(df['datetime'][mask], df['ROLL'][mask], color=next(colors), line_cap='round',line_dash='dashed',line_width=2)
 
    p2.legend.location = 'bottom_right'
    p2.add_tools(ht)
    format_ticks(p2)

    p3 = figure(width=width, height=height, x_range=p1.x_range, y_range=[0,10])
    p3.add_layout(Title(text="Variance [m2/s2]", align="center"), "left")
    p3.line(df['datetime'], w_var_ts, color='black', legend_label='Var(w)')
    p3.line(df['datetime'], tas_var_ts, color='red', legend_label='Var(TAS)')
    p3.legend.location = 'bottom_right'
    p3.add_tools(ht)
    format_ticks(p3)

    p4 = figure(width=width, height=height, x_range=p1.x_range)
    p4.add_layout(Title(text="Theta [K]", align="center"), "left")
    p4.line(df['datetime'], df['THETA'], color='black', legend_label='THETA')
    p4.legend.location = 'bottom_right'
    p4.add_tools(ht)
    format_ticks(p4)

    p5 = figure(width=width, height=height, x_range=p1.x_range)
    p5.add_layout(Title(text="e [hPa]", align="center"), "left")
    p5.line(df['datetime'], df['EW_VXL'], color='black', legend_label='e')
    p5.legend.location = 'bottom_right'
    p5.add_tools(ht)
    format_ticks(p5)

    if all_ts:
        p = gridplot([[p1], [p3], [p4], [p5]])
    else:
        p = gridplot([[p1]])
    show(p)

def plot_profiles(df: pd.DataFrame, profiles: list, title = ''):

    # generate the altitude, heading and gps quality plots
    height = 300
    width = 250

    

    colors = itertools.cycle(Category20[20])
    p1 = figure(width=width, height=height, title=title)
    p1.add_layout(Title(text="Height [km]", align="center"), "left")
    p1.add_layout(Title(text="Potential Temperature [K]", align="center"), "below")
    for pair in profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p1.line(df['THETA'][mask], df['GGALT'][mask]/1000, color=next(colors), width=1.5)

    colors = itertools.cycle(Category20[20])
    p2 = figure(width=width, height=height, title=title, y_range=p1.y_range)
    p2.add_layout(Title(text="Height [km]", align="center"), "left")
    p2.add_layout(Title(text="e [hPa]", align="center"), "below")
    for pair in profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p2.line(df['EW_VXL'][mask], df['GGALT'][mask]/1000, color=next(colors), width=1.5)

    colors = itertools.cycle(Category20[20])
    p3 = figure(width=width, height=height, title=title, y_range=p1.y_range)
    p3.add_layout(Title(text="Height [km]", align="center"), "left")
    p3.add_layout(Title(text="Wind Speed [m/s]", align="center"), "below")
    for pair in profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p3.line(df['WSC'][mask], df['GGALT'][mask]/1000, color=next(colors), width=1.5)

    colors = itertools.cycle(Category20[20])
    p4 = figure(width=width, height=height, title=title, y_range=p1.y_range)
    p4.add_layout(Title(text="Height [km]", align="center"), "left")
    p4.add_layout(Title(text="Wind Direction [deg]", align="center"), "below")
    for pair in profiles:
        mask = mask_in_times(df,pair[0],pair[1])
        p4.line(df['WDC'][mask], df['GGALT'][mask]/1000, color=next(colors), width=1.5)

    p = gridplot([[p1, p2, p3, p4],])
    show(p)

from bokeh.models import LinearColorMapper, ColorBar, Title
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.palettes import tol
import numpy as np

def plot_profiles_by_time_split(data_dict, all_profiles_dict, domain='land', title=''):
    """
    Plots potential temperature profiles colored by relative local time of day (-3 to +3 hours),
    split into four 6-hour periods: Midnight, Morning, Noon, and Evening.
    Profiles are sorted and plotted in chronological order of local time.
    """
    height = 400
    width = 420 
    
    # Initialize the four plots, linking their x and y ranges for easy comparison
    p_midnight = figure(width=width, height=height, title=f"{title} - Midnight (21:00-03:00)")
    p_midnight.add_layout(Title(text="Height [km]", align="center"), "left")
    p_midnight.add_layout(Title(text="Potential Temperature [K]", align="center"), "below")
    
    p_morning = figure(width=width, height=height, title=f"{title} - Morning (03:00-09:00)",
                       x_range=p_midnight.x_range, y_range=p_midnight.y_range)
    p_morning.add_layout(Title(text="Height [km]", align="center"), "left")
    p_morning.add_layout(Title(text="Potential Temperature [K]", align="center"), "below")
    
    p_noon = figure(width=width, height=height, title=f"{title} - Noon (09:00-15:00)", 
                    x_range=p_midnight.x_range, y_range=p_midnight.y_range)
    p_noon.add_layout(Title(text="Height [km]", align="center"), "left")
    p_noon.add_layout(Title(text="Potential Temperature [K]", align="center"), "below")
    
    p_evening = figure(width=width, height=height, title=f"{title} - Evening (15:00-21:00)", 
                       x_range=p_midnight.x_range, y_range=p_midnight.y_range)
    p_evening.add_layout(Title(text="Height [km]", align="center"), "left")
    p_evening.add_layout(Title(text="Potential Temperature [K]", align="center"), "below")
    
    # Grab the 11-color Sunset palette
    palette = tol['Sunset'][11]
    
    # Create the colorbar mapping -3 to +3 hours
    color_mapper = LinearColorMapper(palette=palette, low=-3, high=3)
    
    # Add identical colorbars to ALL plots so that their inner drawing areas remain exactly the same size
    for p in [p_midnight, p_morning, p_noon, p_evening]:
        p.add_layout(ColorBar(color_mapper=color_mapper, title="Rel Time [h]", width=15), 'right')
    
    # List to store profile data for sorting
    profiles_to_plot = []
    
    for flight, df in data_dict.items():
        if not "rf" in flight:
            continue
            
        profs = all_profiles_dict[flight].get(f'{domain}_profiles', [])
        
        for pair in profs:
            mask = mask_in_times(df, pair[0], pair[1])
            if not np.any(mask):
                continue
            
            # Calculate mean UTC time and Longitude to find local time
            mean_time_utc = np.nanmean(df['Time'][mask]) 
            mean_lon = np.nanmean(df['GGLON'][mask])
            
            # Convert UTC seconds from midnight to local time in hours (0-24)
            utc_hours = (mean_time_utc % 86400) / 3600.0
            local_hours = (utc_hours + mean_lon / 15.0) % 24.0
            
            # Route the profile to the correct 6-hour plot and set the center hour
            if 3.0 <= local_hours < 9.0:
                target_plot = p_morning
                center_hour = 6.0
            elif 9.0 <= local_hours < 15.0:
                target_plot = p_noon
                center_hour = 12.0
            elif 15.0 <= local_hours < 21.0:
                target_plot = p_evening
                center_hour = 18.0
            else:  # 21:00 to 03:00
                target_plot = p_midnight
                center_hour = 0.0
                
            # Calculate time difference from the center of the period (-3 to +3)
            dt = ((local_hours - center_hour) + 12) % 24 - 12
            
            # Map dt (-3 to +3) to a color index (0 to 10)
            normalized_dt = (dt + 3) / 6.0
            color_idx = max(0, min(10, int(normalized_dt * 11))) 
            
            hex_color = palette[color_idx]
            
            # Append the data to our list so we can sort it later
            profiles_to_plot.append({
                'local_hours': local_hours,
                'plot': target_plot,
                'theta': df['THETA'][mask],
                'alt': df['GGALT'][mask] / 1000.0,
                'color': hex_color
            })
            
    # Sort the profiles chronologically by local time
    profiles_to_plot.sort(key=lambda x: x['local_hours'])
    
    # Plot the sorted profiles
    for prof in profiles_to_plot:
        prof['plot'].line(prof['theta'], prof['alt'], color=prof['color'], width=1.5)
            
    # Arrange the plots chronologically in a 2x2 grid
    p = gridplot([[p_midnight, p_morning], 
                  [p_noon, p_evening]])
    show(p)

def get_freq_and_window(df: pd.DataFrame, window_seconds: int = 10):
    """Helper to dynamically calculate the row-length of a time window based on the dataframe frequency"""
    dt = df['Time'].iloc[1] - df['Time'].iloc[0]
    freq_hz = int(round(1.0 / dt)) if dt > 0 else 1
    return freq_hz * window_seconds

def detect_climb_descent(df: pd.DataFrame, window_seconds: int = 10):
    profiles = []
    
    # Dynamically scale 10-second window
    win_10s = get_freq_and_window(df, window_seconds)
    
    # Smoothed to prevent rapid 25Hz noise from breaking the logic
    vspd = df['GGVSPD'].rolling(window=win_10s, center=True, min_periods=1).mean().to_numpy()
    roll = df['ROLL'].rolling(window=win_10s, center=True, min_periods=1).mean().to_numpy()
    t_sfm = df['Time'].to_numpy()
    
    in_climb = False
    in_descent = False

    ascent_thresh = 2.5
    ascent_stop_thresh = 0.5 
    descent_thresh = -2.5
    descent_stop_thresh = -0.5
    min_period = 90 

    for i in range(len(t_sfm)):
        if not np.isnan(vspd[i]):
            if not in_climb:
                if vspd[i] >= ascent_thresh:
                    in_climb = True
                    st_time = int(t_sfm[i])
            else:
                if vspd[i] < ascent_stop_thresh:
                    in_climb = False
                    sp_time = int(t_sfm[i])
                    if sp_time - st_time > min_period:
                        profiles.append([st_time, sp_time])

    for i in range(len(t_sfm)):
        if not np.isnan(vspd[i]):
            if not in_descent:
                if vspd[i] <= descent_thresh:
                    in_descent = True
                    st_time = int(t_sfm[i])
            else:
                if vspd[i] > descent_stop_thresh:
                    in_descent = False
                    sp_time = int(t_sfm[i])
                    if sp_time - st_time > min_period:
                        profiles.append([st_time, sp_time])

    return profiles

def detect_straight_level(df: pd.DataFrame, window_seconds: int = 10):
    legs = []
    win_10s = get_freq_and_window(df, window_seconds)

    vspd = df['GGVSPD'].rolling(window=win_10s, center=True, min_periods=1).mean().to_numpy()
    roll = df['ROLL'].rolling(window=win_10s, center=True, min_periods=1).mean().to_numpy()
    tas = df['TASF'].rolling(window=win_10s, center=True, min_periods=1).mean().to_numpy()
    t_sfm = df['Time'].to_numpy()
    
    in_leg = False
    max_vspd_thresh = 1.5 
    max_roll_thresh = 5.0 
    min_tas_thresh = 60.0 
    min_period = 90 

    for i in range(len(t_sfm)):
        if np.isnan(vspd[i]) or np.isnan(roll[i]) or np.isnan(tas[i]):
            is_sl = False
        else:
            is_sl = (abs(vspd[i]) <= max_vspd_thresh) and \
                    (abs(roll[i]) <= max_roll_thresh) and \
                    (tas[i] >= min_tas_thresh)
            
        if not in_leg:
            if is_sl:
                in_leg = True
                st_time = int(t_sfm[i])
        else:
            if not is_sl:
                in_leg = False
                sp_time = int(t_sfm[i])
                if sp_time - st_time > min_period:
                    legs.append([st_time, sp_time])
                    
    if in_leg:
        sp_time = int(t_sfm[-1])
        if sp_time - st_time > min_period:
            legs.append([st_time, sp_time])
            
    return legs

def plot_track_flux(df: pd.DataFrame, legs: list, title: str = ''):
    # get latitude and longitude from dataframe
    latitude = df["GGLAT"].to_numpy().squeeze()
    longitude = df["GGLON"].to_numpy().squeeze()
    
    # update to mercator projection
    k = 6378137
    longitude = longitude * (k * np.pi/180.0)
    latitude = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k

    # create the plot layout and add axis labels
    try:
        p = figure(width=800, height=500, title=f"{title} Flux Legs", x_axis_type="mercator", y_axis_type="mercator") 
        p.add_layout(Title(text="Longitude [Degrees]", align="center"), "below")
        p.add_layout(Title(text="Latitude [Degrees]", align="center"), "left")
        
        # Plot full flight track
        p.line(longitude, latitude, color="black")
        
        # Highlight flux legs
        colors = itertools.cycle(Category20[20])
        for pair in legs:
            mask = mask_in_times(df, pair[0], pair[1])
            p.line(longitude[mask], latitude[mask], color=next(colors), line_width=5, line_cap='round')
            
        p.add_tile("CARTODBPOSITRON_RETINA_NOLABELS", retina=True)
    
        show(p)
    except Exception as e:
        print(e)

def plot_flux_leg_ts(df: pd.DataFrame, legs: list, all_ts: bool = False, width: int = 1000, height: int = 200):
    """
    Plots a time series of flight data, highlighting specific flux legs.
    """
    # set up hover tool
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    n_ts = len(df['datetime'])
    w_var_ts = np.zeros(n_ts)
    tas_var_ts = np.zeros(n_ts)
    avg_n_sec = 10
    half_avg = round(avg_n_sec/2)
    w = df['WIC'].to_numpy()
    tas = df['TASF'].to_numpy()
    for i in range(half_avg, n_ts-half_avg):
        w_var_ts[i] = np.var(w[i-half_avg:i+half_avg])
        tas_var_ts[i] = np.var(tas[i-half_avg:i+half_avg])

    # 1. Altitude Plot
    p1 = figure(width=width, height=height)
    p1.add_layout(Title(text="Pres. Alt. [ft]", align="center"), "left")
    p1.line(df['datetime'], df['PALTF'], color='black', legend_label='PALTF')
    colors = itertools.cycle(Category20[20])
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p1.line(df['datetime'][mask], df['PALTF'][mask], color=next(colors), line_cap='round', line_width=5)
    p1.legend.location = 'bottom_right'
    p1.add_tools(ht)
    format_ticks(p1)

    # 2. Roll Plot
    colors = itertools.cycle(Category20[20])
    p2 = figure(width=width, height=height, x_range=p1.x_range)
    p2.add_layout(Title(text="Roll [deg]", align="center"), "left")
    p2.line(df['datetime'], df['ROLL'], color='black', legend_label='Roll')
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p2.line(df['datetime'][mask], df['ROLL'][mask], color=next(colors), line_cap='round', line_width=3)
    p2.legend.location = 'bottom_right'
    p2.add_tools(ht)
    format_ticks(p2)

    # 3. Vertical Speed Plot (Explicitly labeled GGVSPD)
    colors = itertools.cycle(Category20[20])
    p3 = figure(width=width, height=height, x_range=p1.x_range)
    p3.add_layout(Title(text="GGVSPD [m/s]", align="center"), "left")
    p3.line(df['datetime'], df['GGVSPD'], color='black', legend_label='GGVSPD')
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p3.line(df['datetime'][mask], df['GGVSPD'][mask], color=next(colors), line_cap='round', line_width=3)
    p3.legend.location = 'bottom_right'
    p3.add_tools(ht)
    format_ticks(p3)

    p = gridplot([[p1], [p2], [p3]])
        
    show(p)

from bokeh.layouts import column

def plot_interactive_flux_dashboard(df: pd.DataFrame, legs: list, title: str = ''):
    """
    Creates a dashboard combining a map and time-series plots. 
    Zooming/panning the time series dynamically updates the visible track on the map.
    Includes both raw (light gray) and 10-s smoothed (black) traces.
    """
    # -----------------------------------------------------------
    # 1. Prepare Data & Projections
    # -----------------------------------------------------------
    k = 6378137
    latitude = df["GGLAT"].to_numpy().squeeze()
    longitude = df["GGLON"].to_numpy().squeeze()
    
    longitude_merc = longitude * (k * np.pi/180.0)
    latitude_merc = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
    
    # Pre-calculate 10-second smoothed variables for the plots
    paltf_smooth = df['PALTF'].rolling(window=10, center=True, min_periods=1).mean()
    roll_smooth = df['ROLL'].rolling(window=10, center=True, min_periods=1).mean()
    vspd_smooth = df['GGVSPD'].rolling(window=10, center=True, min_periods=1).mean()

    # We need two sets of DataSources: 'orig' keeps the full arrays untouched in JS, 
    # 'map' gets dynamically overwritten to show only the zoomed time window.
    map_sources = []
    orig_sources = []
    
    # Base Flight Track
    track_src = ColumnDataSource(dict(time=df['datetime'], lon=longitude_merc, lat=latitude_merc))
    orig_track_src = ColumnDataSource(dict(time=df['datetime'], lon=longitude_merc, lat=latitude_merc))
    map_sources.append(track_src)
    orig_sources.append(orig_track_src)
    
    # Highlighted Legs
    leg_colors = []
    colors = itertools.cycle(Category20[20])
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        l_src = ColumnDataSource(dict(time=df['datetime'][mask], lon=longitude_merc[mask], lat=latitude_merc[mask]))
        o_src = ColumnDataSource(dict(time=df['datetime'][mask], lon=longitude_merc[mask], lat=latitude_merc[mask]))
        map_sources.append(l_src)
        orig_sources.append(o_src)
        leg_colors.append(next(colors))

    # -----------------------------------------------------------
    # 2. Build Map Plot
    # -----------------------------------------------------------
    p_map = figure(width=700, height=400, title=f"{title} Map (Linked to Time Series)", 
                   x_axis_type="mercator", y_axis_type="mercator") 
    p_map.add_layout(Title(text="Longitude", align="center"), "below")
    p_map.add_layout(Title(text="Latitude", align="center"), "left")
    p_map.add_tile("CARTODBPOSITRON_RETINA_NOLABELS", retina=True)

    # Plot base track
    p_map.line('lon', 'lat', source=map_sources[0], color="black", line_width=1)
    
    # Plot legs
    for i in range(1, len(map_sources)):
        p_map.line('lon', 'lat', source=map_sources[i], color=leg_colors[i-1], line_width=5, line_cap='round')

    # -----------------------------------------------------------
    # 3. Build Time Series Plots
    # -----------------------------------------------------------
    ht = HoverTool(tooltips=[('time', '@x{%H:%M:%S}'), ('y', '@y')], formatters={'@x': 'datetime'})

    # PALTF (Altitude)
    p_alt = figure(width=700, height=160, x_axis_type='datetime')
    p_alt.add_layout(Title(text="Alt [ft]", align="center"), "left")
    p_alt.line(df['datetime'], df['PALTF'], color='lightgray', legend_label='Raw')
    p_alt.line(df['datetime'], paltf_smooth, color='black', legend_label='10s Avg')
    
    c_iter = itertools.cycle(Category20[20])
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p_alt.line(df['datetime'][mask], paltf_smooth[mask], color=next(c_iter), line_width=4, line_cap='round')
    p_alt.add_tools(ht)
    p_alt.legend.location = 'bottom_right'
    format_ticks(p_alt)

    # ROLL
    c_iter = itertools.cycle(Category20[20])
    p_roll = figure(width=700, height=160, x_range=p_alt.x_range, x_axis_type='datetime')
    p_roll.add_layout(Title(text="Roll [deg]", align="center"), "left")
    p_roll.line(df['datetime'], df['ROLL'], color='lightgray', legend_label='Raw')
    p_roll.line(df['datetime'], roll_smooth, color='black', legend_label='10s Avg')
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p_roll.line(df['datetime'][mask], roll_smooth[mask], color=next(c_iter), line_width=3, line_cap='round')
    p_roll.add_tools(ht)
    p_roll.legend.location = 'bottom_right'
    format_ticks(p_roll)

    # GGVSPD
    c_iter = itertools.cycle(Category20[20])
    p_vspd = figure(width=700, height=160, x_range=p_alt.x_range, x_axis_type='datetime')
    p_vspd.add_layout(Title(text="GGVSPD [m/s]", align="center"), "left")
    p_vspd.line(df['datetime'], df['GGVSPD'], color='lightgray', legend_label='Raw')
    p_vspd.line(df['datetime'], vspd_smooth, color='black', legend_label='10s Avg')
    for pair in legs:
        mask = mask_in_times(df, pair[0], pair[1])
        p_vspd.line(df['datetime'][mask], vspd_smooth[mask], color=next(c_iter), line_width=3, line_cap='round')
    p_vspd.add_tools(ht)
    p_vspd.legend.location = 'bottom_right'
    format_ticks(p_vspd)

    # -----------------------------------------------------------
    # 4. JavaScript Callback for Interactivity
    # -----------------------------------------------------------
    callback_code = """
    const start = xr.start;
    const end = xr.end;
    
    for (let j = 0; j < map_sources.length; j++) {
        const orig = orig_sources[j].data;
        const target = map_sources[j].data;
        
        const t = orig['time'];
        const lon = orig['lon'];
        const lat = orig['lat'];
        
        const new_t = [];
        const new_lon = [];
        const new_lat = [];
        
        for(let i=0; i<t.length; i++) {
            if(t[i] >= start && t[i] <= end) {
                new_t.push(t[i]);
                new_lon.push(lon[i]);
                new_lat.push(lat[i]);
            }
        }
        
        target['time'] = new_t;
        target['lon'] = new_lon;
        target['lat'] = new_lat;
        map_sources[j].change.emit();
    }
    """
    
    js_callback = CustomJS(args=dict(xr=p_alt.x_range, map_sources=map_sources, orig_sources=orig_sources), code=callback_code)
    p_alt.x_range.js_on_change('start', js_callback)
    p_alt.x_range.js_on_change('end', js_callback)

    # -----------------------------------------------------------
    # 5. Display Layout
    # -----------------------------------------------------------
    # Map on top, time series underneath
    dashboard = column(p_map, p_alt, p_roll, p_vspd)
    show(dashboard)

import xarray as xr
import cdsapi
import numpy as np
import os

def download_era5_for_flight(df: pd.DataFrame, output_file: str = "era5_flight_data.nc"):
    """
    Finds the spatial and temporal bounding box of the flight and downloads 
    the corresponding ERA5 Sea Level Pressure and Skin Temperature (Land + Ocean).
    Skips download if the output file already exists locally.
    """
    # Check if the file already exists to save time!
    if os.path.exists(output_file):
        print(f"Found existing ERA5 file at {output_file}. Skipping download.")
        return output_file

    # Extract native Python floats (JSON serializable) using built-in float()
    # and round to 2 decimal places to keep the API request clean.
    min_lat = round(float(df['GGLAT'].min()) - 0.5, 2)
    max_lat = round(float(df['GGLAT'].max()) + 0.5, 2)
    min_lon = round(float(df['GGLON'].min()) - 0.5, 2)
    max_lon = round(float(df['GGLON'].max()) + 0.5, 2)

    # Get times
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()
    year = start_time.strftime("%Y")
    month = start_time.strftime("%m")
    day = start_time.strftime("%d")

    # Get unique hours encompassing the flight
    hours = pd.date_range(start_time.floor('h'), end_time.ceil('h'), freq='h').strftime("%H:00").tolist()

    print(f"Requesting ERA5 SLP & SKT for {year}-{month}-{day} across {len(hours)} hours...")
    print(f"Bounding Box: N:{max_lat}, W:{min_lon}, S:{min_lat}, E:{max_lon}")
    
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'mean_sea_level_pressure', 
                'skin_temperature', # Now fetching skin temperature (land + ocean)
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': hours,
            'area': [max_lat, min_lon, min_lat, max_lon], # North, West, South, East
            'format': 'netcdf',
        },
        output_file)
    
    print(f"ERA5 data successfully downloaded to {output_file}")
    return output_file

def add_era5_to_flight_data(df: pd.DataFrame, era5_file: str = "era5_flight_data.nc"):
    """
    Interpolates the downloaded ERA5 NetCDF to the aircraft track.
    Optimized to compute the 3D interpolation at 1 Hz and map it back to 
    the high-rate track to save computation time, then computes Skin Potential Temp.
    """
    ds = xr.open_dataset(era5_file)
    
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    df_sub = df.iloc[::25].copy()
    
    time_da = xr.DataArray(pd.to_datetime(df_sub['datetime']), dims='points')
    lat_da = xr.DataArray(df_sub['GGLAT'], dims='points')
    lon_da = xr.DataArray(df_sub['GGLON'], dims='points')
    
    print("Interpolating ERA5 data to 1Hz flight track (Fast Mode)...")
    ds_interp = ds.interp(time=time_da, latitude=lat_da, longitude=lon_da, method='linear')
    
    df_sub['ERA5_SLP'] = ds_interp['msl'].values / 100.0  
    df_sub['ERA5_SKT'] = ds_interp['skt'].values  # Skin temperature (skt) in Kelvin
    
    print("Upsampling ERA5 data back to high-rate flight track...")
    df['ERA5_SLP'] = np.nan
    df['ERA5_SKT'] = np.nan
    
    df.loc[df_sub.index, 'ERA5_SLP'] = df_sub['ERA5_SLP'].values
    df.loc[df_sub.index, 'ERA5_SKT'] = df_sub['ERA5_SKT'].values
    
    # Linearly interpolate the 24 NaNs between each 1Hz point
    df['ERA5_SLP'] = df['ERA5_SLP'].interpolate(method='linear').bfill().ffill()
    df['ERA5_SKT'] = df['ERA5_SKT'].interpolate(method='linear').bfill().ffill()
    
    print("Calculating Skin Potential Temperature...")
    df['THETA_SKT'] = df['ERA5_SKT'] * (1000.0 / df['ERA5_SLP']) ** 0.286
    
    return df

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Try importing map tile and boundary libraries safely
try:
    import contextily as cx
    HAS_CX = True
except ImportError:
    HAS_CX = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def plot_static_profile_dashboard(df: pd.DataFrame, bl_periods: list, title: str = '', window_n: int = 10):
    """
    Creates a static dashboard using Matplotlib combining a map, time-series plots, and profile figures.
    Outputs a lightweight PNG image to prevent Jupyter Notebook crashes during large loops.
    Uses Cartopy (if available) for state/political boundaries and lat/lon lines.
    """
    # 1. Prepare Data & Projections
    latitude_deg = df["GGLAT"].to_numpy().squeeze()
    longitude_deg = df["GGLON"].to_numpy().squeeze()
    
    # Manual Mercator for fallback (if Cartopy isn't installed)
    k = 6378137
    longitude_merc = longitude_deg * (k * np.pi/180.0)
    latitude_merc = np.log(np.tan((90 + latitude_deg) * np.pi/360.0)) * k
    
    wic_std = df['WIC'].rolling(window=window_n, center=True, min_periods=1).std()
    theta_skt = df.get('THETA_SKT', pd.Series([np.nan]*len(df)))
    surf_alt = np.zeros(len(df))
    t = df['datetime']

    # 2. Setup Figure and Grid Layout
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"{title} Static Boundary Layer Review", fontsize=16, y=0.98)
    
    # 5 rows, 3 columns. Top 4 rows for Time Series, Bottom row for Map & Profiles
    gs = fig.add_gridspec(5, 3, height_ratios=[1.2, 1, 1.2, 1, 3.5], hspace=0.3)
    
    ax_alt = fig.add_subplot(gs[0, :])
    ax_wic = fig.add_subplot(gs[1, :], sharex=ax_alt)
    ax_theta = fig.add_subplot(gs[2, :], sharex=ax_alt)
    ax_e = fig.add_subplot(gs[3, :], sharex=ax_alt)
    
    # Initialize map axes with Cartopy projection if available
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[4, 0], projection=ccrs.Mercator())
    else:
        ax_map = fig.add_subplot(gs[4, 0])
        
    ax_ptheta = fig.add_subplot(gs[4, 1])
    ax_pe = fig.add_subplot(gs[4, 2], sharey=ax_ptheta)

    # 3. Plot Base Traces
    ax_alt.plot(t, df['GGALT'] / 1000.0, color='black', lw=1, alpha=0.5)
    ax_wic.plot(t, wic_std, color='black', lw=1, alpha=0.5)
    
    ax_theta.plot(t, df['THETA'], color='black', lw=1, alpha=0.5, label='Aircraft Theta')
    ax_theta.plot(t, theta_skt, color='red', lw=1, alpha=0.5, linestyle='--', label='Surface Theta')
    ax_theta.legend(loc='upper right', framealpha=0.9)
    
    ax_e.plot(t, df['EW_VXL'], color='black', lw=1, alpha=0.5)

    # Plot Map Base Track
    if HAS_CARTOPY:
        ax_map.plot(longitude_deg, latitude_deg, color='black', lw=1, alpha=0.3, transform=ccrs.PlateCarree())
    else:
        ax_map.plot(longitude_merc, latitude_merc, color='black', lw=1, alpha=0.3)
    
    # 4. Iterate over BL periods and plot highlights
    colors = plt.cm.tab20.colors  
    
    for idx, pair in enumerate(bl_periods):
        c = colors[idx % len(colors)]
        
        st_sec = hms_to_sfm(pair[0])
        sp_sec = hms_to_sfm(pair[1])
        mask = (df['Time'] >= st_sec) & (df['Time'] <= sp_sec)
        
        if not mask.any():
            continue
            
        t_sub = t[mask]
        
        # Highlight Time Series
        ax_alt.plot(t_sub, (df['GGALT'] / 1000.0)[mask], color=c, lw=3.5, alpha=0.9)
        ax_wic.plot(t_sub, wic_std[mask], color=c, lw=3.5, alpha=0.9)
        ax_theta.plot(t_sub, df['THETA'][mask], color=c, lw=3.5, alpha=0.9)
        ax_e.plot(t_sub, df['EW_VXL'][mask], color=c, lw=3.5, alpha=0.9)
        
        # Highlight Map
        if HAS_CARTOPY:
            ax_map.plot(longitude_deg[mask], latitude_deg[mask], color=c, lw=5, solid_capstyle='round', transform=ccrs.PlateCarree())
        else:
            ax_map.plot(longitude_merc[mask], latitude_merc[mask], color=c, lw=5, solid_capstyle='round')
        
        # Highlight Profiles (Air)
        ax_ptheta.plot(df['THETA'][mask], (df['GGALT'] / 1000.0)[mask], color=c, lw=2)
        ax_ptheta.scatter(df['THETA'][mask], (df['GGALT'] / 1000.0)[mask], color=c, s=12, alpha=0.7, zorder=3)
        
        ax_pe.plot(df['EW_VXL'][mask], (df['GGALT'] / 1000.0)[mask], color=c, lw=2)
        ax_pe.scatter(df['EW_VXL'][mask], (df['GGALT'] / 1000.0)[mask], color=c, s=12, alpha=0.7, zorder=3)
        
        # Highlight Profiles (Surface/SKT)
        ax_ptheta.plot(theta_skt[mask], surf_alt[mask], color=c, lw=2, linestyle='--')
        ax_ptheta.scatter(theta_skt[mask], surf_alt[mask], color=c, s=25, alpha=0.9, zorder=3)

    # 5. Formatting & Cleanup
    ax_alt.set_ylabel("GPS Alt [km]")
    ax_wic.set_ylabel(f"WIC {window_n}p Std")
    ax_theta.set_ylabel("Theta [K]")
    ax_e.set_ylabel("e [hPa]")
    
    # Clean up X-axis on time series
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    for ax in [ax_alt, ax_wic, ax_theta]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Map Formatting
    ax_map.set_title("Flight Track")
    if HAS_CARTOPY:
        # Add Geographic Features
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax_map.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        
        # Add Lat/Lon Gridlines
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        
        # Set map extent slightly past the flight track bounds (safely ignoring NaNs)
        pad = 0.5
        ax_map.set_extent([
            np.nanmin(longitude_deg) - pad, np.nanmax(longitude_deg) + pad,
            np.nanmin(latitude_deg) - pad, np.nanmax(latitude_deg) + pad
        ], crs=ccrs.PlateCarree())
        
        # Add base map tiles (if contextily is installed)
        if HAS_CX:
            try:
                cx.add_basemap(ax_map, crs=ax_map.projection.proj4_init, source=cx.providers.CartoDB.PositronNoLabels)
            except Exception as e:
                pass
    else:
        # Fallback Map Formatting
        ax_map.set_aspect('equal')
        ax_map.set_xticks([]) # Hide ugly mercator coordinates
        ax_map.set_yticks([])
        if HAS_CX:
            try:
                cx.add_basemap(ax_map, source=cx.providers.CartoDB.PositronNoLabels)
            except Exception as e:
                pass

    # Profile Formatting
    ax_ptheta.set_title("Theta Profile")
    ax_ptheta.set_xlabel("Potential Temp [K]")
    ax_ptheta.set_ylabel("Height [km]")
    
    ax_pe.set_title("e Profile")
    ax_pe.set_xlabel("e [hPa]")

    # Render as a static image in the notebook
    plt.tight_layout()
    plt.show()
    
    # Close the figure internally to completely free up memory for the next loop iteration
    plt.close(fig)

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Title, Button, Div, Select, ResetTool
from bokeh.plotting import figure, show
from bokeh.palettes import Category20
import numpy as np
import pandas as pd

def plot_interactive_profile_dashboard(df: pd.DataFrame, title: str = '', window_n: int = 10):
    """
    Creates an interactive dashboard combining a map, time-series plots, and profile figures.
    Includes ERA5 Surface Skin Potential Temp if available in df, plotted on both TS and Profile.
    """
    k = 6378137
    latitude = df["GGLAT"].to_numpy().squeeze()
    longitude = df["GGLON"].to_numpy().squeeze()
    longitude_merc = longitude * (k * np.pi/180.0)
    latitude_merc = np.log(np.tan((90 + latitude) * np.pi/360.0)) * k
    
    wic_std = df['WIC'].rolling(window=window_n, center=True, min_periods=1).std()
    
    # Check if ERA5 data has been appended to this dataframe yet
    theta_skt = df.get('THETA_SKT', pd.Series([np.nan]*len(df)))
    surf_alt = np.zeros(len(df))
    
    data_dict = dict(
        time=df['datetime'],
        lon=longitude_merc,
        lat=latitude_merc,
        alt=df['GGALT'] / 1000.0, 
        surf_alt=surf_alt,
        paltf=df['PALTF'],
        theta=df['THETA'],
        e=df['EW_VXL'],
        wic_std=wic_std,
        theta_skt=theta_skt
    )
    
    orig_src = ColumnDataSource(data_dict)
    active_src = ColumnDataSource(data_dict.copy())
    record_state = ColumnDataSource(dict(start_ms=[], end_ms=[], start_str=[], end_str=[]))
    
    max_highlights = 20
    palette = Category20[max_highlights]
    empty_dict = {k: [] for k in data_dict.keys()}
    h_sources = [ColumnDataSource(empty_dict.copy()) for _ in range(max_highlights)]

    shared_reset = ResetTool()
    ts_tools = ["pan", "box_zoom", "wheel_zoom", "tap", shared_reset, "save"]
    map_tools = ["pan", "wheel_zoom", shared_reset]
    prof_tools = ["pan", "box_zoom", "wheel_zoom", shared_reset, "save"]

    p_map = figure(width=333, height=450, title=f"{title} Map", x_axis_type="mercator", y_axis_type="mercator", tools=map_tools) 
    p_map.add_layout(Title(text="Longitude", align="center"), "below")
    p_map.add_layout(Title(text="Latitude", align="center"), "left")
    p_map.add_tile("CARTODBPOSITRON_RETINA_NOLABELS", retina=True)

    # Base track is a semi-transparent black line
    p_map.line('lon', 'lat', source=orig_src, color="black", line_width=2, alpha=0.7)
    p_map.line('lon', 'lat', source=active_src, color="black", line_width=4, line_cap='round')
    for i in range(max_highlights):
        p_map.line('lon', 'lat', source=h_sources[i], color=palette[i], line_width=6, line_cap='round')

    # ---- Time Series Plots ----
    ts_kwargs = dict(width=1000, x_axis_type='datetime', tools=ts_tools, active_drag="box_zoom")
    
    # 1. Altitude
    p_ts = figure(height=160, title="Select Time Period (Draw a box to filter map/profiles. CLICK to record times.)", **ts_kwargs)
    p_ts.add_layout(Title(text="GPS Alt [km]", align="center"), "left")
    p_ts.line('time', 'alt', source=orig_src, color='black', line_width=2, alpha=0.7)
    p_ts.add_tools(HoverTool(tooltips=[('Time', '@time{%H:%M:%S}'), ('Altitude', '@alt{0.00} km')], formatters={'@time': 'datetime'}, mode='vline'))
    format_ticks(p_ts)

    # 2. WIC Std
    p_ts_wic = figure(height=130, x_range=p_ts.x_range, **ts_kwargs)
    p_ts_wic.add_layout(Title(text=f"WIC {window_n}p Std", align="center"), "left")
    p_ts_wic.line('time', 'wic_std', source=orig_src, color='black', line_width=2, alpha=0.7)
    p_ts_wic.add_tools(HoverTool(tooltips=[('Time', '@time{%H:%M:%S}'), ('WIC Std', '@wic_std{0.00} m/s')], formatters={'@time': 'datetime'}, mode='vline'))
    format_ticks(p_ts_wic)

    # 3. THETA (Aircraft Theta vs SKT Theta)
    p_ts_theta = figure(height=150, x_range=p_ts.x_range, **ts_kwargs)
    p_ts_theta.add_layout(Title(text="Theta [K]", align="center"), "left")
    p_ts_theta.line('time', 'theta', source=orig_src, color='black', line_width=2, alpha=0.7, legend_label='Aircraft Theta')
    p_ts_theta.line('time', 'theta_skt', source=orig_src, color='red', line_width=2, alpha=0.7, line_dash='dashed', legend_label='Surface Theta')
    p_ts_theta.legend.location = "bottom_right"
    p_ts_theta.legend.click_policy = "hide"
    p_ts_theta.add_tools(HoverTool(tooltips=[('Time', '@time{%H:%M:%S}'), ('Air Theta', '@theta{0.00} K'), ('Surf Theta', '@theta_skt{0.00} K')], formatters={'@time': 'datetime'}, mode='vline'))
    format_ticks(p_ts_theta)

    # 4. e (Water Vapor Pressure)
    p_ts_e = figure(height=130, x_range=p_ts.x_range, **ts_kwargs)
    p_ts_e.add_layout(Title(text="e [hPa]", align="center"), "left")
    p_ts_e.line('time', 'e', source=orig_src, color='black', line_width=2, alpha=0.7)
    p_ts_e.add_tools(HoverTool(tooltips=[('Time', '@time{%H:%M:%S}'), ('e', '@e{0.00} hPa')], formatters={'@time': 'datetime'}, mode='vline'))
    format_ticks(p_ts_e)

    # Highlight lines for all time series
    for i in range(max_highlights):
        p_ts.line('time', 'alt', source=h_sources[i], color=palette[i], line_width=4, alpha=0.8)
        p_ts_wic.line('time', 'wic_std', source=h_sources[i], color=palette[i], line_width=4, alpha=0.8)
        p_ts_theta.line('time', 'theta', source=h_sources[i], color=palette[i], line_width=4, alpha=0.8)
        p_ts_e.line('time', 'e', source=h_sources[i], color=palette[i], line_width=4, alpha=0.8)

    # ---- Profile Plots ----
    prof_w, prof_h = 333, 450
    p_theta = figure(width=prof_w, height=prof_h, title="Theta Profile", tools=prof_tools)
    p_theta.add_layout(Title(text="Height [km]", align="center"), "left")
    p_theta.add_layout(Title(text="Potential Temp [K]", align="center"), "below")
    
    p_e = figure(width=prof_w, height=prof_h, title="e Profile", y_range=p_theta.y_range, tools=prof_tools)
    p_e.add_layout(Title(text="Height [km]", align="center"), "left")
    p_e.add_layout(Title(text="e [hPa]", align="center"), "below")

    # Air Theta
    p_theta.line('theta', 'alt', source=active_src, color="black", width=2, legend_label="Air")
    p_theta.circle('theta', 'alt', source=active_src, color="black", size=3, alpha=0.5, legend_label="Air")
    
    # SST Theta (Anchored to surface)
    p_theta.line('theta_skt', 'surf_alt', source=active_src, color="red", width=2, line_dash='dashed', legend_label="Surface")
    p_theta.circle('theta_skt', 'surf_alt', source=active_src, color="red", size=3, alpha=0.5, legend_label="Surface")
    
    p_theta.legend.location = "bottom_right"
    p_theta.legend.click_policy = "hide"
    
    p_e.line('e', 'alt', source=active_src, color="black", width=2)
    p_e.circle('e', 'alt', source=active_src, color="black", size=3, alpha=0.5)

    # ---- Callbacks ----
    zoom_callback_code = """
    const start = xr.start; const end = xr.end;
    const orig = orig_src.data; const target = active_src.data;
    const t = orig['time']; const keys = Object.keys(orig);
    
    const new_data = {}; for (let k of keys) new_data[k] = [];
    
    for (let i = 0; i < t.length; i++) {
        if (t[i] >= start && t[i] <= end) {
            for (let k of keys) new_data[k].push(orig[k][i]);
        }
    }
    for (let k of keys) target[k] = new_data[k];
    active_src.change.emit();
    """
    js_zoom_callback = CustomJS(args=dict(xr=p_ts.x_range, orig_src=orig_src, active_src=active_src), code=zoom_callback_code)
    p_ts.x_range.js_on_change('start', js_zoom_callback)
    p_ts.x_range.js_on_change('end', js_zoom_callback)

    instruction_div = Div(text="<b style='color:#2ca02c; font-size:16px;'>Step 1: Click anywhere on a time-series plot to select the START time.</b>", width=550)
    instruction_div.tags = [0, None, None]
    
    record_div = Div(text="<b style='font-size:14px;'>Python Code (Copy/Paste this into a new cell!):</b><br><div style='background-color:#f0f0f0; padding:10px; border:1px solid #ccc; font-family:monospace; margin-top:5px;'>bl_periods = []</div>", width=500)
    delete_select = Select(title="Select Period to Delete:", value="-1", options=[("-1", "None")], width=200)
    delete_btn = Button(label="Delete Period", button_type="danger", width=100, margin=(23, 0, 0, 10))
    clear_all_btn = Button(label="Cancel Click / Clear All Highlights", button_type="warning", width=250)
    print_btn = Button(label="Print Periods to Notebook", button_type="primary", width=200)

    state_change_code = """
    const data = record_state.data; const n = data['start_ms'].length;
    let py_list = "bl_periods = [<br>"; let raw_list = []; let options = [["-1", "None"]];
    
    for (let i = 0; i < n; i++) {
        const c = colors[i]; const s_str = data['start_str'][i]; const e_str = data['end_str'][i];
        py_list += "&nbsp;&nbsp;&nbsp;&nbsp;['" + s_str + "', '" + e_str + "'], <span style='color:" + c + "; font-weight:bold;'># Highlight " + (i+1) + "</span><br>";
        raw_list.push([s_str, e_str]);
        options.push([i.toString(), s_str + " to " + e_str]);
    }
    py_list += "]";
    rec_div.text = "<b style='font-size:14px;'>Python Code:</b><br><div style='background-color:#f0f0f0; padding:10px; border:1px solid #ccc; font-family:monospace; margin-top:5px;'>" + py_list + "</div>";
    
    try {
        if (window.IPython && window.IPython.notebook && window.IPython.notebook.kernel) {
            const py_code = "bl_periods = " + JSON.stringify(raw_list);
            window.IPython.notebook.kernel.execute(py_code);
        }
    } catch(err) {}
    
    del_select.options = options; del_select.value = "-1";
    for (let i = 0; i < h_sources.length; i++) {
        const h_data = h_sources[i].data;
        for (let k in h_data) h_data[k] = [];
    }
    
    const orig = orig_src.data; const keys = Object.keys(orig); const t = orig['time'];
    for (let i = 0; i < n; i++) {
        const t1 = data['start_ms'][i]; const t2 = data['end_ms'][i]; const h_data = h_sources[i].data;
        for (let j = 0; j < t.length; j++) {
            if (t[j] >= t1 && t[j] <= t2) { for (let k of keys) h_data[k].push(orig[k][j]); }
        }
        h_sources[i].change.emit();
    }
    for (let i = n; i < h_sources.length; i++) h_sources[i].change.emit();
    """
    js_state_change = CustomJS(args=dict(record_state=record_state, rec_div=record_div, del_select=delete_select, h_sources=h_sources, orig_src=orig_src, colors=palette), code=state_change_code)
    record_state.js_on_change('data', js_state_change)

    tap_callback_code = """
    const x = cb_obj.x; if (x == null) return;
    const d = new Date(x); function pad(n) { return (n < 10 ? '0' : '') + n; }
    const time_str = pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()) + ":" + pad(d.getUTCSeconds());
    let state = inst_div.tags[0];
    
    if (state === 0) {
        inst_div.tags = [1, x, time_str];
        inst_div.text = "<b style='color:#d62728; font-size:16px;'>Step 2: Start time set to " + time_str + ". Now click to select the END time.</b>";
    } else {
        const start_ms = inst_div.tags[1]; const start_str = inst_div.tags[2];
        const end_ms = x; const end_str = time_str;
        let t1_ms = start_ms, t2_ms = end_ms, t1_str = start_str, t2_str = end_str;
        if (start_ms > end_ms) { t1_ms = end_ms; t2_ms = start_ms; t1_str = end_str; t2_str = start_str; }
        
        const data = record_state.data;
        if (data['start_ms'].length >= max_hl) {
            alert("Maximum highlights reached."); inst_div.tags = [0, null, null];
            inst_div.text = "<b style='color:#2ca02c; font-size:16px;'>Step 1: Click to select START time.</b>"; return;
        }
        record_state.data = { 'start_ms': [...data['start_ms'], t1_ms], 'end_ms': [...data['end_ms'], t2_ms], 'start_str': [...data['start_str'], t1_str], 'end_str': [...data['end_str'], t2_str] };
        inst_div.tags = [0, null, null]; inst_div.text = "<b style='color:#2ca02c; font-size:16px;'>Step 1: Click anywhere on a time-series plot to select the START time.</b>";
    }
    """
    js_tap_callback = CustomJS(args=dict(inst_div=instruction_div, record_state=record_state, max_hl=max_highlights), code=tap_callback_code)
    
    for p in [p_ts, p_ts_wic, p_ts_theta, p_ts_e]:
        p.js_on_event('tap', js_tap_callback)

    delete_btn.js_on_event('button_click', CustomJS(args=dict(record_state=record_state, del_select=delete_select), code="""
        const idx = parseInt(del_select.value); if (idx === -1 || isNaN(idx)) return;
        const data = record_state.data;
        const n_ms = [], n_e_ms = [], n_s_str = [], n_e_str = [];
        for (let i = 0; i < data['start_ms'].length; i++) {
            if (i !== idx) { n_ms.push(data['start_ms'][i]); n_e_ms.push(data['end_ms'][i]); n_s_str.push(data['start_str'][i]); n_e_str.push(data['end_str'][i]); }
        }
        record_state.data = { 'start_ms': n_ms, 'end_ms': n_e_ms, 'start_str': n_s_str, 'end_str': n_e_str };
    """))

    clear_all_btn.js_on_event('button_click', CustomJS(args=dict(inst_div=instruction_div, record_state=record_state), code="""
        inst_div.tags = [0, null, null]; inst_div.text = "<b style='color:#2ca02c; font-size:16px;'>Step 1: Click anywhere on a time-series plot to select the START time.</b>";
        record_state.data = { 'start_ms': [], 'end_ms': [], 'start_str': [], 'end_str': [] };
    """))

    print_btn.js_on_event('button_click', CustomJS(args=dict(record_state=record_state), code="""
        const data = record_state.data; let raw_list = [];
        for (let i = 0; i < data['start_ms'].length; i++) raw_list.push([data['start_str'][i], data['end_str'][i]]);
        try {
            if (window.IPython && window.IPython.notebook && window.IPython.notebook.kernel) {
                window.IPython.notebook.kernel.execute("bl_periods = " + JSON.stringify(raw_list) + "\\nprint('--- Boundary Layer Periods ---')\\nfor p in bl_periods:\\n    print(p)");
                alert("Success! Periods have been sent to Python and printed below your cell.");
            } else alert("Kernel execution not supported here. Please copy the list from the gray box.");
        } catch(err) { alert("Could not communicate with Python kernel."); }
    """))

    # 6. Display Layout 
    controls_layout = row(record_div, column(row(delete_select, delete_btn), margin=(0, 0, 0, 40)), margin=(10, 0, 0, 0))
    layout = column(
        p_ts, p_ts_wic, p_ts_theta, p_ts_e, 
        row(p_map, p_theta, p_e),
        row(instruction_div, print_btn, clear_all_btn, margin=(20, 0, 0, 0)),
        controls_layout 
    )
    show(layout)
