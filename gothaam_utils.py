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
             'THETA','EW_VXL','WSC','WDC'
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

def open_nc(data_dir):
    # get file names
    # only partial data on return ff's, so those are excluded. Only included the first three
    #ffnames = sorted(["CAESARff01.nc", "CAESARff02.nc", "CAESARff03.nc", "CAESARff04.nc", "CAESARff05.nc", "CAESARff06.nc"])
    ppnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*PP??.nc")])
    ffnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*ff??.nc")])
    tfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*tf??.nc")])
    rfnames = sorted([fname for fname in os.listdir(data_dir) if fnmatch(fname, "*rf??.nc")])
    allfnames = ppnames + ffnames + tfnames + rfnames
    allfnames = sorted(allfnames)
    
    print(f"Found {len(ppnames)} proficiency flights, {len(ffnames)} ferry flights, {len(tfnames)} test flights, and {len(rfnames)} research flights")
    print("Opening all flight NetCDF Files")
    nc_dict = {}
    for fname in allfnames:
        stem = fname.split('.')[0]
        if stem[-1] == "h":
            flname = stem[-5:-1]
    
        else:
            flname = stem[-4:]
    
        try:
            nc_dict[flname] = netCDF4.Dataset(data_dir + "/" + fname)
    
        except Exception as e:
            print(f"Could not read {fname} netcdf.")
            print(e)
    
    # try to get global attributes from the netcdf file if they are present
    # determine preliminary or final status
    try:
        proc_status = nc_dict['tf01'].getncattr('WARNING')
        print(proc_status)
    except:
        proc_status = 'final'
    
    # determine the NIDAS version
    try:
        nidas = nc_dict['tf01'].getncattr('NIDASrevision')
        print('NIDAS version: ' + nidas)
    except Exception as e:
        print(e)
    
    # determine the NIMBUS version
    try:
        nimbus = nc_dict['tf01'].getncattr('RepositoryRevision')
        print('NIMBUS version: ' + nimbus)
    except Exception as e:
        print(e)
    
    # determine the processing date and time
    try:
        proc_date = nc_dict['tf01'].getncattr('date_created')
        print('Processing Date & Time: ' + proc_date)
    except Exception as e:
        print(e)

    return nc_dict

def read_nc(nc: netCDF4._netCDF4.Dataset):
    # sometimes the netcdf4 api produces an issue with big-endian buffer on little-endian compiler
    byte_swap = False
    
    # create empty placeholders for asc, histo_asc and units
    data = {}
    units = {}
    
    # use the netcdf4 api to get the netcdf data into a dataframe
#    try:
        
    # loop over keys in netCDF file and organize
    #for i in nc.variables.keys():
    for i in read_vars:
        try:
            output = nc[i][:]
            data[i] = pd.DataFrame(output)
            units_var = nc.variables[i].getncattr('units')
            units[i] = pd.Series(units_var)
            data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

        except Exception as e:
            print(e)

    # add times
    i = 'Time'
    output = nc[i][:]
    data[i] = pd.DataFrame(output)
    units_var = nc.variables[i].getncattr('units')
    units[i] = pd.Series(units_var)
    data[i].columns = pd.MultiIndex.from_tuples(zip(data[i].columns, units[i]))

    # concatenate the dataframe
    data = pd.concat(data, axis=1, ignore_index=False)
    # clean up the dataframe by dropping some of the multi-index rows
    data.columns = data.columns.droplevel(1)
    data.columns = data.columns.droplevel(1)

    # add a datetime-type time as well
    data['datetime'] = [timedelta(seconds=int(time)) for time in data['Time']]

    return data

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

# function definition for creating generic timeseries plot
def format_ticks(plot):
    plot.xaxis.formatter=DatetimeTickFormatter(days ='%h:%m', hours="%h:%m", minutes="%h:%m",hourmin = '%h:%m')

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

def detect_climb_descent(df: pd.DataFrame):

    profiles = []

    vspd = df['GGVSPD'].to_numpy()
    roll = df['ROLL'].to_numpy()
    t_sfm = df['Time'].to_numpy()
    in_climb = False
    in_descent = False

    # tuning params to define climbs and descents
    ascent_thresh = 2.5 # m/s
    ascent_stop_thresh = 0.5 # m/s
    descent_thresh = -2.5 # m/s
    descent_stop_thresh = -0.5 # m/s
    min_period = 90 # minimum lenght of climb or descent to count


    #  iterate through and find ascents
    for i in range(len(t_sfm)):
        if not in_climb:
            #if vspd[i] >= ascent_thresh and abs(roll[i]) <= 5:
            if vspd[i] >= ascent_thresh:
                in_climb = True
                st_time = int(t_sfm[i])
        else:
            #if vspd[i] < ascent_stop_thresh or abs(roll[i]) > 5:
            if vspd[i] < ascent_stop_thresh:
                in_climb = False
                sp_time = int(t_sfm[i])
                if sp_time - st_time > min_period:
                    profiles.append([st_time,sp_time])

     #  iterate through and find descents
    for i in range(len(t_sfm)):
        if not in_descent:
            #if vspd[i] <= descent_thresh and abs(roll[i]) <= 5:
            if vspd[i] <= descent_thresh:
                in_descent = True
                st_time = int(t_sfm[i])
        else:
            #if vspd[i] > descent_stop_thresh or abs(roll[i]) > 5:
            if vspd[i] > descent_stop_thresh:
                in_descent = False
                sp_time = int(t_sfm[i])
                if sp_time - st_time > min_period:
                    profiles.append([st_time,sp_time])
        
    return profiles


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
