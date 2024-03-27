
"""
Quantifying the effects of snow cover
=====================================

We classify the effect of snow on a PV system's DC array.

The effect of snow is classified into one of four categories:

    Mode 0 : The system is covered with enough opaque snow that the system is
        offline or its voltage is below the inverter's start-up voltage.
    Mode 1 : The system is online and covered with non-uniform snow, such that
        both operating voltage and current are reduced.
    Mode 2 : The system is online and covered with opaque snow, such that
        operating voltage is reduced by the snow, but transmission
        is consistent with snow-free conditions.
    Mode 3 : The system is online and covered with light-transmissive snow,
        such that current is decreased but voltage is consistent with all
        system substrings being online.
    Mode 4 : Current and voltage are consistent with snow-free conditions.

The procedure involves four steps:
    1. Using measured plane-of-array (POA) irradiance and temperature, model
       the module's maximum power current (Imp) and voltage (Vmp) assuming
       that all the POA irradiance reaches the module's cells.
    2. Use the modeled Imp and measured Imp, determine the fraction of
       plane-of-array irradiance that reaches the module's cells. This fraction
       is called the transmittance.
    3. Model the Vmp that would result from the POA irradiance reduced by
       the transmittance.
    4. Classify the effect of snow using the ratio of modeled Vmp (from step 3)
       and measured Vmp.
We demonstrate this analysis using measurements made at the combiner boxes
for a utility-scale system.

"""

#%% Import packages

import pathlib
import os
import json
import pandas as pd
import numpy as np
import re
import pvlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from pvanalytics.features import clipping

#%% Functions needed for the analysis procedure

# Functions for modeling the transmittance.

def get_irradiance_sapm(temp_cell, i_mp, imp0, c0, c1, alpha_imp,
                        irrad_ref=1000, temp_ref=25):
    """
    Model effective irradiance from current at maximum power and cell
    temperature.

    Solves Eqn. 2 from [1]_.

    Parameters
    ----------
    temp_cell : array
        Temperature of cells inside module. [degrees C]
    i_mp : array
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]
    c0, c1 : float
        Empirically determined coefficients relating ``i_mp`` to effective
        irradiance.
    alpha_imp : float
        Normalized temperature coefficient for short-circuit current. [1/°C]
    temp_ref : float
        Reference cell temperature. [degrees C]

    Returns
    -------
    effective_irradiance : array
        Effective irradiance. [W/m2]

    References
    ----------
    .. [1] D. L. King, E.E. Boyson, and J.A. Kratochvil, Photovoltaic Array
       Performance Model, SAND2004-3535, Sandia National Laboratories,
       Albuquerque, NM, 2004.
    """

    a = imp0*c1*(1 + alpha_imp*(temp_cell - temp_ref))
    b = imp0*c0*(1 + alpha_imp*(temp_cell - temp_ref))
    c = -i_mp
    discriminant = np.square(b) - 4*a*c
    effective_irradiance = (-b + np.sqrt(discriminant))/(2*a) * irrad_ref
    return effective_irradiance


def get_irradiance_imp(i_mp, imp0, irrad_ref=1000):
    """
    Model effective irradiance from maximum power current.

    Assumes a linear relationship between effective irradiance and maximum
    power current, i.e., Eqn. 8 from [1]_.

    Parameters
    ----------
    i_mp : array
        Maximum power current at the resolution of a single module. [A]
    imp0 : float
        Short-circuit current at reference condition. [A]

    Returns
    -------
    effective_irradiance : array
        Effective irradiance. [W/m2]

    References
    ----------
    .. [1] C. F. Abe, J. B. Dias, G. Notton, G. A. Faggianelli, G. Pigelet, and
       D. Ouvrard, David, Estimation of the effective irradiance and bifacial
       gain for PV arrays using the maximum power current, IEEE Journal of
       Photovoltaics, 2022.
    """
    return i_mp / imp0 * irrad_ref


def get_transmission(measured_e_e, modeled_e_e, i_mp):
    
    """
    Estimate transmittance as the ratio of modeled effective irradiance to
    measured irradiance.

    Measured irradiance should be in the array's plane and represent snow-free
    conditions. When possible, the irradiance should be adjusted for
    reflections and spectral content. For example, the measured irradiance
    could be obtained with a heated plane-of-array pyranometer.

    Parameters
    ----------
    measured_e_e : array
        Irradiance absent the effect of snow. [W/m2]
    modeled_e_e : array
        Effective irradiance modeled from measured current at maximum power.
        [W/m2]
    i_mp : array
        Maximum power current at the resolution of a single module. [A]

    Returns
    -------
    T : array
        Effective transmission. [unitless]

    References
    ----------
    .. [1] E. C. Cooper, J. L. Braid and L. Burnham, "Identifying the
       Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
       50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA,
       2023, pp. 1-5. doi:`10.1109/PVSC48320.2023.10360065`
    """
    
    T = modeled_e_e/measured_e_e
    T[T.isna()] = np.nan
    T[i_mp == 0] = 0
    T[T < 0] = np.nan
    T[T > 1] = 1

    return T


# %% Function for categorizing snow coverage

def categorize(vmp_ratio, transmission, voltage, min_dcv,
               threshold_vratio, threshold_transmission):
    
    """
    Categorizes electrical behavior into a snow-related or snow-free "mode"
    as defined in [1].

    Mode 0 = system is covered with enough opaque snow that the system is
    offline or its voltage is below the inverter's MPPT turn-on voltage
    Mode 1 = system is online and covered with non-uniform snow, such that
    both operating voltage and current are decreased by the presence of snow
    Mode 2 = system is online and covered with opaque snow, such that
    operating voltage is decreased by the presence of snow, but transmission
    is consistent with snow-free conditions
    Mode 3 = system is online and covered with light-transmissive snow, such
    that transmission is decreased but voltage is consistent with all
    system substrings being online
    Mode 4 = transmisison and voltage are consistent with snow-free conditions

    Parameters
    ----------
    vratio : float
        Ratio between measured voltage and voltage modeled using
        calculated values of transmission [dimensionless]
    transmission : float
        Fraction of irradiance measured by an onsite pyranometer that the
        array is able to utilize [dimensionless]
    voltage : float
        [V]
    min_dcv : float
        The lower voltage bound on the inverter's maximum power point
        tracking (MPPT) algorithm. [V]
    threshold_vratio : float
        The lower bound on vratio that is found under snow-free conditions,
        determined empirically.
    threshold_transmission : float
        The lower bound on transmission that is found under snow-free
        conditions, determined empirically.

    Returns
    -------
    mode : int

    [1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
    Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
    50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA, 2023,
    pp. 1-5, doi: 10.1109/PVSC48320.2023.10360065.
    """
    
    if np.isnan(vmp_ratio) or np.isnan(transmission):
        return np.nan
    elif voltage < min_dcv:
        return 0
    elif vmp_ratio < threshold_vratio:
        if transmission < threshold_transmission:
            return 1
        elif transmission > threshold_transmission:
            return 2
    elif vmp_ratio > threshold_vratio:
        if transmission < threshold_transmission:
            return 3
        elif transmission > threshold_transmission:
            return 4
    return np.nan

# %% Load in system configuration parameters (dict)

base_dir = pathlib.Path('.')
data_dir = os.path.join(base_dir, 'data')
data_path = os.path.join(data_dir, 'data.csv')
snow_path = os.path.join(data_dir, 'snow.csv')
mask_path = os.path.join(data_dir, 'mask.csv')
config_path = os.path.join(data_dir, 'config.json')

with open(config_path) as json_data:
    config = json.load(json_data)

print(f"Inverter AC power rating: {config['max_ac']} kW")
print(f"Inverter MPPT range: {config['min_dcv']} V - {config['max_dcv']} V")
num_str_per_cb = config['num_str_per_cb']['INV1 CB1']
num_mods_per_str = config['num_mods_per_str']['INV1 CB1']
print(f"There are {num_str_per_cb} modules connected in series in each string,"
      f" and there are {num_mods_per_str} strings connected in"
      f" parallel at each combiner")

#%%
# Read in 15-minute sampled DC voltage and current time series data, AC power,
# module temperature collected by a BOM sensor and plane-of-array
# irradiance data collected by a heated pyranometer. This sample is provided
# by an electric utility.

# Load in utility data
data = pd.read_csv(data_path, index_col='Timestamp')
data.set_index(pd.DatetimeIndex(data.index,ambiguous='infer'), inplace=True)
data = data[~data.index.duplicated()]

# Explore utility datatset
print('Utility-scale dataset')
print('Start: {}'.format(data.index[0]))
print('End: {}'.format(data.index[-1]))
print('Frequency: {}'.format(data.index.inferred_freq))
print('Columns : ' + ', '.join(data.columns))
data.between_time('8:00', '16:00').head()

# Identify current, voltage, and AC power columns
dc_voltage_cols = [c for c in data.columns if 'Voltage' in c]
dc_current_cols = [c for c in data.columns if 'Current' in c]
ac_power_cols = [c for c in data.columns if 'AC' in c]

# Set negative or Nan current, voltage, AC power values to zero. This is
# allows us to calculate losses later.

data.loc[:, dc_voltage_cols] = np.maximum(data[dc_voltage_cols], 0)
data.loc[:, dc_current_cols] = np.maximum(data[dc_current_cols], 0)
data.loc[:, ac_power_cols] = np.maximum(data[ac_power_cols], 0)

data[dc_voltage_cols] = data[dc_voltage_cols].replace({np.nan: 0, None: 0})
data[dc_current_cols] = data[dc_current_cols].replace({np.nan: 0, None: 0})
data.loc[:, ac_power_cols] = data[ac_power_cols].replace({np.nan: 0, None: 0})

# %% Plot DC voltage for each combiner input relative to inverter nameplate limits
fig, ax = plt.subplots(figsize=(10,10))                  
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for v in dc_voltage_cols:
    ax.scatter(data.index, data[v], s=0.5, label=v)
    ax.plot(data.index, data[v], alpha=0.2)
ax.axhline(float(config['max_dcv']), c='r', ls='--',
           label='Maximum MPPT voltage: {} V'.format(config['max_dcv']))
ax.axhline(float(config['min_dcv']), c='g', ls='--',
           label='Minimum MPPT voltage: {} V'.format(config['min_dcv']))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('Voltage [V]', fontsize='large')
ax.legend(loc='lower left')

# %% Plot AC power relative to inverter nameplate limits

fig, ax = plt.subplots(figsize=(10,10))                  
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for a in ac_power_cols:
    ax.scatter(data.index, data[a], s=0.5, label=a)
    ax.plot(data.index, data[a], alpha=0.2)
ax.axhline(float(config['max_ac']), c='r', ls='--',
           label='Maximum allowed AC power: {} kW'.format(config['max_ac']))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('AC Power [kW]', fontsize='large')
ax.legend(loc='upper left')

# %% Filter data.
# Identify periods where the system is operating off of its maximum power
# point (MPP), and correct or mask. Conditions outside of the MPP cannot
# be accurately modeled without external information on the system's
# operating point. To allow us to make a valid comparison between system
# measurements and modeled power at MMP, we set measurements collected below
# the MPPT minimum voltage to zero, which emulates the condition where the
# inverter turns off when it cannot meet the turn-on voltage. When the inverter
# is clipping power, we replace voltage and current measurements with NaN as
# these measurements reflect current and voltage that has been artificially
# adjusted away from the MMP. This masking may result in an omission of some
# snow loss conditions where a very light-transmissive snow cover allows the
# system to reach the inverter's clipping voltage.

ac_power_cols_repeated = ac_power_cols + ac_power_cols + ac_power_cols
for v, i, a in zip(dc_voltage_cols, dc_current_cols, ac_power_cols_repeated):

    # Data where V > MPPT maximum
    data.loc[(data[v] > float(config['max_dcv'])), v] = np.nan
    data.loc[(data[v] > float(config['max_dcv'])), i] = np.nan
    data.loc[(data[v] > float(config['max_dcv'])), a] = np.nan
    
    # Data where V < MPPT minimum
    data.loc[data[v] < float(config['min_dcv']), v] = 0
    data.loc[data[v] < float(config['min_dcv']), i] = 0
    data.loc[data[v] < float(config['min_dcv']), a] = 0
    
    # Data where system is at Voc
    data.loc[data[i] == 0, v] = 0

    # Data where inverter is clipping based on AC power
    mask1 = data[a] > float(config['max_ac'])
    mask2 = clipping.geometric(ac_power=data[a], freq='15T')
    mask3 = np.logical_or(mask1.values, mask2.values)

    data.loc[mask3, v] = np.nan
    data.loc[mask3, i] = np.nan
    data.loc[mask3, a] = np.nan

# %% Plot DC voltage for each combiner inputm with inverter nameplate limits

fig, ax = plt.subplots(figsize=(10,10))                  
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
for v in dc_voltage_cols:
    ax.scatter(data.index, data[v], s=0.5, label=v)
    ax.plot(data.index, data[v], alpha=0.2)
ax.axhline(float(config['max_dcv']), c='r', ls='--',
           label='MPPT maximum: {} V'.format(config['max_dcv']))
ax.axhline(float(config['min_dcv']), c='g', ls='--',
           label='MPPT minimum: {} V'.format(config['min_dcv']))
ax.set_xlabel('Date', fontsize='large')
ax.set_ylabel('Voltage [V]', fontsize='large')
ax.legend(loc='lower left')

# %% We want to exclude periods where array voltage is affected by horizon
# shading
'''
Load in and apply horizon profiling created using approach described in [1]

[1] J. L. Braid and B. G. Pierce, "Horizon Profiling Methods for Photovoltaic
Arrays," 2023 IEEE 50th Photovoltaic Specialists Conference (PVSC),
San Juan, PR, USA, 2023, pp. 1-7. doi:`10.1109/PVSC48320.2023.10359914`

'''

horizon_mask = pd.read_csv(mask_path, index_col='Unnamed: 0')

def apply_mask(mask, x ,y):
    if np.isnan(x) == False:
        if y > mask.at[int(np.floor(x)), '0']:
            return False
        else:
            return True
    else:
        return np.nan

data.loc[:, 'Horizon Mask'] = data.apply(lambda x: apply_mask(
    horizon_mask, x['azimuth'], x['elevation']), axis = 1)
data = data[data['Horizon Mask'] == False]

# %% 

# Define coefficients for modeling transmission and voltage. User can either
# use the SAPM to calculate transmission or an approach based on the ratio
# between measured current and nameplate current. For modeling voltage, the
# user can use the SAPM or a single diode equivalent.

sapm_coeffs = config['sapm_coeff']
cec_module_db = pvlib.pvsystem.retrieve_sam('cecmod')
sde_coeffs = cec_module_db["REC_Solar_REC340TP_72_BLK"]

# %%
"""
Model cell temperature using procedure outlined in Eqn. 12 of [1]
and model effective irradiance using Eqn. 23 of [2].

[1] D. L. King, E.E. Boyson, and J.A. Kratochvil, Photovoltaic Array
Performance Model, SAND2004-3535, Sandia National Laboratories,
Albuquerque, NM, 2004.
[2] B. H. King, C. W. Hansen, D. Riley, C. D. Robinson and L. Pratt,
“Procedure to Determine Coefficients for the Sandia Array Performance
Model (SAPM)," SAND2016-5284, June 2016.
"""

irrad_ref = 1000
data['Cell Temp [C]'] = data['Module Temp [C]'] + \
    3*data['POA [W/m²]']/irrad_ref

# %% Plot cell temperature
fig, ax = plt.subplots(figsize=(10,10))                        
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
ax.scatter(data.index, data['Cell Temp [C]'], s=0.5, c='b')
ax.plot(data.index, data['Cell Temp [C]'], alpha=0.3, c='b')
ax.set_ylabel('Cell Temp [C]', c='b', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='xx-large')

# %% For one combiner, demonstrate the transmission calculation using two 
# different approaches to modeling effective irradiance from measured Imp.

# Choose one combiner box
j = 0

# Get key for configuration dict
matched = re.match(r'INV(\d+) CB(\d+)', dc_current_cols[j])
inv_cb = matched.group(0)

# Number of strings connected in parallel to combiner.
# Used to scale measured current down to the resolution
# of a single string connected in series, which should
# be the same current as a single module.

i_scaling_factor = int(config['num_str_per_cb'][f'{inv_cb}'])
# String current
imp = data[dc_current_cols[j]] / i_scaling_factor

# Approach 1 using SAPM
modeled_e_e1 = get_irradiance_sapm(data['Cell Temp [C]'],
                                   imp,
                                   sapm_coeffs['Impo'], sapm_coeffs['C0'],
                                   sapm_coeffs['C1'], sapm_coeffs['Aimp'])

T1 = get_transmission(data['POA [W/m²]'], modeled_e_e1, imp)

# Approach 2 using a linear irradiance-Imp model
modeled_e_e2 = get_irradiance_imp(imp, sapm_coeffs['Impo'])

T2 = get_transmission(data['POA [W/m²]'].values, modeled_e_e2, imp)

# %% 
# Plot transmission calculated using two different approaches

fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d \n%H:%M")
ax.xaxis.set_major_formatter(date_form)

ax.scatter(T1.index, T1, s=0.5, c='b', label='SAPM')
ax.plot(T1.index, T1, alpha=0.3, c='b')

ax.scatter(T2.index, T2, s=0.3, c='g', label='Linear model')
ax.plot(T2.index, T2, alpha=0.3, c='g')

ax.legend()
ax.set_ylabel('Transmission', fontsize='xx-large')
ax.set_xlabel('Date + Time', fontsize='large')

# %% 
# Model voltage using calculated transmission (two different approaches)

# Number of modules in series in a string
v_scaling_factor = int(config['num_mods_per_str'][inv_cb])

# Approach 1. Reduce measured POA using the transmission.
modeled_vmp_sapm = pvlib.pvsystem.sapm(data['POA [W/m²]']*T1,
                                       data['Cell Temp [C]'],
                                       sapm_coeffs)['v_mp']
modeled_vmp_sapm *= v_scaling_factor

# Approach 2  %TODO not sure we need this
# Code borrowed from pvlib-python/docs/examples/iv-modeling/plot_singlediode.py

# adjust the reference parameters according to the operating
# conditions using the De Soto model:
IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
    data['POA [W/m²]']*T1,
    data['Cell Temp [C]'],
    alpha_sc=sde_coeffs['alpha_sc'],
    a_ref=sde_coeffs['a_ref'],
    I_L_ref=sde_coeffs['I_L_ref'],
    I_o_ref=sde_coeffs['I_o_ref'],
    R_sh_ref=sde_coeffs['R_sh_ref'],
    R_s=sde_coeffs['R_s'],
   )

# plug the parameters into the SDE and solve for IV curves:
SDE_params = {
    'photocurrent': IL,
    'saturation_current': I0,
    'resistance_series': Rs,
    'resistance_shunt': Rsh,
    'nNsVth': nNsVth
}
modeled_vmp_sde = pvlib.pvsystem.singlediode(**SDE_params)['v_mp']
modeled_vmp_sde *= v_scaling_factor

# %% Plot modeled and measured voltage

fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form)

ax.scatter(modeled_vmp_sapm.index, modeled_vmp_sapm, s=1, c='b', label='SAPM')
ax.scatter(modeled_vmp_sde.index, modeled_vmp_sde, s=1, c='g', label='SDE')

ax.scatter(data.index, data[inv_cb + ' Voltage [V]'], s=1, c = 'r',
           label='Measured')
ax.legend(fontsize='xx-large')
ax.set_ylabel('Voltage [V]', fontsize='xx-large')
ax.set_xlabel('Date', fontsize='large')


# %% Function to do analysis so we can loop over combiner boxes

def wrapper(voltage, current, temp_cell, effective_irradiance,
            coeffs, config, temp_ref=25, irrad_ref=1000):
    
    '''
    Models effective irradiance based on measured current using the SAPM
    calculates transmission, and uses transmission to model voltage with the
    SAPM. Categorizes each data point as mode 0 -4 based on transmission and
    the ratio between measured and modeled votlage.

    Parameters
    ----------
    voltage : array
        Voltage [V] measured at inverter.
    current : array
        Current [A] measured at combiner.
    temp_cell : array
        Cell temperature. [degrees C]
    effective_irradiance : array
        Snow-free POA irradiance measured by a heated pyranometer. [W/m²] 
    coeffs : dict
        A dict defining the SAPM parameters, used for pvlib.pvsystem.sapm.
    config : dict
        min_dcv : float
            The lower voltage bound on the inverter's maximum power point
            tracking (MPPT) algorithm. [V]
        max_dcv : numeric
            Upper bound voltage for the MPPT algorithm. [V]
        threshold_vratio : float
            The lower bound on vratio that is found under snow-free conditions,
            determined empirically
        threshold_transmission : float
            The lower bound on transmission that is found under snow-free
            conditions, determined empirically
        num_mods_per_str : int
            Number of modules in series in each string.
        num_str_per_cb : int
            Number of strings in parallel at the combiner.
    
    Returns
    -------
    my_dict : dict
        Keys are "transmission", "modeled_vmp", "vmp_ratio", and "mode"
    
    '''
    
    # Calculate transmission
    modeled_e_e = get_irradiance_sapm(temp_cell,
                                      current/config['num_str_per_cb'],
                                      coeffs['Impo'], coeffs['C0'],
                                      coeffs['C1'], coeffs['Aimp'])

    T = get_transmission(effective_irradiance, modeled_e_e,
                         current/config['num_str_per_cb'])

    name_T = inv_cb + ' Transmission'
    data[name_T] = T

    # Model voltage for a single module, scale up to array
    modeled_vmp = pvlib.pvsystem.sapm(effective_irradiance*T, temp_cell, 
                                      coeffs)['v_mp']
    modeled_vmp *= config['num_mods_per_str']

    # Voltage is modeled as NaN if T = 0, but V = 0 makes more sense
    modeled_vmp[T == 0] = 0
    
    # Identify periods where modeled voltage is outside of MPPT range,
    # and correct values
    modeled_vmp[modeled_vmp > config['max_dcv']] = np.nan
    modeled_vmp[modeled_vmp < config['min_dcv']] = 0

    # Calculate voltage ratio
    with np.errstate(divide='ignore'):
        vmp_ratio = np.divide(voltage, modeled_vmp,
                              where=((voltage > 0) & (modeled_vmp>0)))
    vmp_ratio[modeled_vmp==0] = 0
    
    categorize_v = np.vectorize(categorize)

    mode = categorize_v(vmp_ratio, T, voltage, config['min_dcv'],
                        config['threshold_vratio'],
                        config['threshold_transmission'])
    my_dict = {'transmission' : T,
               'modeled_vmp' : modeled_vmp,
               'vmp_ratio' : vmp_ratio,
               'mode' : mode}
    
    return my_dict

# %% 
# Demonstrate transmission, modeled voltage calculation and mode categorization
# on voltage, current pair

j = 0
v = dc_voltage_cols[j]
i = dc_current_cols[j]

# Used to get key for configuration dict
matched = re.match(r'INV(\d+) CB(\d+)', i)
inv_cb = matched.group(0)

# Number of strings connected in parallel at the combiner
i_scaling_factor = int(config['num_str_per_cb'][f'{inv_cb}'])

# %TODO: should we make these defaults in categorize()?
threshold_vratio, threshold_t = 0.9331598025404861, 0.5976185185741869

my_config = {'threshold_vratio' : threshold_vratio,
             'threshold_transmission' : threshold_t,
             'min_dcv' : float(config['min_dcv']),
             'max_dcv' : float(config['max_dcv']),
             'num_str_per_cb' : int(config['num_str_per_cb'][f'{inv_cb}']),
             'num_mods_per_str' : int(config['num_mods_per_str'][f'{inv_cb}'])}

out = wrapper(data[v], data[i],
              data['Cell Temp [C]'],
              data['POA [W/m²]'], sapm_coeffs,
              my_config)

# %%
# Calculate the transmission, model the voltage, and categorize into modes for
# all combiners. Use the SAPM to calculate transmission and model the voltage.

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):
    
    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}' 
    
    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])
    i_scaling_factor = int(
        config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])
    
    my_config = {
        'threshold_vratio' : threshold_vratio,
        'threshold_transmission' : threshold_t,
        'min_dcv' : float(config['min_dcv']),
        'max_dcv' : float(config['max_dcv']),
        'num_str_per_cb' : int(config['num_str_per_cb'][f'{inv_cb}']),
        'num_mods_per_str' : int(config['num_mods_per_str'][f'{inv_cb}'])}
    
    out = wrapper(data[v_col], data[i_col],
              data['Cell Temp [C]'],
              data['POA [W/m²]'], sapm_coeffs,
              my_config)
    
    for k, v in out.items():
        data[inv_cb + ' ' + k] = v


# %%
# Look at transmission for all DC inputs
        
transmission_cols = [c for c in data.columns if 'transmission' in c]
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in transmission_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)
ax.set_xlabel('Date', fontsize='xx-large')
ax.legend()

# %%
# Look at voltage ratios for all DC inputs

vratio_cols = [c for c in data.columns if "vmp_ratio" in c]
fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data['2022-01-06 07:45:00': '2022-01-09 17:45:00']

for c in vratio_cols:
    ax.scatter(temp.index, temp[c], s=0.5, label=c)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('Voltage Ratio (measured/modeled)', fontsize='xx-large')
ax.axhline(1, c='k', alpha=0.1, ls='--')
ax.legend();

# %% Calculate all power losses - snow and non-snow 

modeled_df = pvlib.pvsystem.sapm(data['POA [W/m²]'],
                                 data['Cell Temp [C]'],
                                 sapm_coeffs)

for v_col, i_col in zip(dc_voltage_cols, dc_current_cols):
    matched = re.match(r'INV(\d+) CB(\d+) Current', i_col)
    inv_num = matched.group(1)
    cb_num = matched.group(2)
    inv_cb = f'INV{inv_num} CB{cb_num}'
    i_scaling_factor = int(
        config['num_str_per_cb'][f'INV{inv_num} CB{cb_num}'])
    v_scaling_factor = int(config['num_mods_per_str'][inv_cb])

    modeled_power = modeled_df['p_mp']*v_scaling_factor*i_scaling_factor
    name_modeled_power = inv_cb + ' Modeled Power [W]'
    data[name_modeled_power] = modeled_power

    name_loss = inv_cb + ' Loss [W]'
    loss = np.maximum(data[name_modeled_power] - data[i_col]*data[v_col], 0)
    data[name_loss] = loss

# %%
    
snow = pd.read_csv(snow_path, index_col='DATE')

# Plot power losses, color points by mode
# Plot daily snowfall

loss_cols = [c for c in data.columns if "Loss" in c]
mode_cols = [c for c in data.columns if "mode" in c and "modeled" not in c]
modeled_power_cols = [c for c in data.columns if "Modeled Power" in c]

i = 1
l = loss_cols[i]
m = mode_cols[i]
p = modeled_power_cols[i]

# (Green = no snow, Red = snow)
cmap = {0 : 'r',
        1: 'r',
        2: 'r',
        3: 'r',
        4: 'g'}

fig, ax = plt.subplots(figsize=(10,10))                             
date_form = DateFormatter("%m/%d")
ax.xaxis.set_major_formatter(date_form)
temp = data[~data[m].isna()]

# Plot each day individually so we are not exaggerating losses
days_mapped = temp.index.map(lambda x : x.date())
days = np.unique(days_mapped)
grouped = temp.groupby(days_mapped)

for d in days:
        temp_grouped = grouped.get_group(d)
        ax.plot(temp_grouped.index, temp_grouped[p], c='k', alpha=0.2)
        ax.scatter(temp_grouped.index, temp_grouped[p] - temp_grouped[l],
                   c=temp_grouped[m].map(cmap), s=1)
        ax.fill_between(temp_grouped.index, temp_grouped[p] - temp_grouped[l],
                        temp_grouped[p], color='y', alpha=0.4)

# Add different colored points to legend
handles, labels = ax.get_legend_handles_labels()
red_patch = mpatches.Patch(color='r', label='Snow conditions present')
green_patch = mpatches.Patch(color='g', label='No snow present')
yellow_patch = mpatches.Patch(color='y', label='Loss [W]')
handles.append(red_patch) 
handles.append(green_patch)
handles.append(yellow_patch)

ax.set_xlabel('Date', fontsize='xx-large')
ax.set_ylabel('DC Power [W]', fontsize='xx-large')
ax.legend(handles=handles, fontsize='xx-large', loc='upper right')

ax2 = ax.twinx()
ax2.bar(days, snow['SNOW'].values/(10*2.54), color='b', alpha=0.5, width=0.2,
        ec='k')
ax2.set_ylabel('Snowfall [in]', c='b', fontsize='xx-large', alpha=0.5);


# %% 
# Calculate daily snow losses

loss_cols = [c for c in data.columns if "Loss" in c]
mode_cols = [c for c in data.columns if "mode" in c and "modeled" not in c]
modeled_power_cols = [c for c in data.columns if "Modeled Power" in c]

days_mapped = data.index.map(lambda x: x.date())
days = np.unique(days_mapped)
data_gped = data.groupby(days_mapped)

columns = [re.match(r'INV(\d) CB(\d)', c).group(0) for c in loss_cols]

snow_loss = pd.DataFrame(index=days, columns=columns)

for d in days:
    temp = data_gped.get_group(d)

    for c, m, l, p in zip(columns, mode_cols, loss_cols, modeled_power_cols):
        snow_loss_filter = ~(temp[m].isna()) & (temp[m] != 4)
        daily_snow_loss = 100*temp[snow_loss_filter][l].sum()/temp[p].sum()
        snow_loss.at[d, c] = daily_snow_loss


fig, ax = plt.subplots()
date_form = DateFormatter("%m/%d")

days_mapped = data.index.map(lambda x: x.date())
days = np.unique(days_mapped)

xvals = np.arange(0, len(days), 1)
xwidth = 0.05
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, c in enumerate(columns):
    ax.bar(xvals + xwidth*i, snow_loss[c], width=xwidth, color=colors[i],
           ec='k', label=c)

ax.legend()
ax.set_ylabel('Snow loss [%]')
ax.set_xticks(xvals, days)
ax.xaxis.set_major_formatter(date_form);       
