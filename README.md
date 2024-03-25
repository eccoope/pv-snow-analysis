# pv-snow-analysis

This repository hosts an [example analysis](https://github.com/eccoope/pv-snow-analysis/blob/main/notebook.ipynb) that uses the concepts and methods
published in [1], [2] to quantify snow losses and identify the snow conditions
that contribute to losses based on a [utility-scale dataset](https://github.com/eccoope/pv-snow-analysis/blob/main/data/data.csv). Details on the system that produced the datatset (such as module parameters and the number of modules per string) are given in [config.json](https://github.com/eccoope/pv-snow-analysis/blob/main/data/config.json). Each of the system's inverters' has three DC combiner inputs. Voltage and current are measured for each of these inputs, and AC power is measured for each inverter. Daily snowfall measurements are in [snow.csv](https://github.com/eccoope/pv-snow-analysis/blob/main/data/snow.csv) and a horizon shading mask is given in [mask.json](https://github.com/eccoope/pv-snow-analysis/blob/main/data/mask.csv).


**The analysis includes the following steps:** <br>

*Preprocessing*
1. Identify periods where measured voltage exceeds the upper bound of the inverter's MPPT range. During these periods, the system is operating at an unknown location on its IV-curve, and it is not possible to compare measured values to a model. These periods are excluded from analysis by replacing relavant voltage, current, and ac power measurements with NaN values.  
2. Idenitfy periods where the measured voltage is less than the lower bound of the inverter's MPPT range. Again, the system is operating at an unknown location on its IV-curve and we cannot compare measured values to a model. Snow conditions can lower system voltage, and excluding these measurements could introduce bias into snow loss estimates. To account for snow-induced low voltage conditions by making a valid comparison to a model, we set voltage values below the minimum MPPT threshold to zero. By assuming that the system remains off, we emulate how many inverters will turn off when they are unable to meet the minimum bound of the MPPT range. This allows us to make a valid comparison between a model that assumes maximum power point (MPP) operating conditions and a system that is offline. This choice may bias the rate of total outages over the rate of partial outages.
3. Idenitfy periods where the system is operating at an open-ciruit voltage, and set relevant voltages to zero.
4. Identify and exclude periods where the system is clipping based on AC power
5. Identify and exclude periods where the system is persistently shaded by local features using a horizon mask.

*Analysis*
1. Model an effective transmission based on measured current and plane-of-array (POA) irradiance
2. Use effective transmission to model a predicted voltage
3. Categorize periods into "modes" based on effective transmission and the ratio between measured and modeled voltage
4. Calculate differences between measured power and power poredicted using a single diode model; use mode categorization to identify which losses are attributable to snow


[1] E. C. Cooper, J. L. Braid and L. M. Burnham, "Identifying the
    Electrical Signature of Snow in Photovoltaic Inverter Data," 2023 IEEE
    50th Photovoltaic Specialists Conference (PVSC), San Juan, PR, USA, 2023,
    pp. 1-5, doi: 10.1109/PVSC48320.2023.10360065.

[2] E. C. Cooper, L. Burnham, J. L. Braid, "Photovoltaic inverter-based
    quantification of snow conditions and power loss, EPJ Photovoltaics 15, 6 (2024),
    doi: 10.1051/epjpv/2024004.
