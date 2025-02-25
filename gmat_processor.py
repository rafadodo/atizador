# -*- coding: utf-8 -*-
"""
"""
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load script configuration
with open('gmat_reader_config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

# Plot settings
PLOT_FLUX_GRAPHS = config['plot_flux_graphs']
PLOT_VECTOR_ANIMATION = config['plot_vector_animation']

# Run configuration
ATTITUDE = config['attitude']
POSITIONS_FILENAME = config['positions_filename_template'].format(attitude=ATTITUDE)
ECLIPSE_FILENAME = config['eclipse_filename_template'].format(attitude=ATTITUDE)
SURFACE_NORMALS = config['surface_normals']

# Constants
SOLAR_RADIATION = config['solar_radiation']
INFRARED_RADIATION = config['infrared_radiation']
ALBEDO_FRACTION = config['albedo_fraction']

# Time formatting
TIME_FORMAT = config['time_format']

# Load GMAT position data
gmat_data_df = pd.read_csv(POSITIONS_FILENAME)
time_date_objs = [datetime.strptime(date_str, TIME_FORMAT) for date_str in gmat_data_df['DefaultSC.A1Gregorian']]
time = pd.DataFrame({'time': time_date_objs})
sc_position = gmat_data_df[['DefaultSC.EarthMJ2000Eq.X', 'DefaultSC.EarthMJ2000Eq.Y', 'DefaultSC.EarthMJ2000Eq.Z']].to_numpy()
sun_position = gmat_data_df[['Sun.EarthMJ2000Eq.X', 'Sun.EarthMJ2000Eq.Y', 'Sun.EarthMJ2000Eq.Z']].to_numpy()
sc_attitude = gmat_data_df[['DefaultSC.Q1', 'DefaultSC.Q2', 'DefaultSC.Q3', 'DefaultSC.Q4']].to_numpy()

# Obtain spacecraft rotated normals for each time step
rotations = R.from_quat(sc_attitude)
rotated_normals= {}
for label, normal in SURFACE_NORMALS.items():
    rotated_normals[label] = rotations.apply(normal)

# Organize into a dictionary
# surface_normals_dict = {label: rotated_normals[:, idx, :] for idx, label in enumerate(SURFACE_LABELS)}

# Load GMAT eclipse data
eclipse_data = pd.read_fwf(ECLIPSE_FILENAME, skiprows=2, skipfooter=7)
time_merged = time.merge(eclipse_data, how='cross')
time_merged['Eclipse'] = ((time_merged['time'] > time_merged['Start Time (UTC)']) & (time_merged['time'] < time_merged['Stop Time (UTC)']))
time_umbra_tagged = time_merged.groupby('time').agg({'Eclipse':any}).reset_index()

# Compute sun vectors
sun_vectors = sc_position + sun_position

# Compute incident heat fluxes
all_fluxes = {}
output_fluxes = {}
for surface, normals in rotated_normals.items():
    sun_dot_product = np.einsum('ij,ij->i', sun_vectors, normals) / np.linalg.norm(sun_vectors, axis=1)
    earth_dot_product = np.einsum('ij,ij->i', sc_position, normals) / np.linalg.norm(sc_position, axis=1)

    sun_flux = SOLAR_RADIATION * np.maximum(sun_dot_product, 0)
    albedo_flux = ALBEDO_FRACTION * SOLAR_RADIATION * np.maximum(earth_dot_product, 0)
    infrared_flux = INFRARED_RADIATION * np.maximum(earth_dot_product, 0)
    surf_fluxes = {'sun_flux': sun_flux,
                   'albedo_flux': albedo_flux,
                   'infrared_flux': infrared_flux
                   }
    # Zero-out solar-related fluxes during eclipse
    flux_df = pd.concat([time_umbra_tagged, pd.DataFrame(surf_fluxes)], axis=1)
    flux_df[['sun_flux', 'albedo_flux']] = flux_df[['sun_flux', 'albedo_flux']].mask(flux_df['Eclipse']==True, 0)
    flux_df['total_flux'] = flux_df[['sun_flux', 'albedo_flux', 'infrared_flux']].sum(axis=1)
    all_fluxes[surface] = flux_df
    output_fluxes[surface] = flux_df['total_flux'] * 1e-6 # W/mm^2

# Prepare csv output files for NX
elapsed_seconds = [(dt - time_date_objs[0]).total_seconds() for dt in time_date_objs]
for surface in SURFACE_NORMALS.keys():
    data = pd.DataFrame({
        "Time [s]": elapsed_seconds,
        "Heat Flux [W/mm^2]": output_fluxes[surface]
    })
    exit_code = data.to_csv(f"{surface}_heat_flux_table.csv", index=False)
    print(exit_code)

# Optional flux graphs
if PLOT_FLUX_GRAPHS:
    for surface in SURFACE_NORMALS.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, all_fluxes[surface]['total_flux'], color='k', marker='*')
        ax.plot(time, all_fluxes[surface]['sun_flux'], color='r')
        ax.plot(time, all_fluxes[surface]['albedo_flux'], color='b')
        ax.plot(time, all_fluxes[surface]['infrared_flux'], color='g')
        ax.legend(loc='lower right')
        plt.title(f'{surface} Heat Flux')
        plt.ylim([0, np.max(all_fluxes[surface]['total_flux'])])
        plt.xlabel("Time (s)")
        plt.ylabel("Heat Flux (W/m^2)")
        plt.grid()
    plt.show()

# Optional vector animation
if PLOT_VECTOR_ANIMATION:
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    def get_arrow(arr):
        x = 0
        y = 0
        z = 0
        u = arr[0] / np.linalg.norm(arr)
        v = arr[1] / np.linalg.norm(arr)
        w = arr[2] / np.linalg.norm(arr)
        return x,y,z,u,v,w

    sc_quiver = ax.quiver(*get_arrow(-sc_position[0]))
    sun_quiver = ax.quiver(*get_arrow(sun_vectors[0]))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(index):
        global sc_quiver
        global sun_quiver
        sun_quiver.remove()
        sc_quiver.remove()
        sc_quiver = ax.quiver(*get_arrow(-sc_position[index]))
        sun_quiver = ax.quiver(*get_arrow(sun_vectors[index]), color='r')

    ani = FuncAnimation(fig, update, frames=len(sun_vectors), interval=100)
    plt.show()