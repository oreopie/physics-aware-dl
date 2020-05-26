import os
import pandas as pd
import numpy as np


class DataforIndividual():
    def __init__(self, working_path, basin_id):
        self.working_path = working_path
        self.basin_id = basin_id

    def check_validation(self, basin_list, basin_id):
        assert isinstance(basin_id, str), "The basin ID should be a string"
        assert (len(basin_id) == 8 and basin_id.isdigit()), "Basin ID can only be represented by 8 digits"
        assert (basin_id in basin_list.values), "Please confirm the basin specified is in basin_list.txt"

    def load_forcing_data(self, working_path, huc_id, basin_id):
        forcing_path = os.path.join(working_path, 'camels', 'basin_mean_forcing', 'daymet', huc_id,
                                    basin_id + '_lump_cida_forcing_leap.txt')
        forcing_data = pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
        forcing_data.rename(columns={"Mnth": "Month"}, inplace=True)
        forcing_data['date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day']])
        forcing_data['dayl(day)'] = forcing_data['dayl(s)'] / 86400
        forcing_data['tmean(C)'] = (forcing_data['tmin(C)'] + forcing_data['tmax(C)']) / 2

        ## load area from header
        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return forcing_data, area

    def load_flow_data(self, working_path, huc_id, basin_id, area):
        flow_path = os.path.join(working_path, 'camels', 'usgs_streamflow', huc_id,
                                 basin_id + '_streamflow_qc.txt')
        flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                header=None, engine='python')
        flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']])
        flow_data['flow(mm)'] = 28316846.592 * flow_data['Q'] * 86400 / (area * 10 ** 6)
        return flow_data

    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.working_path, 'basin_list.txt'),
                                 sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
        self.check_validation(basin_list, self.basin_id)
        huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
        print('Now load data in basin #{} at huc #{}.'.format(self.basin_id, huc_id))
        forcing_data, area = self.load_forcing_data(self.working_path, huc_id, self.basin_id)
        flow_data = self.load_flow_data(self.working_path, huc_id, self.basin_id, area)
        merged_data = pd.merge(forcing_data, flow_data, on='date')
        merged_data = merged_data[(merged_data['date'] >= pd.datetime(1980, 10, 1)) &
                                  (merged_data['date'] <= pd.datetime(2010, 9, 30))]
        merged_data = merged_data.set_index('date')
        pd_data = merged_data[['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'flow(mm)']]

        return pd_data


class DataofAttributes():
    def __init__(self, working_path):
        self.working_path = working_path

    def load_attr_data(self, working_path):
        attr_folder = os.path.join(working_path, 'camels', 'camels_attributes_v2.0')
        clim_data = pd.read_csv(f"{attr_folder}/camels_clim.txt", sep=";", header=0, dtype={'gauge_id': str})
        topo_data = pd.read_csv(f"{attr_folder}/camels_topo.txt", sep=";", header=0, dtype={'gauge_id': str})
        vege_data = pd.read_csv(f"{attr_folder}/camels_vege.txt", sep=";", header=0, dtype={'gauge_id': str})
        soil_data = pd.read_csv(f"{attr_folder}/camels_soil.txt", sep=";", header=0, dtype={'gauge_id': str})
        geol_data = pd.read_csv(f"{attr_folder}/camels_geol.txt", sep=";", header=0, dtype={'gauge_id': str})

        clim_data = clim_data[
            ['gauge_id', 'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq',
             'high_prec_dur', 'low_prec_freq', 'low_prec_dur']].set_index('gauge_id')
        topo_data = topo_data[['gauge_id', 'elev_mean', 'slope_mean', 'area_geospa_fabric']].set_index('gauge_id')
        vege_data = vege_data[['gauge_id', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff']].set_index(
            'gauge_id')
        soil_data = soil_data[['gauge_id', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
                               'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
                               'organic_frac']].set_index('gauge_id')
        geol_data = geol_data[['gauge_id', 'geol_permeability']].set_index('gauge_id')

        dfs = [clim_data, topo_data, vege_data, soil_data, geol_data]
        return pd.concat(dfs, join='outer', axis=1)

    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.working_path, 'basin_list.txt'),sep='\t', header=0,
                                 usecols=['BASIN_ID'], dtype={'BASIN_ID': str}).values[:, 0].tolist()
        attr_data = self.load_attr_data(self.working_path)

        return attr_data.loc[basin_list]
