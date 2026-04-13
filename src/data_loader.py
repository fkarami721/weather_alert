import os
import glob
import xarray as xr
import numpy as np

def grib2_reader(target_dir):
    lat_list = []
    lon_list = []
    value_list = []

    # Find all grib2 files and sort them to maintain the time order (001 to 048)
    files = sorted(glob.glob(os.path.join(target_dir, "*.grib2")))
    
    for file_path in files:
        try:
            # Open the dataset
            ds = xr.open_dataset(file_path, engine='cfgrib')
            
            data_var_names = list(ds.data_vars)
            if not data_var_names:
                print(f"No data variables found in {file_path}")
                ds.close()
                continue
                
            var_name = data_var_names[0]
            
            # Append values to your lists
            values_32bit = ds[var_name].values.astype(np.float32)
            value_list.append(values_32bit)
            
            # value_list.append(ds[var_name].values)
            
            # Close the dataset
            ds.close()

        except Exception as e:
            print(f"Could not read {file_path}: {e}")
    
    lat_list.append(ds.latitude.values.astype(np.float32))
    lon_list.append(ds.longitude.values.astype(np.float32))
    # Verification
    # print(f"\nTotal steps processed: {len(value_list)}")
    return lat_list, lon_list, np.stack(value_list, axis=0)

def load_event_datasets(dataset_names, dataset_paths):
    """
    Helper function to load the required datasets for an event.
    Returns a dictionary mapping the dataset name to its local directory path.
    """
    loaded_data = {}
    for name in dataset_names:
        if name in dataset_paths:
            # You can replace this with xarray logic (e.g., xr.open_mfdataset) 
            # if you prefer to pass open dataset objects instead of paths.
            lat_list, lon_list, value_list = grib2_reader(dataset_paths[name])
            
            if name == "temperature":
                value_list = value_list - 273.15  # Convert from Kelvin to Celsius
            elif name == "conditional_snow_accumulation":
                value_list = value_list *100 # convert to cm of snow depth 
            elif name == "gust_max" :
                value_list = value_list[1:] * 3.6 # convert from m/s to km/hr
            elif name in ("wind_gust", "sustained_wind_speed"):
                value_list = value_list * 3.6 # convert from m/s to km/hr
                # print(f" {name} converted from m/s to km/hr")
            elif name == "conditional_rain_accumulation":
                value_list = value_list * 1000 # convert from m to mm
            
            loaded_data[name] = value_list 
            print(f" {name}  loaded successfully")   
    loaded_data['latitude'] = lat_list[0]
    loaded_data['longitude'] = lon_list[0]
    
         
    return loaded_data 

