import numpy as np
import os
import requests
import geopandas as gpd
import shapely

def calculate_snow_alerts_inside_can(
    prob_snow_data,           # PROBSN or PROBSNSQ 3D array
    accum_window_data,        # 3D array of snow accumulation over the window length
    hourly_snow_data,         # Base 1h snow accumulation (for total event sums)
    location,                 # [lat, lon] matrices
    window_length,            # 1, 12, or 24 hours
    prob_threshold,           # e.g., 80 or 50
    accum_min_threshold,      # e.g., 5, 15, or 3
    accum_max_threshold=None, # e.g., 10 (for Advisory), None for others
    gust_data=None,           # Optional 3D array for wind gust speed (m/s)
    gust_min_threshold=None   # e.g., 25
):
    """
    Identifies specific snow alert periods and calculates event metrics inside Canada.
    """
    # 1. Vectorized Alert Detection
    # Base conditions: Probability and Minimum Accumulation
    alert_mask = (prob_snow_data > prob_threshold) & (accum_window_data >= accum_min_threshold)
    
    # Optional upper limit (used for Snowfall Advisory 5-10 cm)
    if accum_max_threshold is not None:
        alert_mask &= (accum_window_data <= accum_max_threshold)
        
        
    # Optional wind condition (used for Snow Squalls)
    if gust_data is not None and gust_min_threshold is not None:
        alert_mask &= (gust_data >= gust_min_threshold)

    any_alert = np.any(alert_mask, axis=0)

    # -----------------------------------------------------------------
    # 2. Apply Canada Spatial Mask
    # -----------------------------------------------------------------
    [lat, long] = location
    world_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    filepath = 'countries.geojson'

    try:
        if not os.path.exists(filepath):
            r = requests.get(world_url)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)                
    except Exception as e:
        print(f"Failed to download boundary data: {e}")

    world_gdf = gpd.read_file(filepath)
    canada_gdf = world_gdf[world_gdf['name'] == 'Canada']
    canada_boundary = canada_gdf.geometry.union_all()

    lon_flat = long.flatten()
    lat_flat = lat.flatten()
    grid_points = shapely.points(lon_flat, lat_flat)
    
    is_in_canada_flat = shapely.contains(canada_boundary, grid_points)
    canada_mask = is_in_canada_flat.reshape(any_alert.shape)

    # Keep alerts ONLY where conditions are met AND it is inside Canada
    any_alert = any_alert & canada_mask
    
    # -----------------------------------------------------------------
    # 3. Map windows back to hourly mask
    # -----------------------------------------------------------------
    num_windows = alert_mask.shape[0]
    hourly_alert_mask = np.zeros_like(hourly_snow_data, dtype=bool)

    for i in range(num_windows):
        if np.any(alert_mask[i]):
            hourly_alert_mask[i : i + window_length] |= alert_mask[i]
            
    # 4. Calculate Total Event Snow
    total_snow_during_alerts = np.sum(hourly_snow_data * hourly_alert_mask, axis=0)

    # 5. Calculate Timing Metrics
    # argmax finds the first 'True'
    start_hour = np.argmax(alert_mask, axis=0).astype(float)
    
    # Reverse to find the last 'True', then adjust for window length
    end_hour = (alert_mask.shape[0] - 1 - np.argmax(alert_mask[::-1, :, :], axis=0)).astype(float) + window_length

    # 6. Intermittency Logic
    active_window_count = np.nansum(alert_mask, axis=0)
    max_continuous_span = window_length + (active_window_count - 1)
    actual_duration = end_hour - start_hour
    intermittent = actual_duration > max_continuous_span

    # 7. Post-processing: Mask out pixels with no alerts
    start_hour[~any_alert] = np.nan
    end_hour[~any_alert] = np.nan
    intermittent[~any_alert] = np.nan
    total_snow_during_alerts[~any_alert] = np.nan
    
    return (start_hour, end_hour, intermittent, total_snow_during_alerts, any_alert)


def calculate_snow_alerts(processed_data, loaded_data, tier_name,  tier_params, latitude, longitude):
    if tier_name == 'snowfall_advisory':
        #light snow (24 h ): PROBSN > 80% and 5 cm <= accum < 10 cm
        # print(f"prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, accum_max_threshold: {tier_params['accum_max_threshold']}")
        out = calculate_snow_alerts_inside_can(
                processed_data['avg_prob_24hr'],           # PROBSN or PROBSNSQ 3D array
                processed_data['sum_cond_snow_24hr'],        # 3D array of snow accumulation over the window length
                loaded_data['conditional_snow_accumulation'],         # Base 1h snow accumulation (for total event sums)
                location = [latitude, longitude],  # [lat, lon] matrices
                window_length = 24,            # 1, 12, or 24 hours
                prob_threshold = tier_params['prob_min_threshold'],           # e.g., 80 or 50 %
                accum_min_threshold = tier_params['accum_min_threshold'],      # e.g., 5, 15, or 3 cm
                accum_max_threshold= tier_params['accum_max_threshold'], # e.g., 10 (for Advisory), None for others 
                )
        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out
    
    elif tier_name == 'snowfall_warning':
        # Snow fall warning (12 h): PROBSN > 80% and accum >= 15 cm
        # print(f"prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}")
        out = calculate_snow_alerts_inside_can(
                    processed_data['avg_prob_12hr'],            
                    processed_data['sum_cond_snow_12hr'],         
                    loaded_data['conditional_snow_accumulation'],         
                    location = [latitude, longitude],   
                    window_length = 12,             
                    prob_threshold = tier_params['prob_min_threshold'],            
                    accum_min_threshold = tier_params['accum_min_threshold']
                    )
                
        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out
    
    elif tier_name == 'snow_squall':
    
        # Snow squall warning (1 h): PROBSN > 50%, 1h snow accumulation >= 3 cm,  and max_gusts >= 25 km/h
        # print(f"prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, gust_min_threshold: {tier_params['gust_min_threshold']}")
        out = calculate_snow_alerts_inside_can(
            loaded_data['snow_squall_probability'],           # PROBSN or PROBSNSQ 3D array
            loaded_data['conditional_snow_accumulation'],        # 3D array of snow accumulation over the window length
            loaded_data['conditional_snow_accumulation'],         # Base 1h snow accumulation 
            location = [latitude, longitude],  # [lat, lon] matrices
            window_length = 1,            # 1, 12, or 24 hours
            prob_threshold = tier_params['prob_min_threshold'],           # e.g., 80 or 50 %
            accum_min_threshold = tier_params['accum_min_threshold'],      # e.g., 5, 15, or 3 cm    
            gust_data=processed_data['max_gust_1h'],           # Optional 3D array for wind gust speed (m/s)
            gust_min_threshold=tier_params['gust_min_threshold']   # e.g., 25
            
        )
        
        (start_hour, 
        end_hour, intermittent,  
        total_val_during_alerts, 
        any_alert) = out
    
    else:
        raise ValueError(
            f"Invalid tier_name '{tier_name}' "
        )
    
    return (start_hour, end_hour, intermittent, total_val_during_alerts, any_alert)


#%% Rain alert logic
import numpy as np
import geopandas as gpd
import shapely
import requests
import os
def calculate_rain_alerts_inside_can(
    avg_prob_24hr, 
    sum_cond_rain_24hr, 
    min_temp_24hr, 
    cond_rain_1h,
    location,
    window_length=24,
    prob_threshold=60, 
    rain_amount_threshold=15, 
    temp_threshold=0.5
):
    """
    Identifies rain alert periods and calculates event metrics across a spatial grid.
    
    Returns:
        start_hour, end_hour, is_intermittent, total_rain, any_alert
    """
    
    # 1. Vectorized Alert Detection (Much faster than a for-loop)
    # This creates the entire 3D mask in one CPU cycle/operation
    rain_advisory = (
        (avg_prob_24hr >= prob_threshold) & 
        (sum_cond_rain_24hr >= rain_amount_threshold) & 
        (min_temp_24hr > temp_threshold)
    )

    any_alert = np.any(rain_advisory, axis=0)
    #############################################################
    [lat, long] = location
    # Create a mask for locations outside Canada (assuming lat/lon are 2D arrays
    # --- 1. Your Existing Download Code ---
    world_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    filepath = 'countries.geojson'

    try:
        r = requests.get(world_url)
        r.raise_for_status()
        if not os.path.exists(filepath):            
            with open(filepath, 'wb') as f:
                f.write(r.content)                
    except Exception as e:
        print(f"Failed to download boundary data: {e}")

    # --- 2. Load and isolate Canada's geometry ---
    # Read the GeoJSON into a GeoDataFrame
    world_gdf = gpd.read_file(filepath)

    # Extract Canada (column names can vary by dataset, 'name' or 'admin' are most common)
    canada_gdf = world_gdf[world_gdf['name'] == 'Canada']

    # Combine into a single geometry (MultiPolygon)
    canada_boundary = canada_gdf.geometry.union_all()

    # --- 3. Create the spatial mask using your lat/long matrices ---
    # Assuming `long`, `lat`, and `any_alert` are already loaded and have the same shape.
    # Flatten the grids for efficient 1D vectorized processing
    lon_flat = long.flatten()
    lat_flat = lat.flatten()

    # Create Shapely points from the flattened coordinate arrays
    # Note: Coordinate order is usually (Longitude/X, Latitude/Y)
    grid_points = shapely.points(lon_flat, lat_flat)

    # Check which points fall inside Canada (Vectorized, very fast in Shapely 2.0+)
    is_in_canada_flat = shapely.contains(canada_boundary, grid_points)

    # Reshape the boolean 1D array back into your original 2D matrix shape
    canada_mask = is_in_canada_flat.reshape(any_alert.shape)

    # --- 4. Apply the mask to your alerts ---
    # Keep alerts ONLY where both any_alert is True AND it is inside Canada
    any_alert = any_alert & canada_mask
    ###################################################
    
    # 2. Map 24-h windows back to hourly mask
    # We use a loop here because it's the most memory-efficient way to 'smear' 
    # the 24h window alert across the hourly data.
    num_windows = rain_advisory.shape[0]
    
    hourly_alert_mask = np.zeros_like(cond_rain_1h, dtype=bool)

    for i in range(num_windows):
        if np.any(rain_advisory[i]):

            hourly_alert_mask[i : i + window_length] |= rain_advisory[i]
            
    # 3. Calculate Total Unique Rain
    total_rain_during_alerts = np.sum(cond_rain_1h * hourly_alert_mask, axis=0)

    # 4. Calculate Timing Metrics
    # argmax finds the first 'True'
    start_hour = np.argmax(rain_advisory, axis=0).astype(float)
    
    # Reverse to find the last 'True'
    
    end_hour = (rain_advisory.shape[0]- 1 - np.argmax(rain_advisory[::-1, :, :], axis=0)).astype(float)  + window_length

    # 5. Intermittency Logic
    # 'sum_h' is the number of windows that triggered an alert
    active_window_count = np.nansum(rain_advisory, axis=0)
    
    # Theoretical duration if there were zero gaps:
    max_continuous_span = window_length + (active_window_count - 1)
    
    actual_duration = end_hour - start_hour
    intermittent = actual_duration > max_continuous_span

    # 6. Post-processing: Mask out pixels with no alerts
    
    start_hour[~any_alert] = np.nan
    end_hour[~any_alert] = np.nan
    intermittent[~any_alert] = np.nan
    total_rain_during_alerts[~any_alert] = np.nan
    
    return (start_hour, end_hour, intermittent, total_rain_during_alerts, any_alert)


def calculate_rain_alerts(processed_data, loaded_data, tier_name,  tier_params, latitude, longitude):

    if tier_name == 'rain_advisory':
        # print(f"{tier_name} => prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, temp_min_threshold: {tier_params['temp_min_threshold']}")
        # Rain advisonry
        out = calculate_rain_alerts_inside_can(processed_data['avg_prob_24hr'], 
                                               processed_data['sum_cond_rain_24hr'], 
                                               processed_data['min_tmp_24hr'], 
                                                loaded_data['conditional_rain_accumulation'], 
                                                location = [latitude, longitude], 
                                                window_length = 24,                      
                                                prob_threshold = tier_params['prob_min_threshold'],
                                                rain_amount_threshold = tier_params['accum_min_threshold'], 
                                                temp_threshold = tier_params['temp_min_threshold'])
        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out  
        
    elif tier_name == 'rain_warning':
        # print(f"{tier_name} => prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, temp_min_threshold: {tier_params['temp_min_threshold']}")
        # Rain warning
        out = calculate_rain_alerts_inside_can(processed_data['avg_prob_12hr'], 
                                               processed_data['sum_cond_rain_12hr'], 
                                               processed_data['min_tmp_12hr'], 
                                                loaded_data['conditional_rain_accumulation'], 
                                                location = [latitude, longitude], 
                                                window_length = 12,                      
                                                prob_threshold = tier_params['prob_min_threshold'],
                                                rain_amount_threshold = tier_params['accum_min_threshold'], 
                                                temp_threshold = tier_params['temp_min_threshold'])

        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out  
        
    elif tier_name == 'heavy_rain':
        # print(f"{tier_name} => prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, temp_min_threshold: {tier_params['temp_min_threshold']}")
    # heavy rain event
        out = calculate_rain_alerts_inside_can(processed_data['avg_prob_6hr'], 
                                               processed_data['sum_cond_rain_6hr'], 
                                               processed_data['min_tmp_6hr'], 
                                               loaded_data['conditional_rain_accumulation'], 
                                               [latitude, longitude], 
                                               window_length = 6,                      
                                               prob_threshold = tier_params['prob_min_threshold'], 
                                               rain_amount_threshold = tier_params['accum_min_threshold'], 
                                               temp_threshold = tier_params['temp_min_threshold'])

        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out 

    elif tier_name == 'significant_rain':
        # print(f"{tier_name} => prob_threshold: {tier_params['prob_min_threshold']}, accum_min_threshold: {tier_params['accum_min_threshold']}, temp_min_threshold: {tier_params['temp_min_threshold']}")
    # Significant rain event
        out = calculate_rain_alerts_inside_can(processed_data['avg_prob_24hr'], 
                                               processed_data['sum_cond_rain_24hr'], 
                                               processed_data['min_tmp_24hr'], 
                                               loaded_data['conditional_rain_accumulation'], 
                                               [latitude, longitude], 
                                               window_length = 24,                      
                                               prob_threshold = tier_params['prob_min_threshold'], 
                                               rain_amount_threshold = tier_params['accum_min_threshold'], 
                                               temp_threshold = tier_params['temp_min_threshold'])
        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out 
        
    else:
        raise ValueError(
            f"Invalid tier_name '{tier_name}' "
        )
    
    return (start_hour, end_hour, intermittent, total_val_during_alerts, any_alert)

#%% Freezing rain

def calculate_freezing_rain_alerts_inside_can(
    prob_freez_rain_1h, 
    cond_freez_rain_1h, 
    temperature_1h,     
    location,    
    prob_threshold=40, 
    frain_amount_threshold=1.0, 
    temp_threshold=0
):
    """
    Identifies rain alert periods and calculates event metrics across a spatial grid.
    
    Returns:
        start_hour, end_hour, is_intermittent, total_rain, any_alert
    """
    window_length=1
    # 1. Vectorized Alert Detection (Much faster than a for-loop)
    # This creates the entire 3D mask in one CPU cycle/operation
    rain_advisory = (
        (prob_freez_rain_1h >= prob_threshold) & 
        (cond_freez_rain_1h >= frain_amount_threshold) & 
        (temperature_1h <= temp_threshold)
    )

    any_alert = np.any(rain_advisory, axis=0)
    #############################################################
    [lat, long] = location
    # Create a mask for locations outside Canada (assuming lat/lon are 2D arrays
    # --- 1. Your Existing Download Code ---
    world_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    filepath = 'countries.geojson'

    try:
        r = requests.get(world_url)
        r.raise_for_status()
        if not os.path.exists(filepath):            
            with open(filepath, 'wb') as f:
                f.write(r.content)                
    except Exception as e:
        print(f"Failed to download boundary data: {e}")

    # --- 2. Load and isolate Canada's geometry ---
    # Read the GeoJSON into a GeoDataFrame
    world_gdf = gpd.read_file(filepath)

    # Extract Canada (column names can vary by dataset, 'name' or 'admin' are most common)
    canada_gdf = world_gdf[world_gdf['name'] == 'Canada']

    # Combine into a single geometry (MultiPolygon)
    canada_boundary = canada_gdf.geometry.union_all()

    # --- 3. Create the spatial mask using your lat/long matrices ---
    # Assuming `long`, `lat`, and `any_alert` are already loaded and have the same shape.
    # Flatten the grids for efficient 1D vectorized processing
    lon_flat = long.flatten()
    lat_flat = lat.flatten()

    # Create Shapely points from the flattened coordinate arrays
    # Note: Coordinate order is usually (Longitude/X, Latitude/Y)
    grid_points = shapely.points(lon_flat, lat_flat)

    # Check which points fall inside Canada (Vectorized, very fast in Shapely 2.0+)
    is_in_canada_flat = shapely.contains(canada_boundary, grid_points)

    # Reshape the boolean 1D array back into your original 2D matrix shape
    canada_mask = is_in_canada_flat.reshape(any_alert.shape)

    # --- 4. Apply the mask to your alerts ---
    # Keep alerts ONLY where both any_alert is True AND it is inside Canada
    any_alert = any_alert & canada_mask
    ###################################################
    
    # 2. Map 24-h windows back to hourly mask
    # We use a loop here because it's the most memory-efficient way to 'smear' 
    # the 24h window alert across the hourly data.
    num_windows = rain_advisory.shape[0]
    
    hourly_alert_mask = np.zeros_like(cond_freez_rain_1h, dtype=bool)

    for i in range(num_windows):
        if np.any(rain_advisory[i]):

            hourly_alert_mask[i : i + window_length] |= rain_advisory[i]
            
    # 3. Calculate Total Unique Rain
    total_frain_during_alerts = np.sum(cond_freez_rain_1h * hourly_alert_mask, axis=0)

    # 4. Calculate Timing Metrics
    # argmax finds the first 'True'
    start_hour = np.argmax(rain_advisory, axis=0).astype(float)
    
    # Reverse to find the last 'True'
    
    end_hour = (rain_advisory.shape[0]- 1 - np.argmax(rain_advisory[::-1, :, :], axis=0)).astype(float)  + window_length

    # 5. Intermittency Logic
    # 'sum_h' is the number of windows that triggered an alert
    active_window_count = np.nansum(rain_advisory, axis=0)
    
    # Theoretical duration if there were zero gaps:
    max_continuous_span = window_length + (active_window_count - 1)
    
    actual_duration = end_hour - start_hour
    intermittent = actual_duration > max_continuous_span

    # 6. Post-processing: Mask out pixels with no alerts
    
    start_hour[~any_alert] = np.nan
    end_hour[~any_alert] = np.nan
    intermittent[~any_alert] = np.nan
    total_frain_during_alerts[~any_alert] = np.nan
    
    return (start_hour, end_hour, intermittent, total_frain_during_alerts, any_alert)


def calculate_freezing_rain_alerts(processed_data, loaded_data, tier_name,  tier_params, latitude, longitude):

    if tier_name == 'freezing_rain_warning':      
          
        out = calculate_freezing_rain_alerts_inside_can(
            loaded_data['freezing_rain_probability'], 
            loaded_data['conditional_freezing_rain_accumulation'], 
            loaded_data['temperature'],     
            [latitude, longitude],    
            prob_threshold = tier_params['prob_min_threshold'], 
            frain_amount_threshold = tier_params['accum_min_threshold'], 
            temp_threshold = tier_params['temp_max_threshold']
        )

        (start_hour, 
         end_hour, intermittent,  
         total_val_during_alerts, 
         any_alert) = out 
        
    return (start_hour, end_hour, intermittent, total_val_during_alerts, any_alert)    


#%% Wind alert logic
import warnings
from functools import wraps
def ignore_nan_warnings(func):
    """Decorator to silently ignore expected NumPy NaN warnings."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)
    return wrapper

import scipy.ndimage as ndimage

@ignore_nan_warnings
def calculate_wind_alerts_inside_can(
    gust_1h, 
    wspd_1h, 
    location,
    gust_threshold=90.0, 
    wspd_threshold=70.0,
    require_persistence= False
):
    """
    Identifies wind alert periods across a spatial grid using max-pooling and temporal persistence.
    
    Inputs:
        gust_1h: 3D numpy array of HRDPS-WEonG_GUST (time, lat, lon)
        wspd_1h: 3D numpy array of HRDPS-WEonG_WSPD (time, lat, lon)
        location: tuple of (lat_matrix, lon_matrix)
        
    Returns:
        start_hour, end_hour, peak_gust, any_alert
    """
    
    # 1. Spatial Smoothing (Neighborhood Risk)
    # Apply a 3x3 max-pooling kernel to account for model displacement.
    # We only pool over the spatial dimensions (axes 1 and 2), not time (axis 0).
    smoothed_gust = ndimage.maximum_filter(gust_1h, size=(1, 3, 3))
    smoothed_wspd = ndimage.maximum_filter(wspd_1h, size=(1, 3, 3))

    # 2. Base Threshold Detection
    raw_alert_mask = (smoothed_gust >= gust_threshold) | (smoothed_wspd >= wspd_threshold)
    
    if require_persistence:
        # 3. Temporal Persistence (The 2-Hour Rule)
        # Vectorized check for at least 2 consecutive hours of threshold exceedance
        # This avoids loops by checking if an hour and the subsequent hour are both True
        consecutive_mask = np.zeros_like(raw_alert_mask, dtype=bool)
        two_hour_streak = raw_alert_mask[:-1] & raw_alert_mask[1:]
        
        # Smear the streak back to the original time indices
        consecutive_mask[:-1] |= two_hour_streak
        consecutive_mask[1:] |= two_hour_streak

        # Combine the persistence rule with the extreme gust bypass
        valid_alert_3d = consecutive_mask
    else:
        valid_alert_3d = raw_alert_mask 

    # Get the 2D footprint of any valid alert over the 48h period
    any_alert = np.any(valid_alert_3d, axis=0)

    # =========================================================
    # 4. Canada Masking Logic (from your template)
    # =========================================================
    [lat, long] = location
    world_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    filepath = 'countries.geojson'

    try:
        if not os.path.exists(filepath):
            r = requests.get(world_url)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)                
    except Exception as e:
        print(f"Failed to download boundary data: {e}")

    world_gdf = gpd.read_file(filepath)
    canada_gdf = world_gdf[world_gdf['name'] == 'Canada']
    canada_boundary = canada_gdf.geometry.union_all()

    lon_flat = long.flatten()
    lat_flat = lat.flatten()
    grid_points = shapely.points(lon_flat, lat_flat)
    
    is_in_canada_flat = shapely.contains(canada_boundary, grid_points)
    canada_mask = is_in_canada_flat.reshape(any_alert.shape)

    # Apply the mask to your alerts
    any_alert = any_alert & canada_mask
    valid_alert_3d = valid_alert_3d & canada_mask
    # =========================================================

    # 5. Calculate Event Metrics
    # Initialize arrays
    start_hour = np.full_like(any_alert, np.nan, dtype=float)
    end_hour = np.full_like(any_alert, np.nan, dtype=float)
    peak_gust = np.full_like(any_alert, np.nan, dtype=float)

    if np.any(any_alert):
        # start_hour: argmax finds the first 'True' index
        start_hour_raw = np.argmax(valid_alert_3d, axis=0).astype(float)
        
        # end_hour: Reverse the array to find the last 'True' index
        end_hour_raw = (valid_alert_3d.shape[0] - 1 - np.argmax(valid_alert_3d[::-1, :, :], axis=0)).astype(float) + 1
        
        # Extract the maximum gust strictly during the valid alert windows
        masked_gusts = np.where(valid_alert_3d, smoothed_gust, np.nan)
        peak_gust_raw = np.nanmax(masked_gusts, axis=0)

        # 6. Post-processing: Apply the 2D mask
        start_hour = np.where(any_alert, start_hour_raw, np.nan)
        end_hour = np.where(any_alert, end_hour_raw, np.nan)
        peak_gust = np.where(any_alert, peak_gust_raw, np.nan)

    return start_hour, end_hour, peak_gust, any_alert


def calculate_wind_alerts(processed_data, loaded_data, tier_name,  tier_params, latitude, longitude):

    if tier_name == 'wind_warning': 
        
        out = calculate_wind_alerts_inside_can(
        loaded_data['wind_gust'],
        loaded_data['sustained_wind_speed'],
        location=[latitude, longitude],
        gust_threshold=tier_params['wind_gust_min_threshold'], # gust min threshold
        wspd_threshold=tier_params['sustained_wind_speed_min_threshold'], # sustained wind speed min threshold
        require_persistence=True 
        )
        
        (start_hour,
        end_hour,
        peak_gust, 
        any_alert) = out     
    elif tier_name == 'extreme_wind':       

        out = calculate_wind_alerts_inside_can(
        loaded_data['wind_gust'],
        loaded_data['sustained_wind_speed'],
        location=[latitude, longitude],
        gust_threshold=tier_params['wind_gust_min_threshold'], # gust min threshold
        wspd_threshold=tier_params['sustained_wind_speed_min_threshold'], # sustained wind speed min threshold
        require_persistence=False 
        )
        
        (start_hour,
        end_hour,
        peak_gust, 
        any_alert) = out   
    intermittent = []
     
        
    return (start_hour, end_hour,intermittent, peak_gust, any_alert)   