import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, timezone
import pytz
from timezonefinder import TimezoneFinder
import json
import os
import folium

def localize_polygon(row, base_time):
        # Initialize the finder
    tf = TimezoneFinder()
    
    # 1. Get a point inside the polygon for timezone lookup
    # representative_point is safer than centroid for irregular polygons
    pt = row['geometry'].representative_point()
    ln, lt = pt.x, pt.y
    
    # 2. Dynamic Timezone Detection
    tz_str = tf.timezone_at(lng=ln, lat=lt) or "UTC"
    local_tz = pytz.timezone(tz_str)

    # 3. Calculate actual UTC times from offsets
    dt_start_utc = base_time + timedelta(hours=float(row['start']))
    dt_end_utc = base_time + timedelta(hours=float(row['end']))
    
    # 4. Convert to the detected Local Timezone
    dt_start_local = dt_start_utc.astimezone(local_tz)
    dt_end_local = dt_end_utc.astimezone(local_tz)
    
    # 5. Format for final output
    time_fmt = "%b %d, %I:%M %p %Z"
    
    return pd.Series([
        dt_start_local.strftime(time_fmt), 
        dt_end_local.strftime(time_fmt),
        tz_str
    ])

def spatial_summary(alert_cells, lat, lon, alert_type, severity, unit, model_run_time):
    
    (start_hour, end_hour, intermittent, total_val_during_alerts, any_alert) =  alert_cells
    
    # model_run_time = "20260328T06Z" # Example input this is the time the prediction model run
    # if any_alert is None or not np.any(any_alert):
    
    lats_active = lat[any_alert]
    lons_active = lon[any_alert]
    start_active = start_hour[any_alert]
    end_active = end_hour[any_alert]
    # inter_active = intermittent[any_alert]
    val_active = total_val_during_alerts[any_alert]
    
    if len(val_active) == 0:
        return gpd.GeoDataFrame()#[]
    
    # Your dynamic folder logic
    # model_run_time = folder_name#"20260328T06Z" # Example input
    base_utc = datetime.strptime(model_run_time.replace('T', '').replace('Z', ''), "%Y%m%d%H")
    base_utc = pytz.utc.localize(base_utc)


    # 2. Extract only the alert data into a flat DataFrame
    df_alerts = pd.DataFrame({
        'lat': lats_active,
        'lon': lons_active,
        'start': start_active,
        'end': end_active,
        'val': val_active,
        'alert_type': alert_type,
        'severity': severity,
        'unit': unit
    })

    # 3. Convert to GeoDataFrame
    # gpd.points_from_xy is the high-speed way to create geometries
    gdf_points = gpd.GeoDataFrame(
        df_alerts, 
        geometry=gpd.points_from_xy(df_alerts.lon, df_alerts.lat),
        crs="EPSG:4326"
    )

    # 4. Load your regions (Shapefile, GeoJSON, etc.)
    # regions = gpd.read_file(r"D:\Fahime\Fahim MSC\SnowAlert\CanPoly\forcast_region_canada.shp")

    regions = gpd.read_file(r".\src\polygon\forcast_region_canada_5.shp")

    # 5. Ensure the coordinate systems match!
    if regions.crs != gdf_points.crs:
        regions = regions.to_crs(gdf_points.crs)
        
        
    # 6. Spatial Join: 'within' connects the points to the region polygon
    # This adds the region's columns (like 'Region_Name') to our alert points
    joined = gpd.sjoin(gdf_points, regions, predicate='within', how='inner')    

    # 7. Use .agg to get Min Start, Max End, and Average Value per region
    final_alerts = joined.groupby('CLC').agg({
        'start': 'min',     # Earliest onset
        'end': 'max',       # Latest end time
        'val': 'mean',      # Average intensity for the region
        'alert_type': 'first',  
        'severity': 'first',
        'unit': 'first',
 
    }).reset_index()
    
    #merge rigion with final_alerts to have geometry for the geojson
    final_alerts_gdf = regions.merge(final_alerts, on='CLC', how='inner')
    # Re-convert to GeoDataFrame (since groupby usually returns a DataFrame)
    # final_alerts_gdf = gpd.GeoDataFrame(final_alerts, crs=regions.crs)
    
    # add local time
    if len(final_alerts_gdf) == 0:
        return gpd.GeoDataFrame()#[]
    else:    
        final_alerts_gdf[['start_local', 'end_local', 'timezone']] = final_alerts_gdf.apply(
        localize_polygon, 
        axis=1, 
        args=(base_utc,))
    
    return final_alerts_gdf[['CLC', 'NAME', 'PROVINCE_C', 'alert_type', 'severity', 'start_local', 'end_local', 'val','unit', 'geometry']]

  



def geojson_writer_email(polygon_summary, priority_list, output_dir, model_run_time, event):
    # Define priority mapping (lower number = higher priority)
    # priority_list = ['Significant Rain Warning', 'Rain Warning', 'Rain Advisory']
    
    # 1. Gather all potential datasets
    #datasets = [significant_rain_summary, rain_warning_summary, rain_advisory_summary]

    # Filter out empty lists and ensure we only process valid GeoDataFrames
    # valid_gdfs = [df for df in datasets if isinstance(df, gpd.GeoDataFrame) and not df.empty]
    valid_gdfs = []
    for tier, gdf in polygon_summary.items():
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
            valid_gdfs.append(gdf)

    if valid_gdfs:
        # Concatenate all valid datasets into one GeoDataFrame
        combined_gdf = pd.concat(valid_gdfs, ignore_index=True)
    else:
        # Fallback to an empty GeoDataFrame if no alerts were issued
        combined_gdf = gpd.GeoDataFrame()
        
    geojson_dict = {
    "type": "FeatureCollection",
    "metadata": {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_polygons": 0,
        "active_alerts_count": 0
    },
    "features": []
    }
    if combined_gdf.empty:
        return geojson_dict
    # 3. Define priority mapping (lower number = higher priority)
    #priority_list = ['Significant Rain Warning', 'Rain Warning', 'Rain Advisory']
    priority_map = {alert: i for i, alert in enumerate(priority_list)}

    # Map the priority rank to the DataFrame
    combined_gdf['priority_rank'] = combined_gdf['alert_type'].map(priority_map)

    # Sort by Polygon ID (CLC) and then by our priority rank
    combined_gdf = combined_gdf.sort_values(by=['CLC', 'priority_rank'])

    # Drop duplicates for the same polygon, keeping only the first row (the highest priority)
    filtered_gdf = combined_gdf.drop_duplicates(subset=['CLC'], keep='first').copy()

    # Update global metadata
    geojson_dict["metadata"]["total_polygons"] = len(filtered_gdf)
    geojson_dict["metadata"]["active_alerts_count"] = len(filtered_gdf)

    # 4. Construct the GeoJSON Schema
    for _, row in filtered_gdf.iterrows():
            # Use shapely's built-in __geo_interface__ to easily extract the geometry coordinates
            geom_json = row['geometry'].__geo_interface__
            
            feature = {
                "type": "Feature",
                "id": str(row['CLC']),
                "geometry": geom_json,
                "properties": {
                    "location_name": str(row['NAME']),
                    "total_active_alerts": 1,
                    "highest_severity": str(row['severity']),
                    "alerts": [
                        {
                            "event_type": str(row['alert_type']),
                            "severity": str(row['severity']),
                            "starts_at": str(row['start_local']),
                            "expires_at": str(row['end_local']),
                            "value": row['val'],
                            "unit": row['unit'],
                            "Province": str(row['PROVINCE_C'])
                        }
                    ]
                }
            }
            geojson_dict["features"].append(feature)

    target_dir = os.path.join(output_dir, model_run_time)
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, f"{event}_email.json")
    # 3. Write the file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # indent=2 makes the file "pretty" and easier for the AI/Human to read
            # ensure_ascii=False handles any special characters in location names
            json.dump(geojson_dict, f, indent=2, ensure_ascii=False)
        print(f"  Successfully saved GeoJSON to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")

                
    return geojson_dict         



def geojson_writer_all_alerts(polygon_summary, priority_list, output_dir, model_run_time, event):
    # 1. Gather all potential datasets
    # Filter out empty lists and ensure we only process valid GeoDataFrames
    
    valid_gdfs = []
    for tier, gdf in polygon_summary.items():
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
            valid_gdfs.append(gdf)
   
   
    if valid_gdfs:
        # Concatenate all valid datasets into one GeoDataFrame
        combined_gdf = pd.concat(valid_gdfs, ignore_index=True)
    else:
        # Fallback to an empty GeoDataFrame if no alerts were issued
        combined_gdf = gpd.GeoDataFrame()
        
    geojson_dict = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_polygons": 0,
            "active_alerts_count": 0
        },
        "features": []
    }

    if combined_gdf.empty:
        return geojson_dict

    # 2. Define priority mapping (lower number = higher priority)
    priority_map = {alert: i for i, alert in enumerate(priority_list)}

    # Map the priority rank to the DataFrame
    combined_gdf['priority_rank'] = combined_gdf['alert_type'].map(priority_map)

    # Sort by Polygon ID (CLC) and then by our priority rank
    combined_gdf = combined_gdf.sort_values(by=['CLC', 'priority_rank'])

    # Update global metadata
    # total_polygons = number of unique CLCs
    # active_alerts_count = total number of alerts across all polygons
    geojson_dict["metadata"]["total_polygons"] = combined_gdf['CLC'].nunique()
    geojson_dict["metadata"]["active_alerts_count"] = len(combined_gdf)

    # 3. Group by Polygon ID (CLC) to stack the alerts
    grouped_gdf = combined_gdf.groupby('CLC')

    # 4. Construct the GeoJSON Schema
    for clc, group in grouped_gdf:
        # Since the dataframe was sorted by priority rank, the first row in this group 
        # is guaranteed to be the highest priority alert.
        first_row = group.iloc[0]
        
        geom_json = first_row['geometry'].__geo_interface__
        
        # Build the list of all alerts for this specific polygon
        stacked_alerts = []
        for _, row in group.iterrows():
            stacked_alerts.append({
                "event_type": str(row['alert_type']),
                "severity": str(row['severity']),
                "starts_at": str(row['start_local']),
                "expires_at": str(row['end_local']),
                "value": row['val'],
                "unit": row['unit'],
                "Province": str(row['PROVINCE_C'])
            })
            
        feature = {
            "type": "Feature",
            "id": str(clc),
            "geometry": geom_json,
            "properties": {
                "location_name": str(first_row['NAME']),
                "total_active_alerts": len(stacked_alerts),
                "highest_severity": str(first_row['severity']),
                "alerts": stacked_alerts
            }
        }
        geojson_dict["features"].append(feature)

    target_dir = os.path.join(output_dir, model_run_time)
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, f"{event}.json")
    # 3. Write the file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # indent=2 makes the file "pretty" and easier for the AI/Human to read
            # ensure_ascii=False handles any special characters in location names
            json.dump(geojson_dict, f, indent=2, ensure_ascii=False)
        print(f"  Successfully saved GeoJSON to {filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")
        
                    
    return geojson_dict        


#####################################################################################


# Define severity ranking to accurately calculate the 'highest_severity' property
SEVERITY_RANK = {
    "Extreme": 5,
    "Severe": 4,
    "High": 3,
    "Moderate": 2,
    "Minor": 1,
    "Low": 0,
    "Unknown": -1
}

def get_highest_severity(alerts):
    """Returns the highest severity string from a list of alert dictionaries."""
    highest = "Unknown"
    highest_rank = -1
    for alert in alerts:
        severity = alert.get("severity", "Unknown")
        rank = SEVERITY_RANK.get(severity, -1)
        if rank > highest_rank:
            highest_rank = rank
            highest = severity
    return highest

def merge_geojson_alerts(file_paths, output_dir, model_run_time, output_name):
    merged_features = {}
    
    for data in file_paths:
                    
        for feature in data.get("features", []):
            feature_id = feature["id"]
            
            if feature_id in merged_features:
                # Polygon exists: append the new alerts
                existing_alerts = merged_features[feature_id]["properties"]["alerts"]
                new_alerts = feature["properties"]["alerts"]
                existing_alerts.extend(new_alerts)
            else:
                # New polygon: add to dictionary
                merged_features[feature_id] = feature

    # Global counter for active alerts across all polygons
    total_active_alerts_count = 0
    
    # Recalculate properties for every merged polygon
    for feature_id, feature in merged_features.items():
        alerts = feature["properties"]["alerts"]
        
        # Update feature-level properties
        feature["properties"]["total_active_alerts"] = len(alerts)
        feature["properties"]["highest_severity"] = get_highest_severity(alerts)
        
        # Add to global count
        total_active_alerts_count += len(alerts)

    # Reconstruct the final FeatureCollection GeoJSON
    merged_geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_polygons": len(merged_features),
            "active_alerts_count": total_active_alerts_count
        },
        "features": list(merged_features.values())
    }


    target_dir = os.path.join(output_dir, model_run_time)
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, output_name)
    
    # Export the merged dataset
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(merged_geojson, f, indent=2)
    
    return merged_geojson    
    # # print(f"Successfully merged {len(file_paths)} files into {output_path}")
    # print(f"Total Unique Polygons: {merged_geojson['metadata']['total_polygons']}")
    # print(f"Total Active Alerts: {merged_geojson['metadata']['active_alerts_count']}")


#################################

# 2. Define the styling function for Folium
def style_function(feature):
    # Extract the severity from the properties (default to 'Unknown' if missing)
    severity = feature['properties'].get('highest_severity', 'Unknown')
    return {
        'fillColor': get_severity_color(severity),
        'color': 'black',       # Border color
        'weight': 1.5,          # Border thickness
        'fillOpacity': 0.3      # Transparency of the polygon fill
    }
    
# (Assuming geojson_data and get_severity_color from the previous script are loaded)
def get_severity_color(severity):
    """Maps alert severity to standard warning colors."""
    color_map = {
        'Extreme': '#d73027',   # Red
        'Severe': '#fc8d59',    # Orange
        'High': '#fee08b',  # Yellow/Gold
        'Moderate': '#d9ef8b',     # Light Green
        'Unknown': '#999999'    # Grey (Fallback)
    }
    # Return the mapped color, default to Grey if not found
    return color_map.get(severity, color_map['Unknown'])

# 1. Create a function to generate HTML for the popup
def generate_popup_html(properties):
    # Header with Location Name
    html = f"""
    <div style="width: 250px; font-family: Arial, sans-serif;">
        <h4 style="margin-bottom: 5px; color: #333;">{properties.get('location_name', 'Unknown Location')}</h4>
        <p style="margin-top: 0; font-size: 12px; color: #666;">
            Active Alerts: <b>{properties.get('total_active_alerts', 0)}</b>
        </p>
        <hr style="border: 0; border-top: 1px solid #ccc;">
    """
    
    # Loop through the nested alerts array
    for alert in properties.get('alerts', []):
        severity = alert.get('severity', 'Unknown')
        event_type = alert.get('event_type', 'Unknown')
        unit = alert.get('unit', '')
        # Get the color for the left border to indicate severity visually
        border_color = get_severity_color(severity) 
        
        # # --- NEW LOGIC: Determine unit dynamically ---
        # event_type_lower = event_type.lower()
        # snow_events = ['snowfall_advisory', 'snowfall_warning', 'snow_squall']
        # rain_events = ['rain_advisory', 'rain_warning', 'heavy_rain', 'significant_rain']
        # wind_events = ['wind_warning', 'extreme_wind']
        
        # if event_type_lower in snow_events:
        #     unit = "cm"
        # elif event_type_lower in rain_events:
        #     unit = "mm"
        # elif event_type_lower in wind_events:
        #     unit = "km/h"
        # else:
        #     unit = "mm" # Default fallback for freezing rain or other unlisted types
        # ---------------------------------------------        
        
        # Build a "card" for each alert
        html += f"""
        <div style="margin-bottom: 10px; border-left: 4px solid {border_color}; padding-left: 8px; background-color: #f9f9f9; padding-top: 4px; padding-bottom: 4px;">
            <b style="color: black; font-size: 14px;">{alert.get('event_type')}</b> 
            <span style="font-size: 11px; background: #eee; padding: 2px 4px; border-radius: 3px;">{severity}</span>
            <div style="font-size: 12px; margin-top: 4px;">
                ⏱️ <b>Start:</b> {alert.get('starts_at')}<br>
                🛑 <b>Ends:</b> {alert.get('expires_at')}<br>
                💧 <b>Value:</b> {round(alert.get('value', 0), 2)} {unit}<br>
                📍 <b>Prov:</b> {alert.get('Province')}
            </div>
        </div>
        """
    html += "</div>"
    return html

def map_alerts(geojson_data, output_dir, model_run_time, output_html_name):
    # 2. Initialize the map
    m = folium.Map(location=[56.13, -106.34], zoom_start=3, tiles="CartoDB positron")

    # 3. Add polygons feature by feature so we can attach custom HTML popups
    for feature in geojson_data['features']:
        # Generate the HTML for this specific feature
        popup_html = generate_popup_html(feature['properties'])
        
        # Create an IFrame to hold the HTML (prevents CSS conflicts with the rest of the map)
        iframe = folium.IFrame(html=popup_html, width=280, height=200)
        popup = folium.Popup(iframe, max_width=280)
        
        # Add the single feature to the map
        folium.GeoJson(
            feature,
            style_function=style_function, # Uses the styling function from the previous code
            popup=popup,
            tooltip=feature['properties']['location_name'] # Simple tooltip on hover
        ).add_to(m)
    
    target_dir = os.path.join(output_dir, model_run_time)
    os.makedirs(target_dir, exist_ok=True)
    
    filepath = os.path.join(target_dir, f"{model_run_time}_{output_html_name}")    

    m.save(filepath)

