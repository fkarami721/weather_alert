from config_loader import load_config
from downloader import download_HRDPS_data
from data_loader import load_event_datasets
from data_processing import snow_data_proc, rain_data_proc
from alert_processing import calculate_snow_alerts, calculate_rain_alerts, calculate_freezing_rain_alerts, calculate_wind_alerts
from utils import spatial_summary, geojson_writer_all_alerts, geojson_writer_email, merge_geojson_alerts, map_alerts


def main():
    config = load_config()
    system_settings = config['system']

    # ==========================================
    # PHASE 1: Download ALL Required Data First
    # ==========================================
    print("--- Phase 1: Downloading Datasets ---")
    dataset_paths = {} # Keep track of where we saved the data

    for dataset_name, dataset_info in config['datasets'].items():
        var_code = dataset_info['variable_code']
        # target_dir = os.path.join(system_settings['data_dir'], var_code)
        
        print(f"Fetching GRIB2 files for {var_code}...")
        # Your multi-threaded downloader runs here
        target_dir = download_HRDPS_data(data_dir = system_settings['data_dir'], 
                                            folder_name= system_settings['date_mode_run'], 
                                            variable_code=var_code, 
                                            base_url=system_settings['base_url'])
        
        # Store the path for Phase 2
        dataset_paths[dataset_name] = target_dir
        
    # ==========================================
    # PHASE 2: Process Alerts
    # ==========================================
    print("\n--- Phase 2: Processing Alerts ---")


    for event_group, event_details in config['alert_tiers'].items():
        print(f"\nProcessing {event_group}...")
        
        # 1) Load related datasets
        required_datasets = event_details['dataset']
        print(f"\n Loading raw data started...")
        loaded_data = load_event_datasets(required_datasets, dataset_paths)
        
        polygon_summary = {}
        # Route to the correct processing logic based on the event group
        if event_group == 'snow_events':
            # 2) Run snow_data_proc
            print(f"\n Alert evaluation has started for {event_group}...")
            processed_data = snow_data_proc(loaded_data)
            # 3) For each snow alert type
            
            for tier_name, tier_params in event_details['tiers'].items():
                
                # 3-1: Identify cells meeting the alert criteria
                alert_cells = calculate_snow_alerts(processed_data, loaded_data, tier_name,  tier_params, loaded_data['latitude'], loaded_data['longitude'])
                # 3-2: Return polygons with active alerts of the identified cells 
                summary = spatial_summary(alert_cells, loaded_data['latitude'], loaded_data['longitude'], alert_type = tier_name,
                                        severity = tier_params['severity'], unit = tier_params['unit'], model_run_time = system_settings['date_mode_run'])
                polygon_summary[tier_name] = summary
                print(f" {tier_name} evaluation completed."
                        + (f" No active alert for {tier_name}." if summary.empty else ""))  
            
            priority_list = ['snow_squall', 'snowfall_warning', 'snowfall_advisory']    
            snow_json_email = geojson_writer_email(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)
                
            snow_json = geojson_writer_all_alerts(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)  

        elif event_group == 'rain_events':
            # 2) Run rain_data_proc
            print(f"\n Alert evaluation has started for {event_group}...")
            processed_data = rain_data_proc(loaded_data)
            # 3) For each rain alert type            
            for tier_name, tier_params in event_details['tiers'].items():
                
                # 3-1: Identify cells meeting the alert criteria
                alert_cells = calculate_rain_alerts(processed_data, loaded_data, tier_name,  tier_params, loaded_data['latitude'], loaded_data['longitude'])
                # 3-2: Return polygons with active alerts of the identified cells 
                summary = spatial_summary(alert_cells, loaded_data['latitude'], loaded_data['longitude'], alert_type = tier_name,
                                        severity = tier_params['severity'], unit = tier_params['unit'], model_run_time = system_settings['date_mode_run'])
                polygon_summary[tier_name] = summary
                print(f" {tier_name} evaluation completed."
                        + (f" No active alert for {tier_name}." if summary.empty else ""))  
                
            priority_list = ['significant_rain', 'heavy_rain' , 'rain_warning', 'rain_advisory']   
            rain_json_email = geojson_writer_email(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)
                
            rain_json = geojson_writer_all_alerts(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)    
            
            
        elif event_group == 'freezing_rain_events':
            # 2) Freezing Run rain_data_proc
            print(f"\n Alert evaluation has started for {event_group}...")
            # processed_data = freezing_rain_data_proc(loaded_data)    # there is no specific processing for freezing rain, as we are using hourly data (no moving windows)
            
            for tier_name, tier_params in event_details['tiers'].items():
                
                # 3-1: Identify cells meeting the alert criteria
                alert_cells = calculate_freezing_rain_alerts([], loaded_data, tier_name,  tier_params, loaded_data['latitude'], loaded_data['longitude'])
                # 3-2: Return polygons with active alerts of the identified cells 
                summary = spatial_summary(alert_cells, loaded_data['latitude'], loaded_data['longitude'], alert_type = tier_name,
                                        severity = tier_params['severity'], unit = tier_params['unit'],model_run_time = system_settings['date_mode_run'])
                polygon_summary[tier_name] = summary
                print(f" {tier_name} evaluation completed."
                        + (f" No active alert for {tier_name}." if summary.empty else ""))     
                
            priority_list = ['freezing_rain_warning']   
            freezing_rain_json_email = geojson_writer_email(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)
                
            freezing_rain_json = geojson_writer_all_alerts(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)   

           
        elif event_group == 'wind_events':
            # 2) Freezing Run rain_data_proc
            print(f"\n Alert evaluation has started for {event_group}...")
            # processed_data = freezing_rain_data_proc(loaded_data)    # there is no specific processing for freezing rain, as we are using hourly data (no moving windows)
            
            for tier_name, tier_params in event_details['tiers'].items():
                
                # 3-1: Identify cells meeting the alert criteria
                alert_cells = calculate_wind_alerts([], loaded_data, tier_name,  tier_params, loaded_data['latitude'], loaded_data['longitude'])
                # 3-2: Return polygons with active alerts of the identified cells 
                summary = spatial_summary(alert_cells, loaded_data['latitude'], loaded_data['longitude'], alert_type = tier_name,
                                        severity = tier_params['severity'], unit = tier_params['unit'],model_run_time = system_settings['date_mode_run'])
                polygon_summary[tier_name] = summary
                print(f" {tier_name} evaluation completed."
                        + (f" No active alert for {tier_name}." if summary.empty else ""))                     

            priority_list = ['extreme_wind', 'wind_warning']   
            wind_json_email = geojson_writer_email(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)
                
            wind_json = geojson_writer_all_alerts(polygon_summary, priority_list, 
                                    output_dir = system_settings['output_dir'], 
                                    model_run_time = system_settings['date_mode_run'],
                                    event = event_group)   
            
    # ==========================================
        # PHASE 3: Write GeoJSON Outputs
    # ==========================================
                
    ######## merge all alerts    
    input_files = [snow_json, rain_json, freezing_rain_json, wind_json]
    # input_files = [wind_json]
    all_for_visualization = merge_geojson_alerts(input_files, 
                                                output_dir = system_settings['output_dir'], 
                                                model_run_time = system_settings['date_mode_run'],
                                                output_name = 'merged_events.json')  

    input_files = [snow_json_email, rain_json_email, freezing_rain_json_email, wind_json_email]
    # input_files = [wind_json]
    all_for_email = merge_geojson_alerts(input_files, 
                            output_dir = system_settings['output_dir'], 
                            model_run_time = system_settings['date_mode_run'],
                            output_name = 'merged_events_email.json')    
    
    # ==========================================
        # PHASE 4: Visualization
    # ==========================================
    map_alerts(geojson_data = all_for_visualization, output_dir = system_settings['output_dir'], model_run_time = system_settings['date_mode_run'], 
            output_html_name = 'alerts_map.html') 
    
    # ==========================================
        # PHASE 5: Email Generation
    # ==========================================
    from email_writer_agent import email_generator_agent
    email_text = email_generator_agent(all_for_email)
    print("--- FINAL EMAIL ---\n")
    print(email_text)
    
    
    
if __name__ == "__main__":
    main()             