import numpy as np
def snow_data_proc(loaded_data):
    processed = {}
    sum_cond_snow_24hr = []
    for i in range(loaded_data['conditional_snow_accumulation'].shape[0]-24 +1):
        sum_cond_snow = np.nansum(loaded_data['conditional_snow_accumulation'][i:i+24], axis=0)
        sum_cond_snow_24hr.append(sum_cond_snow)
    sum_cond_snow_24hr = np.stack(sum_cond_snow_24hr, axis=0) # cumulative sum of snow precip over 24 hours
    processed['sum_cond_snow_24hr'] = sum_cond_snow_24hr
    
    sum_cond_snow_12hr = []
    for i in range(loaded_data['conditional_snow_accumulation'].shape[0] -12 +1):
        sum_cond_snow = np.nansum(loaded_data['conditional_snow_accumulation'][i:i+12], axis=0)
        sum_cond_snow_12hr.append(sum_cond_snow)
    sum_cond_snow_12hr = np.stack(sum_cond_snow_12hr, axis=0) # cumulative sum of snow precip over 12 hours
    processed['sum_cond_snow_12hr'] = sum_cond_snow_12hr

    avg_prob_24hr = []
    for i in range(loaded_data['snow_probability'].shape[0] -24 + 1):
        avg_prob = np.nanmean(loaded_data['snow_probability'][i:i+24], axis=0)
        avg_prob_24hr.append(avg_prob)
    avg_prob_24hr = np.stack(avg_prob_24hr, axis=0) 
    processed['avg_prob_24hr'] = avg_prob_24hr
    
    avg_prob_12hr = []
    for i in range(loaded_data['snow_probability'].shape[0] -12 + 1):
        avg_prob = np.nanmean(loaded_data['snow_probability'][i:i+12], axis=0)
        avg_prob_12hr.append(avg_prob)
    avg_prob_12hr = np.stack(avg_prob_12hr, axis=0)
    processed['avg_prob_12hr'] = avg_prob_12hr

    processed['prob_snow_squall_1h'] = loaded_data['snow_squall_probability']
    processed['max_gust_1h'] = loaded_data['gust_max']

    return processed

def rain_data_proc(loaded_data):
    processed = {}


    sum_cond_rain_24hr = []
    for i in range(loaded_data['conditional_rain_accumulation'].shape[0] -24 +1):
        sum_cond_rain = np.nansum(loaded_data['conditional_rain_accumulation'][i:i+24], axis=0)
        sum_cond_rain_24hr.append(sum_cond_rain)
    sum_cond_rain_24hr = np.stack(sum_cond_rain_24hr, axis=0) # cumulative sum of liquid precip over 24 hours
    processed['sum_cond_rain_24hr'] = sum_cond_rain_24hr
    
    sum_cond_rain_12hr = []
    for i in range(loaded_data['conditional_rain_accumulation'].shape[0] -12 +1):
        sum_cond_rain = np.nansum(loaded_data['conditional_rain_accumulation'][i:i+12], axis=0)
        sum_cond_rain_12hr.append(sum_cond_rain)
    sum_cond_rain_12hr = np.stack(sum_cond_rain_12hr, axis=0) # cumulative sum of liquid precip over 12 hours
    processed['sum_cond_rain_12hr'] = sum_cond_rain_12hr
    
    sum_cond_rain_6hr = []
    for i in range(loaded_data['conditional_rain_accumulation'].shape[0] -6 +1):
        sum_cond_rain = np.nansum(loaded_data['conditional_rain_accumulation'][i:i+6], axis=0)
        sum_cond_rain_6hr.append(sum_cond_rain)
    sum_cond_rain_6hr = np.stack(sum_cond_rain_6hr, axis=0) # cumulative sum of liquid precip over 6 hours
    processed['sum_cond_rain_6hr'] = sum_cond_rain_6hr





    avg_prob_24hr = []
    for i in range(loaded_data['rain_probability'].shape[0] -24 +1):
        avg_prob = np.nanmean(loaded_data['rain_probability'][i:i+24], axis=0)
        avg_prob_24hr.append(avg_prob)
    avg_prob_24hr = np.stack(avg_prob_24hr, axis=0) 
    processed['avg_prob_24hr'] = avg_prob_24hr

    avg_prob_12hr = []
    for i in range(loaded_data['rain_probability'].shape[0] -12 +1):
        avg_prob = np.nanmean(loaded_data['rain_probability'][i:i+12], axis=0)
        avg_prob_12hr.append(avg_prob)
    avg_prob_12hr = np.stack(avg_prob_12hr, axis=0) 
    processed['avg_prob_12hr'] = avg_prob_12hr

    avg_prob_6hr = []
    for i in range(loaded_data['rain_probability'].shape[0] -6 +1):
        avg_prob = np.nanmean(loaded_data['rain_probability'][i:i+6], axis=0)
        avg_prob_6hr.append(avg_prob)
    avg_prob_6hr = np.stack(avg_prob_6hr, axis=0) 
    processed['avg_prob_6hr'] = avg_prob_6hr



    min_tmp_24hr = []
    for i in range(loaded_data['temperature'].shape[0] -24 +1):
        min_tmp = np.min(loaded_data['temperature'][i:i+24], axis=0)
        min_tmp_24hr.append(min_tmp)
    min_tmp_24hr = np.stack(min_tmp_24hr, axis=0) 
    processed['min_tmp_24hr'] = min_tmp_24hr
    
    min_tmp_12hr = []
    for i in range(loaded_data['temperature'].shape[0] -12 +1):
        min_tmp = np.min(loaded_data['temperature'][i:i+12], axis=0)
        min_tmp_12hr.append(min_tmp)
    min_tmp_12hr = np.stack(min_tmp_12hr, axis=0) 
    processed['min_tmp_12hr'] = min_tmp_12hr

    min_tmp_6hr = []
    for i in range(loaded_data['temperature'].shape[0] -6 +1):
        min_tmp = np.min(loaded_data['temperature'][i:i+6], axis=0)
        min_tmp_6hr.append(min_tmp)
    min_tmp_6hr = np.stack(min_tmp_6hr, axis=0) 
    processed['min_tmp_6hr'] = min_tmp_6hr
    
    return processed
    
    
