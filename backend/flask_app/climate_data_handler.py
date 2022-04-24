import requests
import numpy as np
import json

with open('../agri_product/condition.json', 'r') as f:
  conditions = json.load(f)

def fetch_climate_data(lon, lat):
    json_body = {
                    "filters": {
                        "dddms": "dd",
                        "xcoord": lon,
                        "ycoord": lat
                    }
                }
    r = requests.post('https://us-central1-fao-aquastat.cloudfunctions.net/aquastatGetClimateData?action=getData', json=json_body)
    if r.status_code == 200:
        response = r.json()
    else:
        return None
    res = response["cruInfo"]["measures"]
    result_json = {
        "av_temp": str(round(np.mean([elt[1] for elt in res["meantemp"]["data"]]),2)),
        "min_temp": str(min([round(elt[1],2) for elt in res["mintemp"]["data"]])),
        "max_temp": str(max([round(elt[1],2) for elt in res["maxtemp"]["data"]])),
        "annual_rain_mm": str(sum([round(elt[1],2) for elt in res["pmmm"]["data"]])),
        "monthly_average_wetday" : str(round(np.mean([elt[1] for elt in res["wdays"]["data"]]),2)), 
        "yearly_frost_free": str(sum([round(elt[1],2) for elt in res["gfrstd"]["data"]])),
        "av_humidity" : str(round(np.mean([elt[1] for elt in res["relh"]["data"]]),2)),
        "av_solar_fraction" : str(round(np.mean([elt[1] for elt in res["sunf"]["data"]]),2)),
        "annual_evapotranspiration_mm" : str(sum([round(elt[1],2) for elt in res["etrfm"]["data"]])) 
    }
    return result_json

def calculate_product_score(product_av_temp, loc_av_temp):
    if product_av_temp == -99:
        return 0
    distance = abs(float(product_av_temp) - float(loc_av_temp))
    if distance > 10:
        return 0
    else:
        return 100 - distance * 10

def is_possible(product_json, loc_json):
    if float(product_json["min_temp"]) != -99:
        cond1 = float(loc_json["min_temp"]) > float(product_json["min_temp"]) #should be true
    else:
        cond1 = True
        
    if float(product_json["max_temp"]) != -99:
        cond2 = float(loc_json["max_temp"]) > float(product_json["max_temp"]) #should be true
    else:
        cond2 = True
        
    if float(product_json["annual_min_rain_mm"]) != -99:
        cond3 = float(loc_json["annual_rain_mm"]) > float(product_json["annual_min_rain_mm"]) #should be true
    else:
        cond3 = True
        
    if float(product_json["annual_max_rain_mm"]) != -99:
        cond4 = float(loc_json["annual_rain_mm"]) < float(product_json["annual_max_rain_mm"]) #should be true
    else:
        cond4 = True
        
    if float(product_json["yearly_frost_free"]) != -99:
        cond5 = float(loc_json["yearly_frost_free"]) < float(product_json["yearly_frost_free"]) #should be true
    else:
        cond5 = True
    
    return cond1 and cond2 and cond3 and cond4 and cond5
        

def get_good_products(climate_json):
    possible_products_json = []
    for k,v in conditions.items():
        if is_possible(v, climate_json):
            score = calculate_product_score(v["av_temp"], climate_json["av_temp"])
            if score > 0:
                possible_products_json.append((k,str(round(score,2))))
    return possible_products_json