from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
import time
from sentinel import get_sat_image
from model import get_prediction
from climate_data_handler import fetch_climate_data, get_good_products

app = Flask("KUAI App")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def get_bb(lon, lat):
    lon1 = lon - 12.48 / 111
    lon2 = lon + 12.48 / 111
    lat1 = lat - 12.48 / 111
    lat2 = lat + 12.48 / 111
    lon_space = np.linspace(lon1, lon2, num=40)
    lat_space = np.linspace(lat1, lat2, num=40)
    return [lon1, lat1, lon2, lat2], lon_space, lat_space


def divede_sat_img(sat_img, lon_space, lat_space):
    sat_img = np.array(sat_img)
    divided_imgs = []
    divided_imgs_names = []
    cord_list = []
    for i in range(39):
        for j in range(39):
            divided_imgs.append(sat_img[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64])
            lon1, lon2 = lon_space[i], lon_space[i + 1]
            lat1, lat2 = lat_space[j], lat_space[j + 1]
            cord_list.append([lon1, lat1, lon2, lat2])
            divided_imgs_names.append("{}_{}".format(i, j))
    return divided_imgs, divided_imgs_names, cord_list


@app.route('/', methods=('POST',))
def predict():
    print(request.json)
    data = request.json
    lon, lat = float(data['longitude']), float(data['latitude'])
    [lon1, lat1, lon2, lat2], lon_space, lat_space = get_bb(lon, lat)
    start = time.perf_counter()
    sat_img = get_sat_image([lon1, lat1, lon2, lat2])
    end = time.perf_counter()
    print(f"Sat image {end - start:0.4f} seconds")
    start = time.perf_counter()
    divided_imgs, divided_imgs_names, cord_list = divede_sat_img(sat_img, lon_space, lat_space)
    end = time.perf_counter()
    print(f"division image {end - start:0.4f} seconds")
    start = time.perf_counter()
    preds, score_matrix = get_prediction(divided_imgs)
    end = time.perf_counter()
    print(f"Prediction {end - start:0.4f} seconds")
    res = {"response": []}
    climate_json = fetch_climate_data(lon,lat)
    possible_good_products = get_good_products(climate_json)
    
    
    # res["response"].append({"preds": preds, "img_ids": divided_imgs_names, "coords": cord_list, "score_matrix": score_matrix, 
    #                         "general_info": climate_json, "product_info": possible_good_products})
    score_array = []
    for k,v in score_matrix.items():
        i,j = k.split("_")
        c = [elt for elt in cord_list[int(i)*39 + int(j)]]
        score_array.append([str(elt) for elt in [round(score_matrix[k],2)] +c])
    
    res["response"].append({"score_matrix": score_array, "general_info": climate_json, "product_info": possible_good_products})
    response = jsonify(res)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return make_response(response, 201)


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8080)
