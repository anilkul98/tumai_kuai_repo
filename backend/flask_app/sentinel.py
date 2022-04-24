from sentinelhub import (CRS, BBox, DataCollection, MimeType,
                         SentinelHubRequest, SHConfig, bbox_to_dimensions)
from PIL import Image

client_id = "20ea34c6-aee2-45c7-a632-a30b4ca48461"
client_secret = "e(.bAi#|QV|sh]V83Eh&oj6S5H!Z4ICxk>70.N^D"
resolution = 10

def get_sat_image(bb):
    bbox = BBox(bbox=bb, crs=CRS.WGS84)
    # bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [2.55 * sample.B04, 2.55 * sample.B03, 2.55 * sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=('2021-05-12', '2022-03-31'),
                mosaicking_order='leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=bbox,
        size=(2496,2496),
        config=config,
    )
    response = request_true_color.get_data()[0]
    return Image.fromarray(response, 'RGB')