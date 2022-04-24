import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { Wrapper, Status } from '@googlemaps/react-wrapper';
import LandDetailCard from 'productive-agriculture/components/LandDetailCard';
import LandGeneralCard from 'productive-agriculture/components/LandGeneralCard';
import GoogleMaps from 'productive-agriculture/components/GoogleMaps';
import BasicGoogleMaps from 'productive-agriculture/components/BasicGoogleMaps';
import styles from '../../styles/LandScorePage.module.css';
import wheatPic from '../../assets/wheat.png';
import cornPic from '../../assets/corn.png';
import cottonPic from '../../assets/cotton.png';
import orangePic from '../../assets/orange.png';
import peasPic from '../../assets/peas.png';
import sugarBeetPic from '../../assets/sugar-beet.png';
import sunflowerPic from '../../assets/sunflower.png';
import tomatoPic from '../../assets/tomato.png';

import { StaticImageData } from 'next/image';

const ProductTypes: {[id: string] : StaticImageData} = {
  wheat: wheatPic,
  corn: cornPic,
  cotton: cottonPic,
  orange: orangePic,
  pea: peasPic,
  sugar_beet: sugarBeetPic,
  sunflower: sunflowerPic,
  tomato: tomatoPic,
};

function LandScorePage() {
  const router = useRouter();
  const { longitude, latitude } = router.query;
  const [landDetails, setLandDetails] = useState<Array<string[]>>([]);
  // const [weatherDetails, setWeatherDetails] = useState<Array<string>>([]);
  const [productInfo, setProductInfo] = useState<Array<string[]>>([]);
  const [selectedLand, setSelectedLand] = useState<Array<string>>([]);

  const headers = new Headers();
  headers.append(`Access-Control-Allow-Origin`, `http://localhost:3000`);
  headers.append(`Access-Control-Allow-Credentials`, `true`);
  headers.append(`Content-Type`, `application/json; charset=UTF-8`);

  const url = `http://localhost:8080/`;
  const data = {
    longitude,
    latitude,
  };

  useEffect(() => {
    fetch(url, {
      method: `POST`,
      mode: `cors`,
      headers,
      body: JSON.stringify(data),
    })
      .then(async (response) => response.json())
      .then((json) => [json.response])
      .then((res) => {
        const { general_info, product_info, score_matrix } = res[0][0];
        const general_info_data: Array<string[]> = [];
        score_matrix.forEach((row: any) => {
          let quality = ``;
          if (row[0] < 40) {
            quality = `Low Quality`;
          } else if (row[0] >= 40 && row[0] < 60) {
            quality = `Mid Quality`;
          } else {
            quality = `High Quality`;
          }
          const new_row: Array<string> = [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            general_info.min_temp,
            general_info.max_temp,
            general_info.av_temp,
            general_info.annual_rain_mm,
            general_info.monthly_average_wetday,
            general_info.yearly_frost_free,
            general_info.av_humidity,
            general_info.av_solar_fraction,
            general_info.annual_evapotranspiration_mm,
            quality,
          ];
          general_info_data.push(new_row);
        });
        setSelectedLand(general_info_data[0]);
        setLandDetails(general_info_data);
        setProductInfo(product_info);
      });
  });

  const changeSelectedCardIndex = (newIndex: number) => {
    setSelectedLand(landDetails[newIndex])
  };

  return (
    <div className={styles.landScorePageLayout}>
      <div className={styles.upperPart}>
        {/* <Wrapper apiKey="AIzaSyDRAjTzslW-wnzn_WXU64vDYV6s1NAzaGI">
          <GoogleMaps />
        </Wrapper> */}
        <LandDetailCard
          productPhoto={ProductTypes}
          productName={productInfo}
          landScore={selectedLand[0]}
          lon1={selectedLand[1]}
          lat1={selectedLand[2]}
          lon2={selectedLand[3]}
          lat2={selectedLand[4]}
          minTemp={selectedLand[5]}
          maxTemp={selectedLand[6]}
          avgTemp={selectedLand[7]}
          annualRain={selectedLand[8]}
          monthlyAvgWetDay={selectedLand[9]}
          yearlyFrostFreeDay={selectedLand[10]}
          avgHumidity={selectedLand[11]}
          avgSolarFraction={selectedLand[12]}
          annualEvapotranspiration={selectedLand[13]}
        />
      </div>
      {/* {Object.keys(landDetails).map((land: any, index: number) => ( */}
      {landDetails.map((land: any, index: number) => (
        // eslint-disable-next-line react/no-array-index-key
        <div key={index} className={styles.belowPart}>
          <div className={styles.genericLandCard}>
            <LandGeneralCard
              productPhoto={ProductTypes}
              productName={productInfo}
              landScore={land[0]}
              lon1={selectedLand[1]}
              lat1={selectedLand[2]}
              lon2={selectedLand[3]}
              lat2={selectedLand[4]}
              landQuality={land[14]}
              cardIndex={index}
              changeSelectedCardIndex={changeSelectedCardIndex}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default LandScorePage;
