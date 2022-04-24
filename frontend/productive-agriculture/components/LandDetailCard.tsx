/* eslint-disable jsx-a11y/click-events-have-key-events */
import Image, { StaticImageData } from 'next/image';
import PropTypes from 'prop-types';
// import { useRouter } from 'next/router';
import React from 'react';
import styles from '../../styles/LandDetailCard.module.css';

interface LandDetailCardProps {
  productPhoto: {[id: string] : StaticImageData};
  productName: Array<Array<string>>;
  landScore: string;
  lon1: string;
  lat1: string;
  lon2: string;
  lat2: string;
  minTemp: string;
  maxTemp: string;
  avgTemp: string;
  annualRain: string;
  monthlyAvgWetDay: string;
  yearlyFrostFreeDay: string;
  avgHumidity: string;
  avgSolarFraction: string;
  annualEvapotranspiration: string;
}

function LandDetailCard(props: LandDetailCardProps) {
  const {
    productPhoto,
    productName,
    landScore,
    lon1,
    lat1,
    lon2,
    lat2,
    minTemp,
    maxTemp,
    avgTemp,
    annualRain,
    monthlyAvgWetDay,
    yearlyFrostFreeDay,
    avgHumidity,
    avgSolarFraction,
    annualEvapotranspiration,
  } = props;
  // const router = useRouter();
  // const cardClick = () => {
  //   router.push(`/`);
  // };

  return (
    // <div className={styles.landCardStyle} onClick={() => cardClick()}>
    // eslint-disable-next-line jsx-a11y/no-static-element-interactions
    <div className={styles.landCardStyle}>
      <div className={styles.contentOrder}>
        {/* <div className="mx-5">
          <Image src={productPhoto[productName[0][0]]} alt="logo" width="118.13" height="118.13" />
        </div> */}
        <div className={styles.cardInfoDetailContainer}>
          <div className={styles.productName}>Products: {productName.map((product) => product[0] + ', ')}</div>
          <div className={styles.landScore}>
            Overall Land Score: {landScore}
          </div>
          <div className={styles.landScore}>
            Land Coordinates: {landScore}
          </div>
          <div className={styles.landScore}>
            Land Coordinates: {lon1}, {lat1}, {lon2}, {lat2}
          </div>
          <div className={styles.landScore}>
            Monthly Minimum Temperature: {minTemp}
          </div>
          <div className={styles.landScore}>
            Monthly Maximum Temperature: {maxTemp}
          </div>
          <div className={styles.landScore}>
            Monthly Average Temperature: {avgTemp}
          </div>
          <div className={styles.landScore}>
            Annual Rain (in millimeters): {annualRain}
          </div>
          <div className={styles.landScore}>
            Monthly Average Wet Day: {monthlyAvgWetDay}
          </div>
          <div className={styles.landScore}>
            Yearly Frost Free Day: {yearlyFrostFreeDay}
          </div>
          <div className={styles.landScore}>
            Average Humidity: {avgHumidity}
          </div>
          <div className={styles.landScore}>
            Average Solar Fraction: {avgSolarFraction}
          </div>
          <div className={styles.landScore}>
            Annual Evapotranspiration: {annualEvapotranspiration}
          </div>
        </div>
      </div>
    </div>
  );
}

LandDetailCard.defaultProps = {
  // productPhoto: ``,
  productName: [``],
  landScore: ``,
  minTemp: ``,
  maxTemp: ``,
  avgTemp: ``,
  annualRain: ``,
  monthlyAvgWetDay: ``,
  yearlyFrostFreeDay: ``,
  avgHumidity: ``,
  avgSolarFraction: ``,
  annualEvapotranspiration: ``,
};

LandDetailCard.propTypes = {
  // productPhoto: PropTypes.string,
  productName: PropTypes.arrayOf(PropTypes.string),
  landScore: PropTypes.string,
  minTemp: PropTypes.string,
  maxTemp: PropTypes.string,
  avgTemp: PropTypes.string,
  annualRain: PropTypes.string,
  monthlyAvgWetDay: PropTypes.string,
  yearlyFrostFreeDay: PropTypes.string,
  avgHumidity: PropTypes.string,
  avgSolarFraction: PropTypes.string,
  annualEvapotranspiration: PropTypes.string,
};

export default LandDetailCard;
