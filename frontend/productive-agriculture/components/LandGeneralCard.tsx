/* eslint-disable jsx-a11y/click-events-have-key-events */
import Image, { StaticImageData } from 'next/image';
import PropTypes from 'prop-types';
import { useRouter } from 'next/router';
import React from 'react';
import styles from '../../styles/LandDetailCard.module.css';

interface LandGeneralCardProps {
  productPhoto: {[id: string] : StaticImageData};
  productName: Array<Array<string>>;
  landScore: string;
  landQuality: string;
  cardIndex: number;
  lon1: string;
  lat1: string;
  lon2: string;
  lat2: string;
  changeSelectedCardIndex: (newIndex: number) => void;
}

function LandGeneralCard(props: LandGeneralCardProps) {
  // rain in millimeters
  const {
    productPhoto,
    productName,
    landScore,
    lon1,
    lat1,
    lon2,
    lat2,
    landQuality,
    cardIndex,
    changeSelectedCardIndex
  } = props;
  const router = useRouter();
  const cardClick = () => {
    changeSelectedCardIndex(cardIndex);
  };

  return (
    // eslint-disable-next-line jsx-a11y/no-static-element-interactions
    <div className={styles.landCardStyle} onClick={() => cardClick()}>
      <div className={styles.contentOrder}>
        {/* <div className="mx-5">
          <Image src={productPhoto[productName[0][0]]} alt="logo" width="118.13" height="118.13" />
        </div> */}
        <div className={styles.cardInfoDetailContainer}>
          <div className={styles.productName}>Products Suggested: {productName.map((product) => product[0] + ', ')}</div>
          <div className={styles.landScore}>
            Overall Land Score: {landScore}
          </div>
          <div className={styles.landScore}>
            Land Coordinates: {lon1}, {lat1}, {lon2}, {lat2}
          </div>
          <div className={styles.landScore}>
            Overall Land Quality: {landQuality}
          </div>
        </div>
      </div>
    </div>
  );
}

LandGeneralCard.defaultProps = {
  // productPhoto: ``,
  productName: [``],
  landScore: ``,
  landQuality: ``,
  cardIndex: 0,
};

LandGeneralCard.propTypes = {
  // productPhoto: PropTypes.string,
  productName: PropTypes.arrayOf(PropTypes.string),
  landScore: PropTypes.string,
  landQuality: PropTypes.string,
  cardIndex: PropTypes.number,
};

export default LandGeneralCard;
