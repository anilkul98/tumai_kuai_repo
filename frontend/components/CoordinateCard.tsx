import ccardstyle from '../styles/CoordinateCards.module.css';

function CoordinateCard() {
  return (
    <div className={ccardstyle.cardOutline}>
      <div className={ccardstyle.scoreInfoBox}>
        <div className={ccardstyle.scoreText}>Score</div>
        <div className={ccardstyle.scoreNumber}>96</div>
      </div>
      <div className={ccardstyle.infoBox}>
        <div className={ccardstyle.infoTextContainer}>
          <div>Coordinates:</div>
          <div>16</div>
          <div>1:</div>
          <div>16</div>
          <div>2:</div>
          <div>16</div>
          <div>3:</div>
          <div>16</div>
          <div>4:</div>
          <div>16</div>
          <div>5:</div>
          <div>16</div>
          <div>6:</div>
          <div>16</div>
        </div>
      </div>
    </div>
  );
}

export default CoordinateCard;
