import React, { ChangeEvent, useState } from 'react';
import { useRouter } from 'next/router';
import formstyle from '../styles/Forms.module.css';
import styles from '../styles/Home.module.css';

function Form() {
  const router = useRouter();
  const [longitude, setLongitude] = useState(``);
  const [latitude, setLatitude] = useState(``);

  const submitCoordinates = (event: { preventDefault: () => void }) => {
    event.preventDefault(); // don't redirect the page
    // where we'll add our form logic
    router.push({
      pathname: `/productive-agriculture`,
      query: { longitude, latitude },
    });
  };

  const onLongitudeChanged = (e: ChangeEvent<HTMLInputElement>) => {
    setLongitude(e.target.value);
  };

  const onLatitudeChanged = (e: ChangeEvent<HTMLInputElement>) => {
    setLatitude(e.target.value);
  };

  return (
    <form className={styles.grid} onSubmit={submitCoordinates}>
      <label htmlFor="name">Coordinates</label>
      <input
        className={formstyle.formTextBox}
        id="Longitude"
        type="text"
        autoComplete="Longitude"
        placeholder="Longitude"
        required
        onChange={onLongitudeChanged}
      />
      <input
        className={formstyle.formTextBox}
        id="Latitude"
        type="text"
        autoComplete="Latitude"
        placeholder="Latitude"
        required
        onChange={onLatitudeChanged}
      />
      <button className={formstyle.formButton} type="submit">
        Go
      </button>
    </form>
  );
}

export default Form;
