function BasicGoogleMaps() {
  const myLatlng = new google.maps.LatLng(-34.397, 150.644);
  const mapOptions = {
    zoom: 8,
    center: myLatlng,
    mapTypeId: `satellite`,
  };
  const mapElements = document.getElementById(`map`);
  const map = mapElements ? new google.maps.Map(mapElements, mapOptions) : null;
  return map;
}

export default BasicGoogleMaps;
