from dataclasses import dataclass, field
import pandas as pd
import logging

from meteo.jrc_tmy import get_jrc_tmy
from solar.solar_radiation import calc_solar_model, rad_data_grouped

from misc.log_config import configure_logging
configure_logging()

@dataclass
class Site:
    """
    A class representing a site.

    Attributes:
        latitude (float): Latitude of the site.
        longitude (float): Longitude of the site.
        surface_pitch (float): Pitch of the modelled plane in degrees.
        surface_azimuth (float): Azimuth of the modelled plane in degrees.
        albedo (float): Albedo of the modelled plane.
        refraction_index (float): Refraction index of the modelled plane.
        tmz_hrs_east (int): Timezone hours east of GMT, UTC=0.
        timestep (int): Timestep of weather data in minutes
        tmy_data (pd.DataFrame): DataFrame containing TMY data for the site.
        irrad_model (pd.DataFrame): DataFrame containing POA irradiance values for the site.
        rad_data (pd.DataFrame): DataFrame containing statistical grouping of POA irradiance values for the site.
    """
    latitude: float = 54.60452
    longitude: float = -5.92860
    surface_pitch: float = 35
    surface_azimuth: float = 0
    albedo: float = 0.2
    refraction_index: float = 0.1
    tmz_hrs_east: int = 0
    timestep: int = 60
    tmy_data: pd.DataFrame = field(default=None, init=False)
    irrad_model: pd.DataFrame = field(default=None, init=False)
    rad_data: pd.DataFrame = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization method.
        Fetches TMY data for the site and logs a message.
        """
        # Fetch TMY data from PVGIS
        logging.info(f'Fetching TMY data for latitude: {self.latitude}, longitude: {self.longitude}') 
        self.tmy_data = get_jrc_tmy(self.latitude, self.longitude)
        logging.info(f'TMY data obtained for: {self.latitude}, longitude: {self.longitude}')
        logging.info("*******************")

        # Calculate POA irradiance values
        logging.info(f'Calculating POA irradiance for latitude: {self.latitude}, longitude: {self.longitude}')
        self.irrad_model = calc_solar_model(
            self.tmy_data,
            self.latitude,
            self.longitude,
            self.surface_pitch,
            self.surface_azimuth,
            self.albedo,
            self.refraction_index,
            self.timestep,
            self.tmz_hrs_east
        )
        logging.info(f'POA Irradiance for: {self.latitude}, longitude: {self.longitude} calculated successfully')
        logging.info("*******************")
        logging.info("Generating model statistical grouping.")
        self.rad_data = rad_data_grouped(self.irrad_model)
        logging.info("Model statistical grouping completed.")
        logging.info("*******************")


