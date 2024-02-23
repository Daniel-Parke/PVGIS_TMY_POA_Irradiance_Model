"""Import functionality required to calculate radiation values"""
import pandas as pd
from numpy import radians, degrees, cos, sin, arccos, pi
import numpy as np


def calc_declination(n_day: int) -> float:
    """
    Calculates the solar declination angle for a given day of the year.

    Parameters:
        n_day (int): Day of the year (1 through 365 or 366).

    Returns:
        float: Solar declination angle in degrees.
    """
    return 23.45 * sin(radians((360 / 365) * (284 + n_day)))


def calc_time_correction(n_day: int) -> float:
    """
    Calculates the equation of time correction factor.

    Parameters:
        n_day (int): Day of the year.

    Returns:
        float: Time correction factor in minutes.
    """
    B = radians(360 * (n_day - 1) / 365)
    return 3.82 * (
        0.000075
        + 0.001868 * cos(B)
        - 0.032077 * sin(B)
        - 0.014615 * cos(2 * B)
        - 0.04089 * sin(2 * B)
    )


def calc_solar_time(n_day: int, civil_time: float, longitude: float, 
                    timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the solar time at a given location and time.

    Parameters:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        longitude (float): Longitude of the location.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Solar time in hours.
    """
    time_correction = calc_time_correction(n_day)
    return (
        (civil_time + ((timestep / 60) / 2))
        + (longitude / 15)
        - tmz_hrs_east
        + time_correction
    )


def calc_hour_angle(n_day: int, civil_time: float, longitude: float, 
                    timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the solar hour angle at a given time and location.

    Parameters:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        longitude (float): Longitude of the location.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Hour angle in degrees.
    """
    solar_time = calc_solar_time(n_day, civil_time, longitude, timestep, tmz_hrs_east)

    return (solar_time - 12) * 15


def calc_aoi(n_day: int, civil_time: float, latitude: float, longitude: float, 
             surface_azimuth: float, surface_pitch: float, 
             timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the angle of incidence of solar radiation on a given surface.

    Parameters:
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from the horizontal.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Angle of incidence in degrees.
    """
    hour_angle_rad = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    declination_rad = radians(calc_declination(n_day))

    latitude_rad = radians(latitude)
    surface_pitch_rad = radians(surface_pitch)
    surface_azimuth_rad = radians(surface_azimuth)

    aoi = arccos(
        (sin(declination_rad) * sin(latitude_rad) * cos(surface_pitch_rad))
        - (
            sin(declination_rad)
            * cos(latitude_rad)
            * sin(surface_pitch_rad)
            * cos(surface_azimuth_rad)
        )
        + (
            cos(declination_rad)
            * cos(latitude_rad)
            * cos(surface_pitch_rad)
            * cos(hour_angle_rad)
        )
        + (
            cos(declination_rad)
            * sin(latitude_rad)
            * sin(surface_pitch_rad)
            * cos(surface_azimuth_rad)
            * cos(hour_angle_rad)
        )
        + (
            cos(declination_rad)
            * sin(surface_pitch_rad)
            * sin(surface_azimuth_rad)
            * sin(hour_angle_rad)
        )
    )

    return degrees(aoi)


def calc_zenith(latitude: float, longitude: float, n_day: int, civil_time: float, 
                timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the solar zenith angle at a given time and location.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Zenith angle in degrees.
    """
    latitude_rad = radians(latitude)
    declination_rad = radians(calc_declination(n_day))
    hour_angle_rad = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    return degrees(
        arccos(
            (cos(latitude_rad) * cos(declination_rad) * cos(hour_angle_rad))
            + (sin(latitude_rad) * sin(declination_rad))
        )
    )


def calc_et_normal_radiation(n_day: int) -> float:
    """
    Calculates the extraterrestrial normal radiation for a given day of the year.

    Parameters:
        n_day (int): Day of the year.

    Returns:
        float: Extraterrestrial normal radiation in W/m^2.
    """
    solar_constant = 1367
    return solar_constant * (1 + 0.033 * cos(radians((360 * n_day) / 365)))


def calc_et_horizontal_radiation(
    latitude: float, longitude: float, n_day: int, civil_time: float, 
    timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the extraterrestrial horizontal radiation over a specified timestep.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours at the beginning of the timestep.
        timestep (int, optional): Duration of the timestep in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Extraterrestrial horizontal radiation in W/m^2 for the timestep.
    """
    civil_time_2 = civil_time + (timestep / 60)

    declination_rad = radians(calc_declination(n_day))
    latitude_rad = radians(latitude)

    hour_angle_1 = radians(
        calc_hour_angle(n_day, civil_time, longitude, timestep, tmz_hrs_east)
    )
    hour_angle_2 = radians(
        calc_hour_angle(n_day, civil_time_2, longitude, timestep, tmz_hrs_east)
    )

    et_horizontal_radiation = (
        (12 / pi * calc_et_normal_radiation(n_day))
        * (
            (
                cos(latitude_rad)
                * cos(declination_rad)
                * (sin(hour_angle_2) - sin(hour_angle_1))
                + (
                    (hour_angle_2 - hour_angle_1)
                    * sin(latitude_rad)
                    * sin(declination_rad)
                )
            )
        )
    ) * (60 / timestep)

    et_horizontal_radiation = et_horizontal_radiation

    # Ensure non-negative values
    et_horizontal_radiation = np.where(
        et_horizontal_radiation > 0, et_horizontal_radiation, 0
    )  

    return et_horizontal_radiation


def calc_beam_radiation(dni: float, n_day: int, civil_time: float, latitude: float, longitude: float, 
                        surface_azimuth: float, surface_pitch: float, timestep: int = 60, 
                        tmz_hrs_east: float = 0) -> float:
    """
    Calculates the beam component of solar radiation on a tilted surface.

    Parameters:
        dni (float): Direct normal irradiance in W/m^2.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Beam irradiance on the surface in W/m^2.
    """
    aoi_rad = radians(
        calc_aoi(
            n_day,
            civil_time,
            latitude,
            longitude,
            surface_azimuth,
            surface_pitch,
            timestep,
            tmz_hrs_east,
        )
    )

    # Calculate beam irradiance and ensure it is not calculated for angles > 85 degrees
    e_beam = dni * cos(aoi_rad)
    e_beam = np.where(degrees(aoi_rad) > 85, 0, e_beam)
    e_beam = np.where(e_beam < 0, 0, e_beam)  # Ensure non-negative values

    return e_beam


def calc_diffuse_radiation(dhi: float, ghi: float, surface_pitch: float, latitude: float, 
                           longitude: float, n_day: int, civil_time: float, timestep: int = 60,
                           tmz_hrs_east: float = 0) -> float:
    """
    Calculates the diffuse component of solar radiation on a tilted surface.

    Parameters:
        dhi (float): Diffuse horizontal irradiance in W/m^2.
        ghi (float): Global horizontal irradiance in W/m^2.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Diffuse irradiance on the surface in W/m^2.
    """
    surface_pitch_rad = radians(surface_pitch)
    zenith_rad = radians(
        calc_zenith(latitude, longitude, n_day, civil_time, timestep, tmz_hrs_east)
    )

    e_diffuse = (dhi * ((1 + cos(surface_pitch_rad)) / 2)) + (
        ghi * ((0.12 * zenith_rad) - 0.04) * (1 - cos(surface_pitch_rad)) / 2
    )
    e_diffuse = np.where(
        degrees(zenith_rad) > 85, 0, e_diffuse
    )  # Ensure irradiance is not calculated for zenith angles > 85 degrees
    return e_diffuse


def calc_ground_radiation(ghi: float, surface_pitch: float, albedo: float = 0.2) -> float:
    """
    Calculates the ground-reflected component of solar radiation on a tilted surface.

    Parameters:
        ghi (float): Global horizontal irradiance in W/m^2.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        albedo (float, optional): Ground reflectance factor. Defaults to 0.2.

    Returns:
        float: Ground-reflected irradiance on the surface in W/m^2.
    """
    surface_pitch_rad = radians(surface_pitch)
    e_ground = ghi * albedo * ((1 - cos(surface_pitch_rad)) / 2)
    return e_ground


def calc_poa_radiation(dni: float, dhi: float, ghi: float, n_day: int, civil_time: float,
                       latitude: float, longitude: float, surface_azimuth: float, surface_pitch: float,
                       albedo: float = 0.2, timestep: int = 60, tmz_hrs_east: float = 0) -> float:
    """
    Calculates the plane of array (POA) irradiance, considering beam, diffuse, and ground-reflected components.

    Parameters:
        dni (float): Direct normal irradiance in W/m^2.
        dhi (float): Diffuse horizontal irradiance in W/m^2.
        ghi (float): Global horizontal irradiance in W/m^2.
        n_day (int): Day of the year.
        civil_time (float): Civil time in hours.
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        surface_azimuth (float): Azimuth angle of the surface from north.
        surface_pitch (float): Tilt angle of the surface from horizontal.
        albedo (float, optional): Ground reflectance factor. Defaults to 0.2.
        timestep (int, optional): Time step in minutes. Defaults to 60.
        tmz_hrs_east (float, optional): Time zone hours east of GMT. Defaults to 0.

    Returns:
        float: Total irradiance on the plane of array in W/m^2.
    """
    # Beam Radiation Calculation
    e_beam = calc_beam_radiation(
        dni,
        n_day,
        civil_time,
        latitude,
        longitude,
        surface_azimuth,
        surface_pitch,
        timestep,
        tmz_hrs_east,
    )

    # Diffuse Radiation Calculation
    e_diffuse = calc_diffuse_radiation(
        dhi,
        ghi,
        surface_pitch,
        latitude,
        longitude,
        n_day,
        civil_time,
        timestep,
        tmz_hrs_east,
    )

    # Ground-reflected Radiation Calculation
    e_ground = calc_ground_radiation(ghi, surface_pitch, albedo)

    e_poa = (
        e_beam + e_diffuse + e_ground
    )

    return e_poa


def iam_losses(aoi: float, refraction_index: float = 0.1) -> float:
    """
    Calculates the incident angle modifier (IAM) losses for solar panels based on the angle of incidence (AOI).

    Parameters:
    - aoi (float): Angle of incidence in degrees.
    - refraction_index (float, optional): Refractive index of the panel's surface material. Defaults to 0.1.

    Returns:
    - float: IAM loss factor for each time step.

    References
    ----------
    .. [1] Souka A.F., Safwat H.H., "Determination of the optimum
       orientations for the double exposure flat-plate collector and its
       reflections". Solar Energy vol .10, pp 170-174. 1966.

    .. [2] ASHRAE standard 93-77

    .. [3] PVsyst 7 Help.
       https://www.pvsyst.com/help/index.html?iam_loss.htm retrieved on
       January 30, 2024
    """
    iam_factor = 1 - refraction_index * ((1 / cos(radians(aoi))) - 1)
    iam_factor = np.where(aoi > 85, 0, iam_factor)
    return iam_factor


def calc_solar_model(data: pd.DataFrame, latitude: float, longitude: float, 
                     surface_pitch: float = 35, surface_azimuth: float = 0, albedo: float = 0.2, 
                     refraction_index: float = 0.1, timestep: int = 60, 
                     tmz_hrs_east: float = 0) -> pd.DataFrame:
    """
    Processes TMY data to simulate solar PV system performance over a typical meteorological year.

    Parameters:
        data (pd.DataFrame): TMY data including irradiance and temperature.
        latitude (float): Site latitude.
        longitude (float): Site longitude.
        pv_kwp (float): Rated power of the PV system in kWp.
        surface_pitch (float): Tilt angle of the PV panel from the horizontal plane.
        surface_azimuth (float): Orientation of the PV panel from true north.
        lifespan (int): Estimated lifespan of the PV system in years.
        pv_derating (float): Derating factor of the PV system to account for end-of-life performance.
        albedo (float): Ground reflectance factor.
        cell_temp_coeff (float): Temperature coefficient of the PV cell per degree Celsius.
        refraction_index (float): Refractive index of the panel's cover.
        e_poa_STC (float): Plane of array irradiance under standard test conditions.
        cell_temp_STC (float): Cell temperature under standard test conditions.
        timestep (int): Time interval for calculations in minutes.
        tmz_hrs_east (float): Time zone offset from GMT in hours.

    Returns:
        pd.DataFrame: Enhanced TMY data with added columns for solar radiation calculations and PV system performance metrics.
    """
    # Convert required DataFrame columns to numpy arrays for calculations
    hour_of_day = data['Hour_of_Day'].to_numpy()
    day_of_year = data['Day_of_Year'].to_numpy()
    week_of_year = data['Week_of_Year'].to_numpy()
    month_of_year = data['Month_of_Year'].to_numpy()
    Gb_n = data['Gb(n)'].to_numpy()
    Gd_h = data['Gd(h)'].to_numpy()
    G_h = data['G(h)'].to_numpy()
    Ambient_Temperature_C = data['T2m'].to_numpy()
    wind_speed = data["WS10m"].to_numpy()

    # Perform calculations using numpy arrays
    declination_angle = calc_declination(day_of_year)
    solar_time = calc_solar_time(day_of_year, hour_of_day, longitude, timestep, tmz_hrs_east)
    hour_angle = calc_hour_angle(day_of_year, hour_of_day, longitude, timestep, tmz_hrs_east)
    aoi = calc_aoi(day_of_year, hour_of_day, latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east)
    zenith_angle = calc_zenith(latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    e_beam_w_m2 = calc_beam_radiation(Gb_n, day_of_year, hour_of_day, latitude, longitude, surface_azimuth, surface_pitch, timestep, tmz_hrs_east)
    e_diffuse_w_m2 = calc_diffuse_radiation(Gd_h, G_h, surface_pitch, latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    e_ground_w_m2 = calc_ground_radiation(G_h, surface_pitch, albedo)
    e_poa_w_m2 = e_beam_w_m2 + e_diffuse_w_m2 + e_ground_w_m2
    et_hrad_w_m2 = calc_et_horizontal_radiation(latitude, longitude, day_of_year, hour_of_day, timestep, tmz_hrs_east)
    panel_poa_w_m2 = e_beam_w_m2 * iam_losses(aoi, refraction_index) + e_diffuse_w_m2 + e_ground_w_m2

    # Construct a new DataFrame from the calculated arrays
    results = pd.DataFrame({
        "E_POA_kWm2": e_poa_w_m2 / 1000,                     # Convert to kW
        "Panel_POA_kWm2": panel_poa_w_m2 / 1000,             # Convert to kW 
        "E_Beam_kWm2": e_beam_w_m2 / 1000,                   # Convert to kW
        "E_Diffuse_kWm2": e_diffuse_w_m2 / 1000,             # Convert to kW
        "E_Ground_kWm2": e_ground_w_m2 / 1000,               # Convert to kW             
        "ET_HRad_kWm2": et_hrad_w_m2 / 1000,                 # Convert to kW
        "Wind_Speed_ms": wind_speed,
        "Ambient_Temperature_C": Ambient_Temperature_C,
        "Declination_Angle": declination_angle,
        "Solar_Time": solar_time,
        "Hour_Angle": hour_angle,
        "AOI": aoi,
        "Zenith_Angle": zenith_angle,
        "Hour_of_Day": hour_of_day,
        "Day_of_Year": day_of_year,
        "Week_of_Year": week_of_year,
        "Month_of_Year": month_of_year,
    })

    return results


class SummaryGrouped:
    """
    Organizes grouped summary statistics of PV system performance.
    
    Parameters:
        summaries (dict): Dictionary with time grouping as keys and summary statistics DataFrames as values.
    """

    def __init__(self, summaries):
        for key, df in summaries.items():
            setattr(self, key.lower(), df)


def rad_data_grouped(model_results: pd.DataFrame) -> SummaryGrouped:
    """
    Generates grouped statistics of site radiation data.

    Parameters:
        model_results (pd.DataFrame): DataFrame containing model results.

    Returns:
        SummaryGrouped: Object containing DataFrames of grouped statistics.
    """
    # Define the groupings for different human timeframes
    groupings = {
        "Hourly": "Hour_of_Day",
        "Daily": "Day_of_Year",
        "Weekly": "Week_of_Year",
        "Monthly": "Month_of_Year",
        "Quarterly": model_results["Month_of_Year"].apply(lambda x: (x - 1) // 3 + 1),
    }

    # Columns to sum and to calculate the mean
    columns_to_sum = [
        "E_POA_kWm2",
        "Panel_POA_kWm2",
        "E_Beam_kWm2",
        "E_Diffuse_kWm2",
        "E_Ground_kWm2",
        "ET_HRad_kWm2",
    ]
    columns_to_mean = ["Ambient_Temperature_C", "Wind_Speed_ms"]

    summaries = {}

    # Gets Hourly and Hour of Day from .items() tuple list
    for timeframe, group_by in groupings.items():
        grouped = model_results.groupby(group_by)

        # Summing specified columns and rounding
        summed = round(grouped[columns_to_sum].sum(), 3)

        # Calculating the mean for specified columns and rounding
        meaned = round(grouped[columns_to_mean].mean(), 3)

        # Combine the summed and meaned results into a single DataFrame
        summary_df = pd.concat([summed, meaned], axis=1)

        # Adds summary dataframe to dictionary with timeframe key
        summaries[timeframe] = summary_df

    # Return an instance of SummaryGrouped with summaries as attributes
    return SummaryGrouped(summaries)
