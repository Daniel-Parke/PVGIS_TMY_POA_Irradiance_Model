import argparse
import pandas as pd
import os
from meteo.Site import Site

def main(latitude, longitude, surface_pitch, surface_azimuth, albedo, refraction_index):
    output_dir = "output_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    site = Site(latitude=latitude, longitude=longitude, surface_pitch=surface_pitch, 
                surface_azimuth=surface_azimuth, albedo=albedo, refraction_index=refraction_index)
    
    tmy_file_name = f"{output_dir}/PVGIS_TMY_Data_{latitude}_{longitude}.csv"
    irrad_file_name = f"{output_dir}/POA_Irradiance_Data_{latitude}_{longitude}_{surface_pitch}_{surface_azimuth}.csv"
    
    site.tmy_data.to_csv(tmy_file_name)
    site.irrad_model.to_csv(irrad_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process site parameters and generate TMY and irradiance data.")
    parser.add_argument("--latitude", type=float, required=True, help="Latitude of the site.")
    parser.add_argument("--longitude", type=float, required=True, help="Longitude of the site.")
    parser.add_argument("--surface_pitch", type=float, default=35, help="Surface pitch of the site.")
    parser.add_argument("--surface_azimuth", type=float, default=0, help="Surface azimuth of the site.")
    parser.add_argument("--albedo", type=float, default=0.2, help="Ground reflectance (albedo) of the site.")
    parser.add_argument("--refraction_index", type=float, default=0.1, help="Refraction index for the site's atmosphere.")

    args = parser.parse_args()

    main(latitude=args.latitude, longitude=args.longitude, surface_pitch=args.surface_pitch, 
         surface_azimuth=args.surface_azimuth, albedo=args.albedo, refraction_index=args.refraction_index)
