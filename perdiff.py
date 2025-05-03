import math

# Coordinates (latitude, longitude)
your_coords = (14.202125, 120.878791)
google_coords = (14.202042, 120.878762)

def percent_difference(val1, val2):
    return abs(val1 - val2) / ((val1 + val2) / 2) * 100

# Percent difference calculations
lat_diff = percent_difference(your_coords[0], google_coords[0])
lon_diff = percent_difference(your_coords[1], google_coords[1])

print(f"Percent Difference (Latitude): {lat_diff:.9f}%")
print(f"Percent Difference (Longitude): {lon_diff:.9f}%")
