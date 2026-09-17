

from PIL import ExifTags, Image


def decimal_coords(coords, ref):
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -1 * decimal_degrees
    return decimal_degrees


GPSINFO_TAG = next(
    tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"
)

path = r'/Users/aimerydrewery/Desktop/Work/StartUp/PetrolPrices/MetaTestWalkley.jpg'
image = Image.open(path)
info = image.getexif()

gpsinfo = info.get_ifd(GPSINFO_TAG)


print('Lat : {0}'.format(decimal_coords(gpsinfo[2], gpsinfo[1])))
print('Lon : {0}'.format(decimal_coords(gpsinfo[4], gpsinfo[3])))
print('Alt : {0}'.format(gpsinfo[6]))

