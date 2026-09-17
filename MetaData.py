from PIL import ExifTags, Image
from datetime import datetime


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

'''
print('Lat : {0}'.format(decimal_coords(gpsinfo[2], gpsinfo[1])))
print('Lon : {0}'.format(decimal_coords(gpsinfo[4], gpsinfo[3])))
print('Alt : {0}'.format(gpsinfo[6]))
'''
from PIL import Image
from PIL.ExifTags import TAGS

# open the image
image = Image.open("/Users/aimerydrewery/Desktop/Work/StartUp/PetrolPrices/MetaTestWalkley.jpg")

# extracting the exif metadata
exifdata = image.getexif()

# looping through all the tags present in exifdata
for tagid in exifdata:
    
    # getting the tag name instead of tag id
    tagname = TAGS.get(tagid, tagid)

    # passing the tagid to get its respective value
    value = exifdata.get(tagid)
    
    # printing the final result
    print(f"{tagname:25}: {value}")

print('Latitude Co-ordinates : {0}'.format(decimal_coords(gpsinfo[2], gpsinfo[1])))
print('Longitude Co-ordinates : {0}'.format(decimal_coords(gpsinfo[4], gpsinfo[3])))
print('Altitude Co-ordinates : {0}'.format(gpsinfo[6]))
