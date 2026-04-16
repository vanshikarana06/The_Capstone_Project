import os
import time
from PIL import Image

def get_metadata(filepath):
    stat = os.stat(filepath)
    result = {
        "size_kb": round(stat.st_size / 1024, 2),
        "created": time.ctime(stat.st_ctime),
        "modified": time.ctime(stat.st_mtime)
    }

    # Try to get image EXIF data 
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if exif:
            result["software"] = exif.get(305, "Not available")
            result["camera"]   = exif.get(272, "Not available")
            result["datetime"] = exif.get(306, "Not available")
        else:
            result["exif"] = "No EXIF data found"
    except:
        result["exif"] = "Could not read EXIF"

    return result