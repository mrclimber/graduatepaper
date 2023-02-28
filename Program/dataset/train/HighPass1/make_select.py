import glob
import random
from skimage import io
from tifffile import imwrite

good_files = glob.glob("/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/all/good/*")
bad_files = glob.glob("/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/all/bad/*")

base_good = "/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/all/good/"
base_bad = "/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/all/bad/"

select_good = "/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/select/good/"
select_bad = "/home/masaki/graduatepaper/data/model/new/train/HighPass1/2345/select/bad/"


random.shuffle(good_files)
random.shuffle(bad_files)

for i in range(10000):
    name_good = good_files[i]
    name_bad = bad_files[i]
    new_good = name_good.split('/')
    new_bad = name_bad.split('/')
    img_good = io.imread(name_good)
    img_bad = io.imread(name_bad)
    name_good = new_good[12]
    name_bad = new_bad[12]
    good = select_good + name_good
    bad = select_bad + name_bad
    imwrite(select_good + name_good,img_good)
    imwrite(select_bad + name_bad,img_bad)
