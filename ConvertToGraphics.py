import numpy as np
from PIL import Image
import pandas as pd
import colorsys, requests, math, os
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from scipy.io.wavfile import write

path = os.getcwd()
sprite_path = path + "\\Sprites"
out_path = path + '\\out'
version = "1.0.0"
target_out = "ROTMG_PRIEST"

def get_hz_tbl():
    url = "https://pages.mtu.edu/~suits/notefreqs.html"
    page = requests.get(url)
    soup = BeautifulSoup(page.text)
    table = soup.find_all('table')[1]
    headers = []
    for i in table.find_all('th'):
        title = i.text
        headers.append(title)
    hz_tbl = pd.DataFrame(columns = headers)
    for j in table.find_all('tr')[1:]:
        row_data = j.find_all('td')
        row = [i.text for i in row_data]
        length = len(hz_tbl)
        hz_tbl.loc[length] = row
    return(hz_tbl)

# Open our target image
def open_sprite(series, filename):
    # Open and convert to HSV from RGB
    im = Image.open(sprite_path + '\\' + series + "\\" + filename).convert("HSV")
    return im

# Function to convert hue into Wavelength.
def to_wavelength(hue_array):
    # CREDIT: https://stackoverflow.com/questions/11850105/hue-to-wavelength-mapping
    return(650 - 250 / 270 * hue_array)

# Convert opened image into table of buckets
def freq_table_of_img(img, buckets = 24):
    # Convert to Numpy and keep only Hue Layer
    # This is the best mapping from 3D -> 1D for color
    #  plus it ignores black and white. Just discard 0 and 1
    vec_img = np.array(img)[:,:,0]
    # get counts
    unique, counts = np.unique(vec_img, return_counts=True)
    # Pretty sure 255 and 0 are white / black. Could be wrong
    # Paint says both are 160 but that's dumb because white/black are about saturation not hue
    #  160 is like a blue
    for remove_val in [0, 255]:
        if remove_val in unique:
            res = np.where(unique==remove_val)[0][0]
            unique = np.delete(unique, res)
            counts = np.delete(counts, res)
    # remove dtype
    unique = np.array(unique.tolist())
    counts = np.array(counts.tolist())
    # separate into buckets, can be cleaner but whatever
    unique = unique / 255 # set range to (0, 1)
    unique = unique * (buckets-1)  # set range to [0, buckets-1]
    unique = np.array(unique.round().tolist(), dtype=int)
    bucket_summary = {}
    for i in range(buckets):
        bucket_summary[i] = 0
    for i in range(len(unique)):
        # if not unique[i] in bucket_summary.keys():
        #     bucket_summary[unique[i]] = 0
        bucket_summary[unique[i]] += counts[i]
    # construct DF for easy manip
    df = pd.DataFrame({'Unique' : list(bucket_summary.keys()), "Counts" : list(bucket_summary.values())})
    df['Freq'] = df['Counts'] / sum(df['Counts'])
    return(df)

# saves the plot for this img
def create_freq_plot(tbl, buckets, series, filename):
    if not os.path.exists(out_path + "\\Graphics\\" + series):
        os.mkdir(out_path + "\\Graphics\\" + series)
    fig, ax = plt.subplots()
    my_colors_hsv = [(x/(buckets-1), .5, .5) for x in range(buckets)]
    my_colors_rgb = []
    for cval in my_colors_hsv:
        my_colors_rgb += [colorsys.hsv_to_rgb(cval[0], cval[1], cval[2])]
    tbl['Freq'].plot(ax=ax, kind="bar",xlabel="buckets",ylabel='Freq',width=1,color=my_colors_rgb)
    plt.savefig(out_path + "\\Graphics\\" + series + "\\freq_plot_" + filename)
    plt.close()

# creates freq for each bucket
# If you pass a custom set of indexes with length == buckets size for custom_scale, that will be used
#  in place of every note. 
def get_freqs(buckets, middle = 47, custom_scale = None):
    hz_tbl = pd.read_csv(path + "\\hz_tbl.csv")
    HZs = np.array(hz_tbl['Frequency (Hz)'])
    # if you've defined a custom scale use it
    if custom_scale:
        Hz = HZs[custom_scale]
    else:
        # just grab all notes around middle
        halfs = int((buckets-1)/2)
        Hz = HZs[(middle-halfs):(middle+halfs+1)]
    return(Hz)

# Final step, generate waveform and save sound file
def Generate_Sound(Hz, Scalars, series, file, seconds = 1, thz = 44100, save = True):
    timesteps = np.array([x/thz for x in range(int(seconds*thz))])
    waveform = [0]*int(thz*seconds)
    for i in range(len(Hz)):
        waveform += np.sin(Hz[i] * timesteps * math.pi * 2)*Scalars[i]*32767
    if save:
        write(out_path + "\\Sounds\\" + series + "_" + file + ".wav" , thz, waveform.astype(np.int16))
    else:
        return(waveform)

# BUCKETS Needs to be ODD, this way you get one octave above and below inclusive
# 25 gets you middle C (c4) at 47, lower C (c3) at 35, and higher C (c5) at 59 
def Generate_Single_Soundwave(series, file, filetype=".png", buckets=25, middle=47, seconds=1, custom_scale = None):
    filename = file + filetype
    img = open_sprite(series, filename)
    tbl = freq_table_of_img(img, buckets)
    create_freq_plot(tbl, buckets, series, filename)
    Hz = get_freqs(buckets, middle, custom_scale)
    Scalars = np.array(list(tbl['Freq']))
    Generate_Sound(Hz, Scalars, series, file, seconds=seconds)


def convert_directory(target="Priest"):
    for fullfile in os.listdir("Sprites\\" + target):
        file = fullfile.split(".")[0]
        Generate_Single_Soundwave(target, file)

#convert_directory()

# Supply the gif folder with 1 gif named target.gif
def convert_gif(target_folder="pcave_gif", num_frames = 20, seconds = .5, buckets = 25, middle = 47, thz=44100, smoothing = 0, custom_scale = None):
    with Image.open(sprite_path + "\\" + target_folder + "\\target.gif") as im:
        for i in range(num_frames):
            im.seek(im.n_frames // num_frames * i)
            im.save(sprite_path + "\\" + target_folder + '\\{}.png'.format(i))
    cont_waveform = np.array([])
    for i in range(num_frames):
        filename = "{}.png".format(i)
        img = open_sprite(target_folder, filename)
        tbl = freq_table_of_img(img, buckets)
        create_freq_plot(tbl, buckets, target_folder, filename)
        Hz = get_freqs(buckets, middle, custom_scale)
        Scalars = np.array(list(tbl['Freq']))
        if smoothing == 0:
            cont_waveform =  np.append( cont_waveform, Generate_Sound(Hz, Scalars, target_folder, str(i), seconds=seconds, thz = thz, save=False))
        else:
            if len(cont_waveform) == 0:
                cont_waveform = Generate_Sound(Hz, Scalars, target_folder, str(i), seconds=seconds, thz = thz, save=False)
            else:
                new = Generate_Sound(Hz, Scalars, target_folder, str(i), seconds=seconds, thz = thz, save=False)
                cont_waveform_smooth = cont_waveform[-smoothing:]
                cont_waveform_sub = cont_waveform[:-smoothing]
                new_smooth = new[:smoothing]
                new_sub = new[smoothing:]
                ranges = ((np.array(range(smoothing))/smoothing) - .5)*12
                cont_sigmoid = abs((1 / (1 + np.exp(-ranges))) - 1)
                new_sigmoid = 1 / (1 + np.exp(-ranges))
                smoothed = cont_waveform_smooth * cont_sigmoid + new_smooth * new_sigmoid
                cont_waveform = np.append(cont_waveform_sub, smoothed)
                cont_waveform = np.append(cont_waveform, new_sub)
    write(out_path + "\\Sounds\\" + target_folder + "_" + target_folder + ".wav" , thz, cont_waveform.astype(np.int16))

#convert_gif(target_folder = "SmoothColor", num_frames = 60, seconds = .1, smoothing = 1000)

