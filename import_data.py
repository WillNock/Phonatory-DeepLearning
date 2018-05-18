"""
Author: Will Nock, ISIS, Vanderbilt School of Engineering
Speech Disorder Diagnosing with Deep Learning

File: import.py :
This file handles importing data in an organized fashion from the Box Sync folder
"""


from os import listdir
from os.path import join
from scipy.misc import imread
from sklearn.utils import shuffle


# ------------- HELPER function ------------------
# Returns the type of Diagnosis/"label", in all caps (e.g. ADSD, NORMAL, etc.)
# for a sample name
def getLabel(sample):
    ret = ""
    splitIndex = sample.find(" - ")
    if splitIndex == -1:
        splitIndex = sample.find("_")
    sample_pre = sample[:splitIndex]
    for char in sample_pre:
        if char.isalpha():
            ret += char.capitalize()
    return ret


def populateData():
    data = {} # Dictionary used to contain arrays of images (np ndarrays)
    # The first element will be the label
    # MODEL: { "Normal":["Normal", Normal_1, Normal_2], "ADSD":[ADSD_1,...], etc.. }
    
    
    box_folder_path = "/Users/Will/Box Sync/"
    spec_dirs = []
    for d in listdir(box_folder_path):
        if d[-5:] == '_spec' and d != "WAV files full exams_spec":
            spec_dirs.append(d)
            
    for spec_dir in spec_dirs:
        samples_path = join(box_folder_path, spec_dir)
        samples = listdir(samples_path)
        for sample in samples:
            if sample != ".DS_Store":
                label = getLabel(sample)
                if label not in data.keys():
                    data[label] = [[], []]
                # Lists of spectrogram png files
                keys = list(data.keys())
                label_category = keys.index(label)//2 + 1
                for imfile in listdir(join(samples_path,sample)):
                    imfile_path = join(join(samples_path,sample),imfile)
                    data[label][0].append(imread(imfile_path))
                    data[label][1].append(label_category)
                    
    return data


# Gets all samples to be tested
def getAllSamples():
    images = []
    labels = []
    
    data = populateData()
        

    for key in data.keys():
        for image in key[0]:
                images.append(image)
        for label in key[1]:
                labels.append(label)
                
    images, labels = shuffle(images, labels)
    
    return (images,labels)

