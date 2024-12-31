#!/usr/bin/env python
# coding: utf-8

# ## Pre processing of midi data

# In[1]:


# from music21 import midi
# mf = midi.MidiFile()
# mf.open(midis[0]) 
# mf.read()
# mf.close()
# s = midi.translate.midiFileToStream(mf)
# s.show('midi')


# In[24]:


from music21 import converter, instrument, note, chord
import json
import sys
import numpy as np
from imageio import imwrite
from PIL import Image

def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))
                
        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element.notes:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

def midi2image(midi_path,image_path):
    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = get_notes(notes_to_parse)
                i+=1
            else:
                data[instrument_i.partName] = get_notes(notes_to_parse)

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0".format(i)] = get_notes(notes_to_parse)

    resolution = 0.25

    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
        upperBoundNote = 127
        lowerBoundNote = 21
        maxSongLength = 100

        index = 0
        prev_index = 0
        repetitions = 0
        while repetitions < 1:
            if prev_index >= len(values["pitch"]):
                break

            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))

            pitchs = values["pitch"]
            durs = values["dur"]
            starts = values["start"]

            for i in range(prev_index,len(pitchs)):
                pitch = pitchs[i]

                dur = int(durs[i]/resolution)
                start = int(starts[i]/resolution)

                if dur+start - index*maxSongLength < maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0:
                            matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255
                else:
                    prev_index = i
                    break
            img_path = image_path + "\\" + midi_path.split("\\")[-1].replace(".mid",f"_{instrument_name}_{index}.png")
            imwrite(img_path,matrix)
            index += 1
            repetitions+=1
            


# In[ ]:


import os
import numpy as np
from PIL import Image
#import py_midicsv as pm
path = r'C:\Users\Monalisha\Downloads\Jazz-Midi\Jazz Midi'
os.chdir(path)
midiz = os.listdir()
midis = []
for midi in midiz:
    midis.append(path+'\\'+midi)
    
new_dir = r'C:\Users\Monalisha\Desktop\generative modelling\project\image_data'
count = 0
for midi_path in midis:
    try:
        midi2image(midi_path,new_dir)
        count += 1
        print(count)
    except:
        pass


# In[1]:


import os
from PIL import Image
from matplotlib import pyplot as plt 
import numpy as np
path = r'C:\Users\Monalisha\Desktop\generative modelling\project'
os.getcwd()
img_list = os.listdir(path)


# In[19]:


pixels = []
for i in range(len(img_list)):
    if "png" in img_list[i]:
        img = Image.open(path+'/'+img_list[i],'r')
        img = img.resize((106,106), Image.ANTIALIAS)
        pix = np.array(img.getdata())
        if np.mean(img) != 0:    
            pix = pix.astype('float32')
            pix /= 255.0
            pixels.append(pix.reshape(106,106,1))
pixels = np.array(pixels)


# In[23]:


with open("numpy_data.npy","wb") as f:
    np.save(f,pixels)


# In[24]:


with open("numpy_data.npy","rb") as f:
    a = np.load(f)


# In[27]:





# In[ ]:




