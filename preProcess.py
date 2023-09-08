import sys
import os
import time
import re
from _io import StringIO
import openai
import subprocess
 
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    print("Python >= 3.10")
    import collections.abc
    import collections
    collections.MutableMapping = collections.abc.MutableMapping
else:
    print("Python < 3.10")
    import collections
    
import traceback
import torch

torch.set_num_threads(1)
useSileroVAD=True
if(useSileroVAD):
    modelVAD, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

useDemucs=True
if(useDemucs):
    from demucsWrapper import load_demucs_model
    from demucsWrapper import demucs_audio
    print("Using Demucs")
    modelDemucs = load_demucs_model()

#useCompressor=True


SAMPLING_RATE = 16000

from threading import Lock, Thread
lock = Lock()


#Helpers 
def remove_wav_extension(path):
    if path.endswith('.wav'):
        return path[:-4]
    return path
def remove_base(path: str) -> str:
    base, ext = os.path.splitext(path)
    return base

    aH = int(aT/3600)
    aM = int((aT%3600)/60)
    aS = (aT%60)
    return "%02d:%02d:%06.3f" % (aH,aM,aS)
def getDuration(aLog:str):
    with open(aLog) as f:
        lines = f.readlines()
        for line in lines:
            if(re.match(r"^ *Duration: [0-9][0-9]:[0-9][0-9]:[0-9][0-9][.][0-9][0-9], .*$", line, re.IGNORECASE)):
                duration = re.sub(r"(^ *Duration: *|[,.].*$)", "", line, 2, re.IGNORECASE)
                return sum(x * int(t) for x, t in zip([3600, 60, 1], duration.split(":")))

import subprocess

def run_command_and_check(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed with error: {stderr.decode()}")


#PROCESSING
def wavPreProcess(path: str) -> str:
    try:
        print("Converting To WAV:")
        initTime = time.time()
        pathWAV = remove_base(path) + "_wav-converted_" + ".wav"
        print("Wave Path: ", pathWAV, flush=True)
        aCmd = f"ffmpeg -y -i \"{path}\" -c:a pcm_s16le -ar {SAMPLING_RATE} \"{pathWAV}\" > \"{pathWAV}.log\" 2>&1"
        print("CMD:", aCmd)
        run_command_and_check(aCmd)
        print("Time=", (time.time() - initTime))
        print("PATH=", pathWAV, flush=True)
        return pathWAV
    except Exception as e:
        print(f"Warning: can't convert to WAV. Error: {e}")
        sys.exit(-1)

def demucsPreProcess(path: str, device: str):
    try:
        from demucsWrapper import load_demucs_model
        from demucsWrapper import demucs_audio
        print("Using Demucs")
        modelDemucs = load_demucs_model()
        startTime = time.time()
        pathDemucs=remove_base(path) +"demucs-vocals_.wav" 
        pathRemoved= "RemovedNoise/"
        if(not os.path.exists("RemovedNoise")):
                os.mkdir("RemovedNoise")
        if(device == "cpu"):
            localDevice = None
        else:
            localDevice = "cuda:0"
        demucs_audio(pathIn=path,model=modelDemucs,device=localDevice,pathVocals=pathDemucs,pathOther=pathRemoved+"other.wav")
        print("T= ",(time.time()-startTime))
        print("PATH= "+pathDemucs,flush=True)
        return pathDemucs
    except:
        print("Warning: can't split vocals")
        sys.exit(-1)

def removeSilencePreProcess(path: str):
    try:
        startTime = time.time()
        pathSILCUT = remove_base(path) +"_slience-loudnorm_"+".wav"
        aCmd = "ffmpeg -y -i \""+path+"\" -af \"silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.2:stop_silence=0.2, loudnorm\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathSILCUT+"\" > \""+pathSILCUT+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T= ",(time.time()-startTime))
        print("PATH= "+pathSILCUT,flush=True)
        duration = getDuration(pathSILCUT+".log")
        print("DURATION="+ str(duration))
        return pathSILCUT
    except:
         print("Unable to remove silences")
         sys.exit(-1)

def sileroVADPreProcess(path: str):
   
    try:
        startTime = time.time()
        pathVAD = remove_base(path) +"_silero_.wav"
        wav = read_audio(path, sampling_rate=SAMPLING_RATE)
        #https://github.com/snakers4/silero-vad/blob/master/utils_vad.py#L161
        speech_timestamps = get_speech_timestamps(wav, modelVAD,threshold=0.5,min_silence_duration_ms=500, sampling_rate=SAMPLING_RATE)
        save_audio(pathVAD,collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
        print("T=",(time.time()-startTime))
        print("PATH="+pathVAD,flush=True)
        return pathVAD
    except:
        print("Warning: can't filter noises")
        sys.exit(-1)

def useCompressor(path):
    try:
        startTime = time.time()
        pathCPS = remove_wav_extension(path)+"_compressed_"+".wav"
        aCmd = "ffmpeg -y -i \""+path+"\" -af \"speechnorm=e=50:r=0.0005:l=1\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathCPS+"\" > \""+pathCPS+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathCPS,flush=True)
        return pathCPS
    except:
        print("Warning: can't compress")
        sys.exit(-1)

def runPreProcessAlgorithim(path: str, device: str, options: dict):
    pathIn = path

    pathIn = wavPreProcess(pathIn)
    pathIn = demucsPreProcess(pathIn, device)
    pathIn = removeSilencePreProcess(pathIn)
    pathIn = sileroVADPreProcess(pathIn)
    print("FINALPATH= "+ pathIn, flush=True)


