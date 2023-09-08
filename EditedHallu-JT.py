import sys
import os
import time
import re
from _io import StringIO
 
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

useCompressor=True


beam_size=2
model = None
device = "cpu" #cuda / cpu
cudaIdx = 0

SAMPLING_RATE = 16000

from threading import Lock, Thread
lock = Lock()

def getPrompt(lng:str):

    return "The transcript is between a Veterinarian and client containing veterinary medical terminology"
    

def transcribePrompt(path: str,lng: str,prompt=None,lngInput=None,isMusic=False,addSRT=False):
    """Whisper transcribe."""

    if(lngInput == None):
        lngInput=lng
        print("Using output language as input language: "+lngInput)
    
    if(prompt == None):
        if(not isMusic):
            prompt=getPrompt(lng)
        else:
            prompt="";
    
    print("=====transcribePrompt",flush=True)
    print("PATH="+path,flush=True)
    print("LNGINPUT="+lngInput,flush=True)
    print("LNG="+lng,flush=True)
    print("PROMPT="+prompt,flush=True)
    opts = dict(language=lng,initial_prompt=prompt)
    return transcribeOpts(path, opts,lngInput,isMusic=isMusic,addSRT=addSRT)


def transcribeOpts(path: str,opts: dict,lngInput=None,isMusic=False,addSRT=False):
    pathIn = path
    pathClean = path
    pathNoCut = path
    
    initTime = time.time()
    
    startTime = time.time()
    try:
        #Convert to WAV to avoid later possible decoding problem
        pathWAV = pathIn+".WAV"+".wav"
        aCmd = "ffmpeg -y -i \""+pathIn+"\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathWAV+"\" > \""+pathWAV+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathWAV,flush=True)
        pathIn = pathClean = pathWAV
    except:
         print("Warning: can't convert to WAV")
    
    if(useDemucs):
        startTime = time.time()
        #try:
        #demucsDir=pathIn+".demucs"
        #if(not os.path.exists(demucsDir)):
        #    os.mkdir(demucsDir)
        pathDemucs=pathIn+".vocals.wav" #demucsDir+"/htdemucs/"+os.path.splitext(os.path.basename(pathIn))[0]+"/vocals.wav"
        #Demucs seems complex, using CLI cmd for now
        #aCmd = "python -m demucs --two-stems=vocals -d "+device+":"+cudaIdx+" --out "+demucsDir+" "+pathIn
        #print("CMD: "+aCmd)
        #os.system(aCmd)
        demucs_audio(pathIn=pathIn,model=modelDemucs,device="cuda:"+cudaIdx,pathVocals=pathDemucs,pathOther=pathIn+".other.wav")
        print("T=",(time.time()-startTime))
        print("PATH="+pathDemucs,flush=True)
        pathNoCut = pathIn = pathDemucs
        #except:
        #     print("Warning: can't split vocals")

    duration = -1
    startTime = time.time()
    try:
        pathSILCUT = pathIn+".SILCUT"+".wav"
        aCmd = "ffmpeg -y -i \""+pathIn+"\" -af \"silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.2:stop_silence=0.2, loudnorm\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathSILCUT+"\" > \""+pathSILCUT+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathSILCUT,flush=True)
        pathIn = pathSILCUT
        duration = getDuration(pathSILCUT+".log")
        print("DURATION="+str(duration))
    except:
         print("Warning: can't filter blanks")

    if(not isMusic and useSileroVAD):
        startTime = time.time()
        try:
            pathVAD = pathIn+".VAD.wav"
            wav = read_audio(pathIn, sampling_rate=SAMPLING_RATE)
            #https://github.com/snakers4/silero-vad/blob/master/utils_vad.py#L161
            speech_timestamps = get_speech_timestamps(wav, modelVAD,threshold=0.5,min_silence_duration_ms=500, sampling_rate=SAMPLING_RATE)
            save_audio(pathVAD,collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
            print("T=",(time.time()-startTime))
            print("PATH="+pathVAD,flush=True)
            pathIn = pathVAD
        except:
             print("Warning: can't filter noises")

    mode=1
    if(duration > 30):
        print("NOT USING MARKS FOR DURATION > 30s")
        mode=0
    
    startTime = time.time()
    result = transcribeMARK(pathIn, opts, mode=mode,lngInput=lngInput,isMusic=isMusic)
    
    if len(result["text"]) <= 0:
        result["text"] = "--"
    
    if(addSRT):
        #Better timestamps using original music clip
        if(isMusic):
            resultSRT = transcribeMARK(pathClean, opts, mode=3,lngInput=lngInput,isMusic=isMusic)
        else:
            resultSRT = transcribeMARK(pathNoCut, opts, mode=3,lngInput=lngInput,isMusic=isMusic)
        
        result["text"] += resultSRT["text"]
    
    print("T=",(time.time()-initTime))
    print("s/c=",(time.time()-initTime)/len(result["text"]))
    print("c/s=",len(result["text"])/(time.time()-initTime))
    
    return result["text"]

def transcribeMARK(path: str,opts: dict,mode = 1,lngInput=None,aLast=None,isMusic=False):
    pathIn = path
    
    lng = opts["language"]
    
    if(lngInput == None):
        lngInput = lng
        
    noMarkRE = "^(ar|he|ru|zh)$"
    if(lng != None and re.match(noMarkRE,lng)):
        #Need special voice marks
        mode = 0

    if(isMusic and mode != 3):
        #Markers are not really interesting with music
        mode = 0
        
    if os.path.exists("markers/WOK-MRK-"+lngInput+".wav"):
        mark1="markers/WOK-MRK-"+lngInput+".wav"
    else:
        mark1="markers/WOK-MRK.wav"
    if os.path.exists("markers/OKW-MRK-"+lngInput+".wav"):
        mark2="markers/OKW-MRK-"+lngInput+".wav"
    else:
        mark2="markers/OKW-MRK.wav"
    
    if(mode == 2):
        mark = mark1
        mark1 = mark2
        mark2 = mark
        
    if(mode == 0):
        print("["+str(mode)+"] PATH="+pathIn,flush=True)
    else:
        try:
            if(mode != 3):
                startTime = time.time()
                pathMRK = pathIn+".MRK"+".wav"
                aCmd = "ffmpeg -y -i "+mark1+" -i \""+pathIn+"\" -i "+mark2+" -filter_complex \"[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]\" -map \"[a]\" -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathMRK+"\" > \""+pathMRK+".log\" 2>&1"
                print("CMD: "+aCmd)
                os.system(aCmd)
                print("T=",(time.time()-startTime))
                print("["+str(mode)+"] PATH="+pathMRK,flush=True)
                pathIn = pathMRK
            
            if(useCompressor and not isMusic):
                startTime = time.time()
                pathCPS = pathIn+".CPS"+".wav"
                aCmd = "ffmpeg -y -i \""+pathIn+"\" -af \"speechnorm=e=50:r=0.0005:l=1\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathCPS+"\" > \""+pathCPS+".log\" 2>&1"
                print("CMD: "+aCmd)
                os.system(aCmd)
                print("T=",(time.time()-startTime))
                print("["+str(mode)+"] PATH="+pathCPS,flush=True)
                pathIn = pathCPS
        except:
             print("Warning: can't add markers")
    
    startTime = time.time()
    lock.acquire()
    try:
        transcribe_options = dict(**opts)#avoid to add beam_size opt several times
        if beam_size > 1:
            transcribe_options = dict(beam_size=beam_size,**opts)
        
        import openai
        openai.api_key = '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'
        audio_file= open(pathIn, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(transcript)

        print("T=",(time.time()-startTime))
        print("TRANS="+result["text"],flush=True)
    except Exception as e: 
        print(e)
        traceback.print_exc()
        lock.release()
        result = {}
        result["text"] = ""
        return result
    
    lock.release()
    
    if(mode == 0 or mode == 3):
        return result
        #Too restrictive
        #if(result["text"] == aLast):
        #    #Only if confirmed
        #    return result
        #result["text"] = ""
        #return result
    
    aWhisper="(Whisper|Wisper|Wyspę|Wysper|Wispa|Уіспер|Ου ίσπερ|위스퍼드|ウィスパー|विस्पर|विसपर)"
    aOk="(o[.]?k[.]?|okay|oké|okej|Окей|οκέι|오케이|オーケー|ओके)"
    aSep="[.,!? ]*"
    if(mode == 1):
        aCleaned = re.sub(r"(^ *"+aWhisper+aSep+aOk+aSep+"|"+aOk+aSep+aWhisper+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(re.match(r"^ *("+aOk+"|"+aSep+"|"+aWhisper+")*"+aWhisper+"("+aOk+"|"+aSep+"|"+aWhisper+")* *$", result["text"], re.IGNORECASE)):
            #Empty sound ?
            return transcribeMARK(path, opts, mode=2,lngInput=lngInput,aLast="")
        
        if(re.match(r"^ *"+aWhisper+aSep+aOk+aSep+".*"+aOk+aSep+aWhisper+aSep+" *$", result["text"], re.IGNORECASE)):
            #GOOD!
            result["text"] = aCleaned
            return result
        
        return transcribeMARK(path, opts, mode=2,lngInput=lngInput,aLast=aCleaned)
    
    if(mode == 2):
        aCleaned = re.sub(r"(^ *"+aOk+aSep+aWhisper+aSep+"|"+aWhisper+aSep+aOk+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(aCleaned == aLast):
            #CONFIRMED!
            result["text"] = aCleaned
            return result
            
        if(re.match(r"^ *("+aOk+"|"+aSep+"|"+aWhisper+")*"+aWhisper+"("+aOk+"|"+aSep+"|"+aWhisper+")* *$", result["text"], re.IGNORECASE)):
            #Empty sound ? 
            result["text"] = ""
            return result
        
        if(re.match(r"^ *"+aOk+aSep+aWhisper+aSep+".*"+aWhisper+aSep+aOk+aSep+" *$", result["text"], re.IGNORECASE)):
            #GOOD!
            result["text"] = aCleaned
            return result
        
        return transcribeMARK(path, opts, mode=0,lngInput=lngInput,aLast=aCleaned)


lng="en"
prompt= "The transcript is a conversation between a veterinarian and client containing Veterinarian terminology"
path = "data/Dobby_raw.mp3"
transcribePrompt(path=path, lng=lng, prompt=prompt)