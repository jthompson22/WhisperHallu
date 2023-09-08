import torch
import torchaudio
import demucs
from demucs.pretrained import get_model_from_args
from demucs.apply import apply_model
from demucs.separate import load_track
from torch._C import device

def load_demucs_model():
    return get_model_from_args(type('args', (object,), dict(name='htdemucs', repo=None))).cpu().eval()


def demucs_audio(pathIn: str,
                 model=None,
                 device=None,
                 pathVocals: str = None,
                 pathOther: str = None):
    if model is None:
        model = load_demucs_model()

    audio = load_track(pathIn, model.audio_channels, model.samplerate)

    audio_dims = audio.dim()
    if audio_dims == 1:
        audio = audio[None, None].repeat_interleave(2, -2)
    else:
        if audio.shape[-2] == 1:
            audio = audio.repeat_interleave(2, -2)
        if audio_dims < 3:
            audio = audio[None]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Demucs using device: "+device)
    result = apply_model(model, audio, device=device, split=True, overlap=.25)
    if device != 'cpu':
        torch.cuda.empty_cache()
    
    for name in model.sources:
        print("Source: "+name)
        source_idx=model.sources.index(name)
        source=result[0, source_idx].mean(0)
        if name == "vocals":
            torchaudio.save(pathVocals, source[None], model.samplerate)
        #torchaudio.save(pathIn+"."+name+".wav", source[None], model.samplerate)
        

# load_demucs_model()
# This function seems to be responsible for loading the htdemucs model.
# type('args', (object,), dict(name='htdemucs', repo=None)) is a way to dynamically create a new type or class. Here, it creates a new class named args with object as its base class and gives it one property named name with a value of 'htdemucs' and another named repo with a value of None.
# This class is then passed to get_model_from_args(), which likely initializes and returns the Demucs model based on the provided arguments.
# The model is moved to the CPU (.cpu()) and set to evaluation mode (.eval()), ensuring that operations like dropout and batch normalization behave consistently during inference.
# demucs_audio(...)
# This function seems to process and separate sources from an audio file using the Demucs model:

# If no model is passed as an argument, it loads the default model using load_demucs_model().
# load_track() likely loads the audio track from the file specified by pathIn. It uses the model's expected audio channels and sample rate for this.
# The following block of code adjusts the dimensions of the loaded audio tensor to ensure compatibility with the model:
# If the audio is mono, it's duplicated to make it stereo.
# If the audio lacks a batch dimension or channel dimension, those are added.
# The audio is then passed through the model using apply_model(). The split=True and overlap=.25 arguments might be related to how the audio is split and processed in chunks, especially for long audio tracks.
# If the processing was done on a device other than the CPU (i.e., a GPU), it clears the GPU cache to free up memory.
# The results are then saved as separate audio files for each source. For example, if the model separates vocals and instruments, there might be a file for vocals and another for instruments. The naming convention is the original pathIn with the source name added, e.g., input_filename.vocals.wav.
# After you call the demucs_audio(...) function with the appropriate path, it should process the specified audio file, separate its sources (like vocals and instruments), and save them as separate files.