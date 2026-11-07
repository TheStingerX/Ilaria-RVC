"""Additional RVC pitch extractors adapted from redpanda343/Redpanda-vc.
MIT License

Copyright (c) 2026 AI Hispano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SwiftF0 license:
MIT License

Copyright (c) 2025 Lars Nieradzik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'ilaria'), str(ROOT)]
import hashlib
import io
import math
import os
import threading
import zipfile
from urllib.request import urlopen
import numpy as np
import torch
_LOCK = threading.RLock()
_CACHE = {}
SWIFT_URL = 'https://files.pythonhosted.org/packages/eb/65/029ed8e87f77f5c1b171f6387e9c99509f1385c9d9a3c50b844c45bd26b4/swift_f0-0.1.2-py3-none-any.whl'
SWIFT_HASH = '212715116025a490be70db0afda8fb27b1eadf267a3b18ed8df4866c1574e717'
BEATRICE_URL = 'https://huggingface.co/fierce-cats/beatrice-trainer/resolve/f34836de014b86956096878aecb8d3b17feaaa0b/assets/pretrained/104_3_checkpoint_00300000.pt'
BEATRICE_HASH = '174e5411009e0e4f6ee8a8c97c4cd2f646791eae1b9aa2b425acb797e0353ef4'

def _asset(name):
    folder = ROOT/'assets'/'pitch'; folder.mkdir(parents=True,exist_ok=True)
    path = folder/('swift_ort115.onnx' if name=='swift' else 'beatrice.pt')
    if path.is_file(): return path
    url, expected = (SWIFT_URL,SWIFT_HASH) if name=='swift' else (BEATRICE_URL,BEATRICE_HASH)
    print('Downloading %s pitch extractor...' % name,flush=True)
    with urlopen(url,timeout=60) as response: data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Pitch model download checksum mismatch.')
    if name=='swift':
        import onnx
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            model=onnx.load_model_from_string(archive.read('swift_f0/model.onnx'))
        # This graph only uses operators whose signatures are valid at opset17.
        # Older bundled ORT has no Reshape19 kernel. Validate before saving.
        model.opset_import[0].version=17
        onnx.checker.check_model(model)
        data=model.SerializeToString()
    temporary=path.with_suffix('.partial')
    temporary.write_bytes(data);os.replace(str(temporary),str(path));return path

from ilaria_beatrice import (BINS_PER_OCTAVE, IN_HOP, IN_SAMPLE_RATE, PITCH_BAND,
    PITCH_BASE_HZ, PITCH_WINDOW, PitchEstimator, Weights, checkpoint_pitch_values)
def ensure_pitch_estimator(): return _asset('beatrice')
class BeatricePitch:
    padding = PITCH_WINDOW // 2
    latency = PITCH_WINDOW + IN_HOP - padding
    lookahead_frames = math.ceil((latency - IN_HOP) / IN_HOP)
    chunk_samples = 10 * IN_SAMPLE_RATE

    def __init__(self, device, sample_rate=16000, hop_size=160, threshold=0.7):
        if sample_rate != IN_SAMPLE_RATE or hop_size != IN_HOP:
            raise ValueError("The Beatrice pitch estimator requires 16 kHz audio with a 160-sample hop.")
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.threshold = threshold
        path = ensure_pitch_estimator()
        self.model = PitchEstimator(Weights(path, self.device, checkpoint_pitch_values(path)), self.device)
        self.offsets = torch.arange(1, PITCH_BAND + 1, device=self.device)
        self.reset()

    def reset(self):
        self.model.reset()
        self.audio = torch.zeros(self.padding, device=self.device)
        self.samples = 0
        self.frames = 0

    def _decode(self, probabilities):
        band = probabilities[:, 1:].unfold(1, PITCH_BAND, 1)
        confidence, top = band.sum(-1).max(1)
        weights = band[torch.arange(band.shape[0], device=self.device), top]
        position = (weights * (top[:, None] + self.offsets)).sum(1) / weights.sum(1)
        f0 = PITCH_BASE_HZ * 2.0 ** (position / BINS_PER_OCTAVE)
        return torch.where(confidence > self.threshold, f0, 0.0)

    @torch.inference_mode()
    def push(self, audio):
        audio = torch.as_tensor(audio, dtype=torch.float32, device=self.device).reshape(-1)
        self.audio = torch.cat([self.audio, audio])
        self.samples += audio.shape[0]
        frames = (self.audio.shape[0] - PITCH_WINDOW) // IN_HOP
        if frames <= 0:
            return torch.zeros(0, device=self.device)
        instfreq, difference, _ = self.model.inputs(self.audio, frames)
        probabilities = self.model.probabilities(instfreq, difference)
        self.audio = self.audio[frames * IN_HOP:]
        self.frames += frames
        return self._decode(probabilities)

    def get_f0(self, x, p_len=None):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if p_len is None:
            p_len = x.shape[0] // self.hop_size
        self.reset()
        x = np.pad(x, (0, self.latency))
        f0 = torch.cat([self.push(x[start:start + self.chunk_samples])
                        for start in range(0, x.shape[0], self.chunk_samples)])
        f0 = f0[:p_len].cpu().numpy()
        return np.pad(f0, (0, p_len - f0.shape[0]))

def release_fcpe():
    with _LOCK:
        for key in list(_CACHE):
            if key[0] == "fcpe":
                del _CACHE[key]


def predict_pitch(audio, method, p_len=None, device='cpu'):
    audio=np.asarray(audio,dtype=np.float32).reshape(-1)
    p_len=len(audio)//160 if p_len is None else int(p_len)
    if p_len<=0:return np.zeros(0,dtype=np.float64)
    with _LOCK:
        if method == "fcpe":
            from ilaria_fcpe import FCPEPitch, fcpe_device

            key = (method, fcpe_device(device))
            if key not in _CACHE:
                _CACHE[key] = FCPEPitch(device)
            return _CACHE[key].get_f0(audio, p_len)
        if method=='beatrice':
            key=(method,str(device))
            if key not in _CACHE:_CACHE[key]=BeatricePitch(device)
            return _CACHE[key].get_f0(audio,p_len).astype(np.float64)
        if method!='swift':raise ValueError('Unknown pitch extraction algorithm.')
        key=('swift','cpu')
        if key not in _CACHE:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=2
            _CACHE[key]=ort.InferenceSession(str(_asset('swift')),sess_options=options,providers=['CPUExecutionProvider'])
        pitch,confidence=_CACHE[key].run(None,{'input_audio':np.pad(audio,(0,max(0,256-len(audio))))[None]})
        pitch,confidence=pitch[0],confidence[0]
        times=(np.arange(len(pitch))*256+127.5)/16000
        target=np.arange(p_len)*.01
        voiced=(confidence>=.5)&(pitch>=50)&(pitch<=1100)
        if not np.any(voiced):return np.zeros(p_len,dtype=np.float64)
        f0=np.exp2(np.interp(target,times[voiced],np.log2(pitch[voiced])))
        f0[np.interp(target,times,confidence)<.5]=0
        return f0


def install_pipeline(pipeline_class):
    import inspect,textwrap
    if getattr(pipeline_class,'_ilaria_pitch_installed',False):return
    original=pipeline_class.get_f0
    source=textwrap.dedent(inspect.getsource(original))
    source=source.replace('    global input_audio_path2wav','    f0_method = "pm" if f0_method == "parselmouth" else f0_method\n    global input_audio_path2wav',1)
    source=source.replace('    if f0_method == "pm":','    if f0_method in ("swift", "beatrice", "fcpe"):\n        f0 = ilaria_predict_pitch(x, f0_method, p_len, self.device)\n    elif f0_method == "pm":',1)
    namespace={};original.__globals__['ilaria_predict_pitch']=predict_pitch
    exec(compile(source,'<ilaria pitch integration>','exec'),original.__globals__,namespace)
    pipeline_class.get_f0=namespace['get_f0'];pipeline_class._ilaria_pitch_installed=True


def extract_dataset(folder,method):
    import soundfile as sf
    folder=Path(folder);files=sorted((folder/'1_16k_wavs').glob('*.wav'))
    if not files:raise ValueError('No processed audio found. Run Process Data first.')
    for name in ['2a_f0','2b-f0nsf']:(folder/name).mkdir(exist_ok=True)
    with (folder/'extract_f0_feature.log').open('a',encoding='utf-8') as log:
        for number,path in enumerate(files):
            message='f0ing,now-%s,all-%s,-%s' % (number,len(files),path)
            print(message,flush=True);log.write(message+'\n');log.flush()
            audio,rate=sf.read(str(path),dtype='float32')
            if rate!=16000:raise ValueError('Pitch extraction requires preprocessed 16 kHz audio.')
            if audio.ndim>1:audio=audio.mean(1)
            f0=predict_pitch(audio,method,device='cpu')
            mel=1127*np.log(1+f0/700);lo=1127*np.log(1+50/700);hi=1127*np.log(1+1100/700)
            positive=mel>0;mel[positive]=(mel[positive]-lo)*254/(hi-lo)+1
            coarse=np.rint(np.clip(mel,1,255)).astype(np.int32)
            np.save(str(folder/'2a_f0'/path.name)+'.npy',coarse,allow_pickle=False)
            np.save(str(folder/'2b-f0nsf'/path.name)+'.npy',f0,allow_pickle=False)
        log.write('Pitch extraction done!\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--extract',required=True);parser.add_argument('method',choices=['swift','beatrice','fcpe'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    try:
        extract_dataset(args.extract,args.method)
    except Exception:
        import traceback
        with (Path(args.extract)/"extract_f0_feature.log").open("a",encoding="utf-8") as log:
            log.write(traceback.format_exc())
        raise
