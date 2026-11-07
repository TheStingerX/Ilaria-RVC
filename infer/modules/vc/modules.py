import os
import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from ilaria_pitch import install_pipeline
install_pipeline(Pipeline)
from infer.modules.vc.utils import get_index_path_from_model


def normalize_protect(value, default=0.33):
    """Gradio updates must never reach the numeric inference controls."""
    import math
    for _ in range(8):
        if not isinstance(value, dict):
            break
        value = value.get("value", default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return min(0.5, max(0.0, value)) if math.isfinite(value) else default


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config
        self._voice_file = None

    def get_vc(self, sid, *to_return_protect, find_index=True, reuse_loaded=False):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": normalize_protect(to_return_protect[0]) if to_return_protect else 0.33,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": normalize_protect(to_return_protect[1]) if len(to_return_protect) > 1 else 0.33,
            "__type__": "update",
        }

        if sid == "" or sid == []:
            had_voice = self.net_g is not None or self.hubert_model is not None
            self._voice_file = None
            self.net_g = self.hubert_model = self.pipeline = self.cpt = None
            self.tgt_sr = self.n_spk = self.if_f0 = self.version = None
            if had_voice and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0["value"],
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1["value"],
                    "__type__": "update",
                },
                "",
                "",
            )
        from ilaria_model_paths import voice_path

        person = str(voice_path(os.environ["weight_root"], sid))
        stat = os.stat(person)
        signature = (person, stat.st_mtime_ns, stat.st_size)
        if not reuse_loaded or signature != self._voice_file:
            self._voice_file = None
            self._load_voice(person)
            self._voice_file = signature
        for control in (to_return_protect0, to_return_protect1):
            control["visible"] = self.if_f0 != 0
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid) if find_index else "", "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def _load_voice(self, person):
        logger.info(f"Loading: {person}")

        previous_embedder = (self.cpt.get("embedder_model") or "contentvec") if getattr(self, "cpt", None) else "contentvec"
        self.cpt = torch.load(person, map_location="cpu")
        if previous_embedder != (self.cpt.get("embedder_model") or "contentvec"):
            self.hubert_model = None
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        from ilaria_vocoder import apply_vocoder, checkpoint_vocoder, weights_for_runtime
        selected_vocoder = checkpoint_vocoder(self.cpt)
        self.net_g = apply_vocoder(self.net_g, selected_vocoder, self.version, self.if_f0, self.tgt_sr)
        del self.net_g.enc_q

        self.net_g.load_state_dict(weights_for_runtime(self.cpt["weight"]), strict=selected_vocoder == "RefineGAN")
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)

    def ensure_embedder_loaded(self):
        if self.hubert_model is None:
            from ilaria_embedders import load_inference_embedder

            embedder = self.cpt.get("embedder_model") or "contentvec"
            self.hubert_model = load_inference_embedder(self.config, embedder)

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        try:
            protect = normalize_protect(protect)
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            self.ensure_embedder_loaded()

            file_index = (
                (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
                if file_index != ""
                else file_index2
            )  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            if float(index_rate) <= 0:
                index_info = "Voice index: off (Index Ratio is 0)."
            elif file_index and os.path.isfile(file_index):
                index_info = "Voice index setting: enabled at %.0f%%.\nIndex file: %s" % (float(index_rate) * 100, file_index)
            else:
                index_info = "Voice index: off (no index file found)."
            pitch_info = ("Pitch detection (%s): %.2f seconds" % (f0_method, times[1])
                          if self.if_f0 else "Pitch detection: not required by this model.")
            message = (
                "Success! Audio conversion completed.\n%s\n"
                "Audio analysis (reading voice features): %.2f seconds\n%s\n"
                "Generating the converted voice: %.2f seconds\n"
                "Total model processing time: %.2f seconds"
            ) % (index_info, times[0], pitch_info, times[2], sum(times))
            return message, (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, None

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
        return_audio=False,
    ):
        latest_audio = None
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    latest_audio = opt
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("File: %s\n%s" % (os.path.basename(path), info))
                yield ("\n\n".join(infos), latest_audio) if return_audio else "\n\n".join(infos)
            yield ("\n\n".join(infos), latest_audio) if return_audio else "\n\n".join(infos)
        except:
            error = traceback.format_exc()
            yield (error, latest_audio) if return_audio else error
