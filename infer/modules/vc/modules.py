import os
from pathlib import Path
import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch

from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from ilaria_inference_pipeline import Pipeline
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
        self._loaded_embedder = None

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
            self._loaded_embedder = None
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
        last_speaker = max(0, self.cpt["config"][-3] - 1)
        speaker = {"visible": True, "maximum": last_speaker, "value": 0, "__type__": "update"}
        index = {"value": get_index_path_from_model(sid) if find_index else "", "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                speaker,
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else speaker
        )

    def _load_voice(self, person):
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
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

    def ensure_embedder_loaded(self, selected=None):
        embedder = selected or self.cpt.get("embedder_model") or "contentvec"
        if not isinstance(embedder, str):
            raise ValueError("Choose an Embedder Model in Inference.")
        if self.hubert_model is None or self._loaded_embedder != embedder:
            from ilaria_embedders import load_inference_embedder

            self.hubert_model = self._loaded_embedder = None
            self.hubert_model = load_inference_embedder(self.config, embedder)
            self._loaded_embedder = embedder
        return embedder

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
        inference_options=None,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        if self.net_g is None or self.cpt is None:
            return "Select a Voice Model before converting.", None
        try:
            speaker = int(sid)
            if float(sid) != speaker or not 0 <= speaker < self.cpt["config"][-3]:
                return "Choose a Speaker ID supported by this model.", None
        except (ValueError, TypeError, OverflowError):
            return "Choose a valid Speaker ID.", None
        options = inference_options or {}
        try:
            f0_up_key = int(f0_up_key)
            protect = normalize_protect(protect)
            audio = load_audio(input_audio_path, 16000)
            if options.get("formant_ratio") is not None:
                from ilaria_inference_effects import apply_formant_shift

                audio = apply_formant_shift(audio, 16000, options["formant_ratio"])
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            try:
                embedder = self.ensure_embedder_loaded(options.get("embedder_model"))
            except (ValueError, OSError, RuntimeError) as error:
                return f"Embedder could not be loaded. {error}", None
            if self.version == "v1" and not getattr(self.hubert_model, "has_final_proj", True):
                return (f"{embedder} needs an RVC v2 voice model. "
                        "Choose a v2 model or switch Embedder Model to ContentVec."), None

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
                speaker,
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
                options,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            if options.get("clean_audio") or options.get("post_process"):
                from ilaria_inference_effects import apply_output_effects

                audio_opt = apply_output_effects(audio_opt, tgt_sr, options)
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
            if not self.if_f0 and any(options.get(name) is not None for name in
                                      ("autotune_strength", "proposed_pitch_hz")):
                message += "\nAutotune and Proposed Pitch require an F0-guided model."
            if options.get("embedder_model"):
                message += f"\nEmbedder: {embedder}"
                if embedder != (self.cpt.get("embedder_model") or "contentvec") and float(index_rate) > 0:
                    message += "\nUse an index built with this embedder for matching features."
            return message, (tgt_sr, audio_opt)
        except ValueError as error:
            return f"Conversion failed. {error}", None
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
        inference_options=None,
        return_export=False,
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
                    inference_options=inference_options,
                )
                if "Success" in info and opt is not None:
                    if not return_export:
                        latest_audio = opt
                    try:
                        from ilaria_inference_export import EXPORT_FORMATS, write_inference_export

                        export_format = str(format1).lower()
                        if export_format not in EXPORT_FORMATS:
                            raise ValueError("Choose WAV, MP3, FLAC, OGG or M4A.")
                        destination = Path(opt_root) / (Path(path).name + "." + export_format)
                        tgt_sr, audio_opt = opt
                        write_inference_export(destination, audio_opt, tgt_sr)
                        if return_export:
                            latest_audio = destination
                    except (ValueError, OSError, RuntimeError) as error:
                        info += "\nExport failed. " + str(error)
                infos.append("File: %s\n%s" % (os.path.basename(path), info))
                yield ("\n\n".join(infos), latest_audio) if return_audio else "\n\n".join(infos)
            yield ("\n\n".join(infos), latest_audio) if return_audio else "\n\n".join(infos)
        except:
            error = traceback.format_exc()
            yield (error, latest_audio) if return_audio else error
