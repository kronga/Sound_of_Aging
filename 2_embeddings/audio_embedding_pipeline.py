# ---------------------------------------------------------------------------
# audio_embedding_pipeline.py  –  unified, batch‑friendly embedding extractor
# ---------------------------------------------------------------------------
# Key design points
#   • BaseEmbedder subclasses – identical public interface (embed_batch).
#   • Thread‑pool I/O → GPU batches → async write.  Minimal idle time.
#   • Mixed precision on GPU by default.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import os
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file
from tqdm import tqdm

# ↓ heavy deps – import lazily inside embedders to avoid startup cost
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
    WavLMForAudioFrameClassification,
)
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.diarization import Speech_Emotion_Diarization
from speechbrain.inference.interfaces import foreign_class
# from pyannote.audio import Model as PyannoteModel

###########################################################################
#                           helpers & utilities                           #
###########################################################################

TARGET_SR = 16_000
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_sr16k(wave: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return wave
    return torchaudio.functional.resample(wave, sr, TARGET_SR)


def _load_wave(path: str) -> torch.Tensor:
    """Load an audio file as a 1-D float32 tensor @16 kHz."""
    wav, sr = torchaudio.load(path)
    wav = _ensure_sr16k(wav, sr)
    return wav.squeeze(0)  # (T)


def pad_stack(wavs: List[torch.Tensor]) -> torch.Tensor:
    """Zero-pad a list of 1D tensors and stack into (B,T)."""
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, : w.shape[0]] = w
    return batch


###########################################################################
#                               embedders                                 #
###########################################################################

class BaseEmbedder(ABC):
    """All embedders expose the same interface."""

    supports_batch: bool = True  # override for MFCC etc.

    def __init__(self):
        self.device = _DEVICE

    # --------------------------------------------------------------
    def _forward(self, wav_batch: torch.Tensor) -> np.ndarray:
        """Actual model forward – expects (B,T) @16 kHz on self.device."""
        raise NotImplementedError(
            "This embedder does not support batched inference."
        )
    # --------------------------------------------------------------
    def embed_batch(self, wavs: List[torch.Tensor]) -> List[np.ndarray]:
        if not self.supports_batch:
            raise RuntimeError(
                "embed_batch() called on a non-batching embedder"
            )
        if not wavs:
            return []
        batch = pad_stack(wavs).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
            out = self._forward(batch.half() if self.device.type == "cuda" else batch)
        return [o for o in out]


# --------------------------- MFCC embedder --------------------------- #
class MFCCEmbedder(BaseEmbedder):
    supports_batch = False

    def __init__(self, n_mfcc: int = 13, sr: int = TARGET_SR):
        super().__init__()
        import librosa  # local import – heavy

        self.n_mfcc = n_mfcc
        self.sr = sr
        self.librosa = librosa

    def _forward(self, wav_batch: torch.Tensor) -> np.ndarray:  # never used
        raise NotImplementedError

    def embed_file(self, path: str) -> np.ndarray:
        audio, _ = self.librosa.load(path, sr=self.sr)
        audio, _ = self.librosa.effects.trim(audio, top_db=20)
        mfcc = self.librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        return mfcc


# -------------- HF Wav2Vec2 / WavLM family (tokenizer‑free) -------------
class HFTransformerEmbedder(BaseEmbedder):
    """Generic wrapper for Wav2Vec2‑type models.
    Uses *only* the feature‑extractor, avoiding the tokenizer download.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _forward(self, wav_batch: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        # Convert padded tensor (B,T) back to list of 1‑D numpy arrays for the extractor
        if isinstance(wav_batch, torch.Tensor):
            wav_list = [t.cpu().numpy() for t in wav_batch]
        else:  # already list
            wav_list = [t.cpu().numpy() for t in wav_batch]

        inputs = self.processor(
            wav_list,
            sampling_rate=TARGET_SR,
            padding=True,
            return_tensors="pt",
        ).input_values.to(self.device)

        hidden = self.model(inputs, output_hidden_states=True).hidden_states[-1]
        return hidden.cpu().numpy()


class RandomWav2VecEmbedder(HFTransformerEmbedder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # randomise conv & linear layers
        def _reinit(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.model.apply(_reinit)


# -------------------------- WavLM large/base -------------------------
class WavLMEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _forward(self, wav_batch):
        if isinstance(wav_batch, torch.Tensor):
            wav_list = [t.cpu().numpy() for t in wav_batch]
        else:
            wav_list = [t.cpu().numpy() for t in wav_batch]
        inputs = self.processor(wav_list, sampling_rate=TARGET_SR, padding=True, return_tensors="pt").input_values.to(self.device)
        hidden = self.model(inputs, output_hidden_states=True).last_hidden_state
        return hidden.cpu().numpy()


class WavLMSDEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = WavLMForAudioFrameClassification.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _forward(self, wav_batch):
        wav_list = [t.cpu().numpy() for t in (wav_batch if isinstance(wav_batch, torch.Tensor) else wav_batch)]
        inputs = self.processor(wav_list, sampling_rate=TARGET_SR, padding=True, return_tensors="pt").input_values.to(self.device)
        hidden = self.model(**{"input_values": inputs}, output_hidden_states=True).hidden_states[-1]
        return hidden.cpu().numpy()

# --------------------------- VoxCeleb embedder ------------------------
class VoxCelebEmbedder(BaseEmbedder):
    supports_batch = False

    def __init__(self, auth_token: str):
        super().__init__()
        self.model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=auth_token).to(self.device)

    def embed_file(self, path: str) -> np.ndarray:
        wav = _load_wave(path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = torch.nn.functional.normalize(self.model(wav))
        return emb.squeeze(0).cpu().numpy()


# ----------------------------- X‑vector --------------------------------
class XVectorEmbedder(BaseEmbedder):
    supports_batch = False

    def __init__(self, model_name: str):
        super().__init__()
        self.model = EncoderClassifier.from_hparams(source=model_name, savedir="pretrained_models/spkrec-xvect-voxceleb")

    def embed_file(self, path: str) -> np.ndarray:
        sig = _load_wave(path).unsqueeze(0)
        return self.model.encode_batch(sig).cpu().numpy()


# -------------------------- EfficientNet stub ----------------------- #

def _import_effnet():
    parent = Path(__file__).resolve().parents[2]
    sys.path.append(str(parent))
    try:
        from DeepVoice.src.models.models import EffNet  # type: ignore
    except ImportError:
        try:
            from src.models.models import EffNet  # type: ignore
        except ImportError:
            from models import EffNet  # type: ignore
    return EffNet


class EfficientNetEmbedder(BaseEmbedder):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__()
        EffNet = _import_effnet()
        self.model = EffNet(in_channel=1, stride=2, dilation=1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 512)
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device)
            state = {k[7:]: v[1] for k, v in zip(state.keys(), state.items())}
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.supports_batch = False

    def embed_file(self, path: str) -> np.ndarray:
        wav = _load_wave(path)
        with torch.no_grad():
            emb = self.model(wav.unsqueeze(0).unsqueeze(0).to(self.device))
        return torch.nn.functional.normalize(emb).squeeze(0).cpu().numpy()

###########################################################################
#                       registry & factory dispatcher                     #
###########################################################################

EmbedderFactory = Dict[str, functools.partial]  # model_type -> constructor partial

FACTORY: EmbedderFactory = {
    "mfcc": functools.partial(MFCCEmbedder, n_mfcc=13, sr=TARGET_SR),
    "wav2vec": functools.partial(HFTransformerEmbedder, model_name="facebook/wav2vec2-large-xlsr-53"),
    "wav2vec2": functools.partial(HFTransformerEmbedder, model_name="facebook/wav2vec2-large-xlsr-53"),
    "wav2vec2_base": functools.partial(HFTransformerEmbedder, model_name="facebook/wav2vec2-base"),
    "wav2vec2_large": functools.partial(HFTransformerEmbedder, model_name="facebook/wav2vec2-large"),
    "random-wav2vec": functools.partial(RandomWav2VecEmbedder, model_name="facebook/wav2vec2-large-xlsr-53"),
    "wavlm_base": functools.partial(WavLMEmbedder, model_name="microsoft/wavlm-base"),
    "wavlm_large": functools.partial(WavLMEmbedder, model_name="microsoft/wavlm-large"),
    "wavlm_sd": functools.partial(WavLMSDEmbedder, model_name="microsoft/wavlm-base-plus-sd"),
    "xvector": functools.partial(XVectorEmbedder, model_name="speechbrain/spkrec-xvect-voxceleb"),
    "voxceleb": functools.partial(VoxCelebEmbedder, auth_token=os.environ.get("HF_TOKEN", "")),
    # EfficientNet & finetuned variants – pass checkpoint later
    "efficientnet": functools.partial(EfficientNetEmbedder, checkpoint=None),
}


###########################################################################
#                              main driver                                #
###########################################################################


def embed_paths(
    embedder: BaseEmbedder,
    paths: List[str],
    batch_size: int = 8,
    num_io_workers: int = 4,
):
    """Generator yielding (path, embedding). Handles batching & I/O overlap."""

    def _io(path_: str):
        return _load_wave(path_)

    # Thread‑pool for disk→RAM
    with cf.ThreadPoolExecutor(max_workers=num_io_workers) as pool:
        for i in range(0, len(paths), batch_size if embedder.supports_batch else 1):
            chunk = paths[i : i + (batch_size if embedder.supports_batch else 1)]
            wavs = list(pool.map(_io, chunk))
            if embedder.supports_batch:
                embs = embedder.embed_batch(wavs)
            else:
                embs = [embedder.embed_file(p) for p in chunk]
            for p, e in zip(chunk, embs):
                yield p, e


def save_embedding(path: str, emb: np.ndarray, out_dir: Path):
    rel = Path(path).stem
    np.save(out_dir / f"{rel}.npy", emb)



###########################################################################
#                                   CLI                                   #
###########################################################################


def parse_args():
    ap = argparse.ArgumentParser("Unified audio embedding extractor")
    ap.add_argument("input", type=Path, help="Directory with audio files")
    ap.add_argument("output", type=Path, help="Directory to save *.npy embeddings")
    ap.add_argument("--model", default="wav2vec", choices=FACTORY.keys())
    ap.add_argument("--bs", type=int, default=1, help="Batch size")
    ap.add_argument("--workers", type=int, default=8, help="I/O worker threads")
    return ap.parse_args()


###########################################################################
#                                   main                                  #
###########################################################################


def main():
    args = parse_args()
    args.output.mkdir(exist_ok=True, parents=True)

    # Build embedder
    if args.model not in FACTORY:
        raise ValueError(f"Unsupported model_type {args.model}")
    embedder: BaseEmbedder = FACTORY[args.model]()

    # Enumerate work
    audio_paths = [str(p) for p in args.input.glob("*.*") if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
    todo = [p for p in audio_paths if not (args.output / f"{Path(p).stem}.npy").exists()]

    print(f"{len(todo)} / {len(audio_paths)} files need processing…")
    for p, e in tqdm(embed_paths(embedder, todo, args.bs, args.workers), total=len(todo), desc=args.model):
        save_embedding(p, e, args.output)

    print("✔ Done!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

