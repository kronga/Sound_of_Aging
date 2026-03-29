import torch
import torchaudio
import numpy as np
import os
from tqdm import tqdm
import librosa
from typing import Union, Optional, Tuple, Dict
from safetensors.torch import load_file
import argparse
import sys
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WavLMForCTC,
    WavLMForAudioFrameClassification,
    WavLMModel,
    Wav2Vec2FeatureExtractor
)
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.diarization import Speech_Emotion_Diarization
from speechbrain.inference.interfaces import foreign_class
from pyannote.audio import Model as PyannoteModel

class AudioProcessor:
    def __init__(
            self,
            model_type: str,
            model_name: str,
            processor_name: Optional[str] = None,
            model_checkpoint: Optional[str] = None,
            use_safetensors: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = self._initialize_model(model_type, model_name, model_checkpoint, use_safetensors)
        self.processor = self._initialize_processor(model_type, processor_name or model_name)
        self.use_gpu = torch.cuda.is_available()
        # MFCC parameters
        self.mfcc_params = {
            "n_mfcc": 13,  # Default number of MFCCs
            "sr": 16000    # Default sampling rate
        }
        if model_type == "mfcc" and isinstance(model_name, dict):
            # If model_name is a dict, it contains MFCC parameters
            self.mfcc_params.update(model_name)

    def _initialize_model(self, model_type, model_name, model_checkpoint, use_safetensors):
        if model_type == "mfcc":
            # MFCC doesn't require a model, return None
            return None
            
        if model_checkpoint:
            state_dict = (
                load_file(model_checkpoint)
                if use_safetensors
                else torch.load(model_checkpoint)
            )

        if model_type == "random-wav2vec":
            # Initialize regular wav2vec model first
            model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            # Randomize all weights
            def reinit_weights(m):
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            model.apply(reinit_weights)
            return model
        
        elif model_type in ["wav2vec2_hebrew_ft", "wav2vec2_hebrew_initial"]:
            # For Hebrew models, load the base model and then apply the checkpoint
            print(f"Initializing Hebrew model from checkpoint: {model_checkpoint}")
            if model_checkpoint:
                # Use Wav2Vec2Model for feature extraction (not CTC)
                model = Wav2Vec2Model.from_pretrained(model_name)
                
                # Load state dict from safetensors
                if use_safetensors:
                    print("Loading weights from safetensors file")
                    state_dict = load_file(model_checkpoint)
                else:
                    print("Loading weights from PyTorch checkpoint")
                    state_dict = torch.load(model_checkpoint)
                
                # Filter state dict to only include keys that match the model
                # This handles cases where the checkpoint might have additional keys
                model_state_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
                
                # Check if we have matching keys
                missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} keys are missing from the checkpoint")
                    print(f"First few missing keys: {list(missing_keys)[:5]}")
                
                # Load the filtered state dict
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Successfully loaded {len(filtered_state_dict)} parameters from checkpoint")
                
                return model.to(self.device)
            else:
                print("No checkpoint provided for Hebrew model, using base model")
                return Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        
        elif model_type == "wavlm_large":
            # Import WavLM model
            from transformers import WavLMModel
            print(f"Loading WavLM-large model from {model_name}")
            return WavLMModel.from_pretrained(model_name).to(self.device)
        
        elif model_type == "wavlm_base":
            # Import WavLM model
            from transformers import WavLMModel
            print(f"Loading WavLM-base model from {model_name}")
            return WavLMModel.from_pretrained(model_name).to(self.device)
    
        elif model_type == "wavlm_sd":
            return WavLMForAudioFrameClassification.from_pretrained(model_name).to(self.device)
        elif model_type == "xvector":
            return EncoderClassifier.from_hparams(
                source=model_name,
                savedir="pretrained_models/spkrec-xvect-voxceleb"
            )
        elif model_type == "emotion_wavlm":
            return Speech_Emotion_Diarization.from_hparams(source=model_name)
        elif model_type == "emotion_wav2vec":
            return foreign_class(
                source=model_name,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
        elif model_type == "voxceleb":
            # Load the pyannote VoxCeleb model
            print(f"Loading VoxCeleb model from pyannote/embedding")
            return PyannoteModel.from_pretrained(
                "pyannote/embedding",
                use_auth_token=model_name  # Pass the auth token
            ).to(self.device)
        elif model_type == "voxceleb_finetuned":
            # Load the base VoxCeleb model first
            print(f"Loading finetuned VoxCeleb model")
            model = PyannoteModel.from_pretrained(
                "pyannote/embedding",
                use_auth_token=model_name["auth_token"]  # Pass the auth token
            )
            # Load finetuned weights
            if model_checkpoint:
                print(f"Loading finetuned weights from {model_checkpoint}")
                state = torch.load(model_checkpoint, map_location=self.device)
                # Remove "module." from the beginning of each key and extract the value
                state = {k[7:]: v[1] for k, v in zip(state.keys(), state.items())}
                model.load_state_dict(state)
            return model.to(self.device)
        elif model_type == "efficientnet":
            # Import EfficientNet model
            import sys
            import os
            # Add the parent directory to sys.path to find the models module
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            sys.path.append(parent_dir)
            try:
                # Try different import paths
                try:
                    # Try the full absolute path
                    from DeepVoice.src.models.models import EffNet
                except ImportError:
                    try:
                        # Try with the parent directory in path
                        from src.models.models import EffNet
                    except ImportError:
                        # Try with the models directory directly
                        from models import EffNet
            except ImportError as e:
                print(f"Error importing EffNet: {e}")
                print(f"Current sys.path: {sys.path}")
                print(f"Looking for module in: {parent_dir}")
                raise ImportError(f"Could not import EffNet model: {e}")
                
            print(f"Loading EfficientNet model")
            model = EffNet(in_channel=1, stride=2, dilation=1)
            model.fc = torch.nn.Linear(model.fc.in_features, 512)
            return model.to(self.device)
        elif model_type == "efficientnet_finetuned":
            # Import EfficientNet model
            import sys
            import os
            # Add the parent directory to sys.path to find the models module
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            sys.path.append(parent_dir)
            try:
                # Try different import paths
                try:
                    # Try the full absolute path
                    from DeepVoice.src.models.models import EffNet
                except ImportError:
                    try:
                        # Try with the parent directory in path
                        from src.models.models import EffNet
                    except ImportError:
                        # Try with the models directory directly
                        from models import EffNet
            except ImportError as e:
                print(f"Error importing EffNet: {e}")
                print(f"Current sys.path: {sys.path}")
                print(f"Looking for module in: {parent_dir}")
                raise ImportError(f"Could not import EffNet model: {e}")
                
            print(f"Loading finetuned EfficientNet model")
            model = EffNet(in_channel=1, stride=2, dilation=1)
            model.fc = torch.nn.Linear(model.fc.in_features, 512)
            # Load finetuned weights
            if model_checkpoint:
                print(f"Loading finetuned weights from {model_checkpoint}")
                state = torch.load(model_checkpoint, map_location=self.device)
                # Remove "module." from the beginning of each key and extract the value
                state = {k[7:]: v[1] for k, v in zip(state.keys(), state.items())}
                model.load_state_dict(state)
            return model.to(self.device)
        elif model_type in ["wav2vec", "wav2vec2", "wav2vec2_base", "wav2vec2_large"]:
            model_class = Wav2Vec2ForCTC if model_checkpoint else Wav2Vec2Model
            return (
                model_class.from_pretrained(model_name, state_dict=state_dict)
                if model_checkpoint
                else model_class.from_pretrained(model_name)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _initialize_processor(self, model_type, processor_name):
        if model_type == "mfcc":
            # MFCC doesn't require a processor, return None
            return None
            
        if model_type == "random-wav2vec":
            return Wav2Vec2FeatureExtractor.from_pretrained(processor_name)
        elif model_type in ["wavlm_sd", "wavlm_large", "wavlm_base", "wav2vec2_base", "wav2vec2_large"]:
            return Wav2Vec2FeatureExtractor.from_pretrained(processor_name)
        elif model_type in ["wav2vec", "wav2vec2", "wav2vec2_hebrew_ft", "wav2vec2_hebrew_initial"]:
            return Wav2Vec2Processor.from_pretrained(processor_name)
        elif model_type in ["voxceleb", "voxceleb_finetuned", "efficientnet", "efficientnet_finetuned"]:
            # These models don't require a processor in the same way
            return None
        return None

    def process_audio(self, audio_file_path: str) -> Union[np.ndarray, Dict]:
        if self.model_type == "mfcc":
            # Extract MFCC features
            audio, sr = librosa.load(audio_file_path, sr=self.mfcc_params["sr"])
            
            # Preprocess audio (remove silence)
            audio = self._preprocess_audio(audio, sr)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.mfcc_params["n_mfcc"]
            )
            return mfcc
            
        elif self.model_type in ["voxceleb", "voxceleb_finetuned"]:
            # Load audio for VoxCeleb models
            waveform, sample_rate = torchaudio.load(audio_file_path)
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            # Process with pyannote model
            with torch.no_grad():
                embeddings = self.model(waveform.to(self.device))
                # Apply normalization as in ssl_pretraining.py
                embeddings = torch.nn.functional.normalize(embeddings)
                return embeddings.cpu().numpy()
                
        elif self.model_type in ["efficientnet", "efficientnet_finetuned"]:
            # Load audio for EfficientNet models
            waveform, sample_rate = torchaudio.load(audio_file_path)
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            # Process with EfficientNet model
            with torch.no_grad():
                # Add channel dimension for EfficientNet (expects [batch, channel, time])
                embeddings = self.model(waveform.unsqueeze(0).to(self.device))
                # Apply normalization as in ssl_pretraining.py
                embeddings = torch.nn.functional.normalize(embeddings)
                return embeddings.cpu().numpy()
            
        elif self.model_type in ["wav2vec", "wavlm_sd", "random-wav2vec", "wav2vec2", "wav2vec2_base", "wav2vec2_large", "wav2vec2_hebrew_ft", "wav2vec2_hebrew_initial", "wavlm_large", "wavlm_base"]:
            audio, sr = librosa.load(audio_file_path, sr=16000)
            inputs = self.processor(
                audio, return_tensors="pt", sampling_rate=16000
            ).input_values.to(self.device)

            with torch.no_grad():
                if self.model_type == "wavlm_sd":
                    outputs = self.model(**{'input_values': inputs})
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits[0])
                    labels = (probabilities > 0.5).long()
                    return {
                        'embeddings': outputs.hidden_states[-1].squeeze(0).cpu().numpy(),
                        'probabilities': probabilities.cpu().numpy(),
                        'labels': labels.cpu().numpy()
                    }
                elif self.model_type in ["wavlm_large", "wavlm_base"]:
                    # WavLM model returns different output structure
                    outputs = self.model(inputs, output_hidden_states=True)
                    # Return the last hidden state as embeddings
                    return outputs.last_hidden_state.squeeze(0).cpu().numpy()
                else:
                    outputs = self.model(inputs, output_hidden_states=True)
                    return outputs.hidden_states[-1].squeeze(0).cpu().numpy()

        elif self.model_type == "xvector":
            signal, _ = torchaudio.load(audio_file_path)
            return self.model.encode_batch(signal).cpu().numpy()

        elif self.model_type in ["emotion_wavlm", "emotion_wav2vec"]:
            signal, sr = torchaudio.load(audio_file_path)
            signal = signal.to(self.device)
            wav_lens = torch.tensor([1.0]).to(self.device)
            return self.model.encode_batch(signal, wav_lens=wav_lens).cpu().numpy()
    
    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio by removing silence from the beginning and end.
        
        Args:
            audio: The audio time series as a numpy array
            sr: The sampling rate of the audio in Hz
            
        Returns:
            The preprocessed audio time series
        """
        # Remove silence from the beginning and end of the audio
        trimmed_audio, _ = librosa.effects.trim(
            audio, 
            top_db=20, 
            frame_length=2048, 
            hop_length=512
        )
        
        return trimmed_audio
            
    def process_audio_batch(self, audio_file_paths: list, batch_size: int = 8) -> list:
        """
        Process a batch of audio files efficiently when GPU is available.
        
        Args:
            audio_file_paths: List of paths to audio files
            batch_size: Number of files to process in each batch
            
        Returns:
            List of results (numpy arrays or dictionaries) for each input file
        """
        results = []
        
        # For MFCC, process files individually since it's CPU-based and doesn't benefit from batching
        if self.model_type == "mfcc":
            for path in audio_file_paths:
                try:
                    result = self.process_audio(path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    results.append(None)
            return results
        
        # For VoxCeleb and EfficientNet models, also process individually for now
        # as they have specific processing requirements
        if self.model_type in ["voxceleb", "voxceleb_finetuned", "efficientnet", "efficientnet_finetuned"]:
            for path in tqdm(audio_file_paths, desc=f"Processing {self.model_type} files"):
                try:
                    result = self.process_audio(path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    results.append(None)
            return results
        
        # Process in batches for other model types
        for i in range(0, len(audio_file_paths), batch_size):
            batch_paths = audio_file_paths[i:i+batch_size]
            batch_results = []
            
            if self.model_type in ["wav2vec", "wavlm_sd", "random-wav2vec", "wav2vec2", "wav2vec2_base", "wav2vec2_large", "wav2vec2_hebrew_ft", "wav2vec2_hebrew_initial", "wavlm_large", "wavlm_base"]:
                # Load and preprocess all audio files in the batch
                batch_inputs = []
                for path in batch_paths:
                    try:
                        audio, sr = librosa.load(path, sr=16000)
                        inputs = self.processor(
                            audio, return_tensors="pt", sampling_rate=16000
                        ).input_values
                        batch_inputs.append(inputs)
                    except Exception as e:
                        print(f"Error loading {path}: {str(e)}")
                        batch_results.append(None)
                        continue
                
                if not batch_inputs:
                    results.extend(batch_results)
                    continue
                    
                # Pad sequences to the same length if needed
                max_length = max(inputs.shape[1] for inputs in batch_inputs)
                padded_inputs = []
                
                for inputs in batch_inputs:
                    if inputs.shape[1] < max_length:
                        padding = torch.zeros(1, max_length - inputs.shape[1])
                        padded_inputs.append(torch.cat([inputs, padding], dim=1))
                    else:
                        padded_inputs.append(inputs)
                
                # Stack inputs and process in a single forward pass
                stacked_inputs = torch.cat(padded_inputs, dim=0).to(self.device)
                
                with torch.no_grad():
                    if self.model_type == "wavlm_sd":
                        outputs = self.model(**{'input_values': stacked_inputs})
                        logits = outputs.logits
                        probabilities = torch.sigmoid(logits)
                        labels = (probabilities > 0.5).long()
                        
                        for i in range(len(batch_paths)):
                            if i >= len(batch_results) or batch_results[i] is None:
                                batch_results.append({
                                    'embeddings': outputs.hidden_states[-1][i].cpu().numpy(),
                                    'probabilities': probabilities[i].cpu().numpy(),
                                    'labels': labels[i].cpu().numpy()
                                })
                    elif self.model_type in ["wavlm_large", "wavlm_base"]:
                        # WavLM model returns different output structure
                        outputs = self.model(stacked_inputs, output_hidden_states=True)
                        for i in range(len(batch_paths)):
                            if i >= len(batch_results) or batch_results[i] is None:
                                batch_results.append(outputs.last_hidden_state[i].cpu().numpy())
                    else:
                        outputs = self.model(stacked_inputs, output_hidden_states=True)
                        for i in range(len(batch_paths)):
                            if i >= len(batch_results) or batch_results[i] is None:
                                batch_results.append(outputs.hidden_states[-1][i].cpu().numpy())
            
            else:
                # For models that don't support batching, process individually
                for path in batch_paths:
                    try:
                        result = self.process_audio(path)
                        batch_results.append(result)
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
                        batch_results.append(None)
            
            results.extend(batch_results)
        
        return results


def process_directory(
        input_directory: str,
        output_directory: str,
        model_type: str,
        model_name: str,
        processor_name: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        use_safetensors: bool = False,
        first_segment_only: bool = False,
        batch_size: int = 8
):
    """
    Process all audio files in a directory, using batch processing when GPU is available.
    
    Args:
        input_directory: Directory containing audio files
        output_directory: Directory to save processed files
        model_type: Type of model to use
        model_name: Name of the model
        processor_name: Name of the processor (optional)
        model_checkpoint: Path to model checkpoint (optional)
        use_safetensors: Whether to use safetensors for loading checkpoint
        first_segment_only: Whether to process only the first segment
        batch_size: Number of files to process in each batch when GPU is available
    """
    os.makedirs(output_directory, exist_ok=True)

    processor = AudioProcessor(
        model_type=model_type,
        model_name=model_name,
        processor_name=processor_name,
        model_checkpoint=model_checkpoint,
        use_safetensors=use_safetensors
    )

    audio_files = [
        f for f in os.listdir(input_directory)
        if f.endswith((".wav", ".mp3", ".flac"))
           and (not first_segment_only or "segment_0" in f)
    ]

    # Filter out already processed files
    unprocessed_files = []
    for audio_file in audio_files:
        output_path = os.path.join(
            output_directory,
            f"{os.path.splitext(audio_file)[0]}.npy"
        )
        if not os.path.exists(output_path):
            unprocessed_files.append(audio_file)
    print("input directory: ", input_directory)
    print("output directory: ", output_directory)
    print(f"Found {len(unprocessed_files)} unprocessed files out of {len(audio_files)} total files")
    
    # Check if GPU is available for batch processing
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU detected. Using batch processing with batch size {batch_size}")
        
        # Process files in batches
        for i in tqdm(range(0, len(unprocessed_files), batch_size), desc="Processing batches"):
            batch_files = unprocessed_files[i:i+batch_size]
            batch_paths = [os.path.join(input_directory, f) for f in batch_files]
            
            try:
                batch_results = processor.process_audio_batch(batch_paths, batch_size)
                
                # Save results
                for file_name, result in zip(batch_files, batch_results):
                    if result is None:
                        continue
                        
                    output_path = os.path.join(
                        output_directory,
                        f"{os.path.splitext(file_name)[0]}"
                    )
                    
                    if isinstance(result, dict):
                        for key, value in result.items():
                            np.save(f"{output_path}_{key}.npy", value)
                    else:
                        np.save(f"{output_path}.npy", result)
            
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print(f"Error details: {type(e).__name__}")
                # Fall back to individual processing for this batch
                for audio_file in batch_files:
                    try:
                        input_path = os.path.join(input_directory, audio_file)
                        output_path = os.path.join(
                            output_directory,
                            f"{os.path.splitext(audio_file)[0]}"
                        )

                        result = processor.process_audio(input_path)

                        if isinstance(result, dict):
                            for key, value in result.items():
                                np.save(f"{output_path}_{key}.npy", value)
                        else:
                            np.save(f"{output_path}.npy", result)

                    except Exception as e:
                        print(f"Error processing {audio_file}: {str(e)}")
                        continue
    else:
        print("GPU not detected. Using single file processing")
        # Process files individually (original method)
        for audio_file in tqdm(unprocessed_files, desc="Processing audio files"):
            try:
                input_path = os.path.join(input_directory, audio_file)
                output_path = os.path.join(
                    output_directory,
                    f"{os.path.splitext(audio_file)[0]}"
                )

                result = processor.process_audio(input_path)

                if isinstance(result, dict):
                    for key, value in result.items():
                        np.save(f"{output_path}_{key}.npy", value)
                else:
                    np.save(f"{output_path}.npy", result)

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                print(f"Error details: {type(e).__name__}")
                continue

def get_default_model(model_type):
    model_configs = {
        "wav2vec": ("facebook/wav2vec2-large-xlsr-53", "srujan00123/wav2vec2-large-medical-speed"),
        "wav2vec2": ("facebook/wav2vec2-large-xlsr-53", "srujan00123/wav2vec2-large-medical-speed"),
        "wav2vec2_base": ("facebook/wav2vec2-base", "facebook/wav2vec2-base"),  # Added Wav2Vec2 base model
        "wav2vec2_large": ("facebook/wav2vec2-large", "facebook/wav2vec2-large"),  # Added Wav2Vec2 large model
        "random-wav2vec": ("facebook/wav2vec2-large-xlsr-53", "facebook/wav2vec2-large-xlsr-53"),  
        "wavlm_sd": ("microsoft/wavlm-base-plus-sd", "microsoft/wavlm-base-plus-sd"),
        "wavlm_base": ("microsoft/wavlm-base", "microsoft/wavlm-base"),  # Added WavLM-base model
        "wavlm_large": ("microsoft/wavlm-large", "microsoft/wavlm-large"),  # Added WavLM-large model
        "xvector": ("speechbrain/spkrec-xvect-voxceleb", None),
        "emotion_wavlm": ("speechbrain/emotion-diarization-wavlm-large", None),
        "emotion_wav2vec": ("speechbrain/emotion-recognition-wav2vec2-IEMOCAP", None),
        "mfcc": ({"n_mfcc": 13, "sr": 16000}, None),  # Default MFCC parameters
        "wav2vec2_hebrew_ft": ("facebook/wav2vec2-large-xlsr-53", "facebook/wav2vec2-large-xlsr-53"),  # Base model for Hebrew fine-tuned model
        "wav2vec2_hebrew_pretrained": ("facebook/wav2vec2-large-xlsr-53", "facebook/wav2vec2-large-xlsr-53"),  # Base model for Hebrew pretrain model
        "voxceleb": ("os.environ.get("HF_TOKEN", "")", None),  # Auth token for VoxCeleb
        "voxceleb_finetuned": ({"auth_token": "os.environ.get("HF_TOKEN", "")"}, None),  # Auth token for VoxCeleb finetuned
        "efficientnet": (None, None),  # EfficientNet doesn't need a pretrained model name
        "efficientnet_finetuned": (None, None)  # EfficientNet finetuned doesn't need a pretrained model name
    }
    return model_configs.get(model_type, (None, None))

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process audio files with different embedding models')
    parser.add_argument('--models', nargs='+', default=['all'], 
                        help='Model types to process (e.g., mfcc wav2vec). Use "all" for all models.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing (default: 16)')
    parser.add_argument('--input-dir', type=str, 
                        default="/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/May25_rerun_tmp_data/trimmed_topDB30_aggregated_audio_full_length_2025May_5s_segments",
                        help='Input directory containing audio files')
    parser.add_argument('--output-suffix', type=str, default="",
                        help='Suffix to add to output directory names')
    parser.add_argument('--first-segment-only', action='store_true',
                        help='Process only the first segment of each audio file')
    args = parser.parse_args()
    
    # Path configurations
    BASE_PATH = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder"
    
    # Common configuration
    base_config = {
        "input_directory": args.input_dir,
        "use_safetensors": False,
        "first_segment_only": False, 
            # args.first_segment_only,
        "batch_size": args.batch_size
    }
    
    # Get all available model types
    all_model_types = [
        # "wav2vec",
        # "wav2vec2",
        # "wav2vec2_base",
        # "wav2vec2_large",
        # "random-wav2vec",
        # "wavlm_sd",
        # "wavlm_base",
        # "wavlm_large",
        # "xvector",
        "emotion_wavlm",
        "emotion_wav2vec",
        # "mfcc",
        # "wav2vec2_hebrew_ft",
        # "wav2vec2_hebrew_initial",
        # "voxceleb",
        # "voxceleb_finetuned",
        # "efficientnet",
        # "efficientnet_finetuned"
    ]
    
    # Filter model types based on command line arguments
    if 'all' not in args.models:
        all_model_types = [model for model in all_model_types if model in args.models]
    
    print(f"Processing the following models: {all_model_types}")
    
    # Process each model type
    for model_type in all_model_types:
        try:
            print(f"\n{'='*50}")
            print(f"Processing with model type: {model_type}")
            print(f"{'='*50}")
            
            # Get model and processor names for this model type
            model_name, processor_name = get_default_model(model_type)
            
            if model_name is None and model_type not in ["efficientnet", "efficientnet_finetuned"]:
                print(f"No default model found for model type: {model_type}. Skipping.")
                continue
            
            # Special handling for MFCC to display parameters
            if model_type == "mfcc":
                print(f"MFCC Parameters: n_mfcc={model_name.get('n_mfcc', 13)}, sr={model_name.get('sr', 16000)}")
            
            # Create model-specific output directory
            output_directory = f"{BASE_PATH}/May25_rerun/{model_type}_embeddings_5sec{args.output_suffix}"
            
            # Special handling for Hebrew models
            model_checkpoint = None
            use_safetensors = False
            
            if model_type == "wav2vec2_hebrew_ft":
                model_checkpoint = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepVoiceFolder/models/finetuned_xlsr_hebrew/checkpoint-1140/model.safetensors"
                use_safetensors = True
                print(f"Using checkpoint: {model_checkpoint}")
                print(f"Using safetensors: {use_safetensors}")
            elif model_type == "wav2vec2_hebrew_initial":
                model_checkpoint = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/output/wav2vec2-multi-large-hebrew__initial/checkpoint-1400/model.safetensors"
                use_safetensors = True
                print(f"Using checkpoint: {model_checkpoint}")
                print(f"Using safetensors: {use_safetensors}")
            elif model_type == "voxceleb_finetuned":
                model_checkpoint = f"{BASE_PATH}/model_SSL_VoxCeleb_b40_lr1e-06_scReduceLROnPlateau_MaxTrainRecall@1.pth"
                print(f"Using VoxCeleb finetuned checkpoint: {model_checkpoint}")
            elif model_type == "efficientnet_finetuned":
                model_checkpoint = f"{BASE_PATH}/model_SSL_EfficientNet_b10_lr0.0001_scReduceLROnPlateau_MaxTrainRecall@1.pth"
                print(f"Using EfficientNet finetuned checkpoint: {model_checkpoint}")
            
            print(f"Model: {model_name}")
            print(f"Processor: {processor_name}")
            print(f"Output directory: {output_directory}")
            
            # Create a complete config for this model
            model_config = base_config.copy()
            model_config["model_type"] = model_type
            model_config["output_directory"] = output_directory
            model_config["use_safetensors"] = use_safetensors
            
            # Process the directory with this model
            process_directory(
                input_directory=model_config["input_directory"],
                output_directory=model_config["output_directory"],
                model_type=model_type,
                model_name=model_name,
                processor_name=processor_name,
                model_checkpoint=model_checkpoint,
                use_safetensors=model_config["use_safetensors"],
                first_segment_only=model_config["first_segment_only"],
                batch_size=model_config["batch_size"]
            )
            
            print(f"Completed processing with model type: {model_type}")
            
        except Exception as e:
            print(f"Error processing model type {model_type}: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            print("Continuing with next model type...")
            continue
    
    print("\nAll model types processed successfully!")