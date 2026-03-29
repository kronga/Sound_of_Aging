import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing as mp
import pickle


def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)  # Fixed sampling rate

        features = {
            'rms_energy': np.mean(librosa.feature.rms(y=y)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'mfccs': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mels=128), axis=1)
        }

        mfcc_features = {f'mfcc_{i}': val for i, val in enumerate(features['mfccs'])}
        features.update(mfcc_features)
        del features['mfccs']

        return features, None
    except Exception as e:
        return None, str(e)


from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

# Top-level function: must be here for multiprocessing
def process_single_row(args):
    filename, row_dict, audio_dir = args
    audio_path = os.path.join(audio_dir, filename)
    features, error = extract_audio_features(audio_path)
    if features:
        features['filename'] = filename
        features['quality'] = row_dict.get('quality', None)
        return features
    return None

def extract_and_save_features_parallel(df, audio_dir, output_file, num_workers=None):
    if num_workers is None:
        num_workers = 30

    # Create a list of arguments: (filename, row_dict, audio_dir)
    tasks = [(idx, row.to_dict(), audio_dir) for idx, row in df.iterrows()]

    results = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_row, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} feature records to {output_file}")

def append_missing_audio_files_with_index(table_path, audio_dir, output_path=None):
    # Load table with filename as index
    df = pd.read_csv(table_path, index_col=0)

    # Existing filenames (from index)
    existing_filenames = set(df.index.map(os.path.basename))

    # Audio filenames in the directory
    audio_filenames = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    audio_filenames_set = set(audio_filenames)

    # Find missing filenames
    missing_filenames = audio_filenames_set - existing_filenames
    print(f"Found {len(missing_filenames)} missing audio files.")

    # Append missing files
    if missing_filenames:
        new_rows = pd.DataFrame(index=sorted(missing_filenames))
        df = pd.concat([df, new_rows], axis=0)

    # Save or return
    if output_path:
        df.to_csv(output_path)
        print(f"Updated table saved to {output_path}")
    return df


if __name__ == "__main__":


    # Paths
    DEEPVOICE_DIR = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/"
    AUDIO_DIR = os.path.join(DEEPVOICE_DIR, "Oct25_voice_full_length", "Raw_voice")

    LABELED_DATA_PATH = "~/PycharmProjects/DeepVoice/data/audio_quality_tags - tagging.csv"
    FEATURES_OUTPUT = os.path.join(DEEPVOICE_DIR, "Oct25_voice_full_length", "audio_features_for_noise_filtering.pkl")

    df = append_missing_audio_files_with_index(LABELED_DATA_PATH, AUDIO_DIR, output_path=None)
    extract_and_save_features_parallel(df, AUDIO_DIR, FEATURES_OUTPUT, num_workers=30)