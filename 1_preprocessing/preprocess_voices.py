import os
import librosa
import soundfile as sf
from tqdm import tqdm

from src.utils.utils import mkdirifnotexists


def preprocess_audio(audio, target_sr=22050, duration=30):
    """
    Normalize, trim/pad an audio signal to a fixed length.

    Args:
    audio (np.ndarray): Audio signal.
    target_sr (int): Target sampling rate.
    duration (int): Target duration of the file in seconds.

    Returns:
    np.ndarray: The processed audio signal.
    """
    # Normalize the audio to unit variance
    audio = librosa.util.normalize(audio)

    # Trim silence from the beginning and end
    audio, _ = librosa.effects.trim(audio)

    # Fix the audio length to 'duration'. Pad if it's shorter, trim if it's longer.
    fixed_length = target_sr * duration
    if len(audio) < fixed_length:
        # this will pad the signal at the beginning and end with zeros
        audio = librosa.util.fix_length(audio, size=fixed_length)
    else:
        audio = audio[:fixed_length]

    return audio


def preprocess_audio_into_segments(
    audio, target_sr=22050, segment_duration=10, trim_start_seconds=0
):
    """
    Normalize, trim silence, and divide an audio signal into fixed-length segments.

    Args:
    audio (np.ndarray): Audio signal.
    target_sr (int): Target sampling rate.
    segment_duration (int): Duration of each segment in seconds.

    Returns:
    list of np.ndarray: A list of the processed audio segments.
    """
    # Normalize the audio to unit variance
    audio = librosa.util.normalize(audio)

    # Trim silence from the beginning and end
    audio, _ = librosa.effects.trim(audio)

    # Trim the first x seconds from the start of the audio
    if trim_start_seconds > 0:
        trim_start_samples = int(trim_start_seconds * target_sr)
        if len(audio) > trim_start_samples:
            audio = audio[trim_start_samples:]
        else:
            # If the audio is shorter than the trim length, return an empty list
            return []

    # Calculate the number of samples per segment
    samples_per_segment = target_sr * segment_duration

    # Calculate the number of segments
    num_segments = len(audio) // samples_per_segment

    # Divide the audio into segments
    segments = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = audio[start:end]
        segments.append(segment)

    return segments


def process_files(base_dir, output_dir):
    """
    Process all audio files in the specified directory and its subdirectories.

    Args:
    base_dir (str): Base directory containing audio files.
    output_dir (str): Directory where processed files will be saved.
    """
    global processed_audio
    for root, dirs, files in os.walk(base_dir):
        for file in tqdm(files, desc="Processing files", unit="files"):
            if file.endswith((".flac", ".wav")):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=22050)

                # Process the audio
                processed_audio = preprocess_audio(audio)
                # Determine output path
                if root == base_dir:
                    output_subdir = "unknown_years"
                    output_file = file  # Keep original filename
                    # change the extension to .flac
                    output_file = os.path.splitext(output_file)[0] + ".flac"
                else:
                    # Extract years_from_baseline and participant ID from the file structure
                    parts = root.strip("/").split("/")
                    years_from_baseline = (
                        parts[-1] if parts[-1].endswith("_visit") else None
                    )
                    participant_id = parts[-2] if years_from_baseline else None

                    if not participant_id and not years_from_baseline:
                        continue  # Skip if the directory structure is not as expected

                    output_subdir = years_from_baseline
                    output_file = f"{participant_id}.flac"

                output_subdir = os.path.join(
                    output_dir, output_subdir
                )  # Correct the output subdir path
                output_path = os.path.join(output_subdir, output_file)

                # Create directory if it doesn't exist
                os.makedirs(output_subdir, exist_ok=True)

                # Save the processed audio
                sf.write(output_path, processed_audio, 22050)


def process_files_into_segments(
    base_dir, output_dir, _segment_duration=10, trim_start_seconds=0
):
    """
    Process all audio files in the specified directory and its subdirectories,
    dividing them into segments and saving each segment with an enumerated suffix.

    Args:
    base_dir (str): Base directory containing audio files.
    output_dir (str): Directory where processed files will be saved.
    """
    # initialize participant_id
    participant_id = None

    mkdirifnotexists(output_dir)
    for root, dirs, files in os.walk(base_dir):
        for file in tqdm(files, desc="Processing files", unit="files"):
            if file.endswith((".flac", ".wav")):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=22050)

                # Process the audio into segments
                processed_segments = preprocess_audio_into_segments(
                    audio=audio,
                    segment_duration=_segment_duration,
                    trim_start_seconds=trim_start_seconds,
                )

                # Determine the output directory structure
                if root == base_dir:
                    output_subdir = "unknown_years"
                else:
                    # Extract date, research stage and participant ID from the file structure
                    date = file.split(".")[0].replace("_", "")
                    parts = root.strip("/").split("/")
                    research_stage = parts[-1] if parts[-1].endswith("_visit") else None
                    participant_id = parts[-2] if research_stage else None

                    if not participant_id and not research_stage:
                        continue  # Skip if the directory structure is not as expected

                    output_subdir = research_stage

                # Create the output directory if it doesn't exist
                output_subdir = os.path.join(output_dir, output_subdir)
                os.makedirs(output_subdir, exist_ok=True)

                # Save each segment with an enumerated suffix
                for i, segment in enumerate(processed_segments):
                    if root == base_dir:
                        output_file = f"{os.path.splitext(file)[0]}_{i}.flac"
                        output_path = os.path.join(output_subdir, output_file)
                        sf.write(output_path, segment, 22050)
                    else:
                        output_file = f"{participant_id}_{date}_{i}.flac"
                        output_path = os.path.join(output_subdir, output_file)
                        sf.write(output_path, segment, 22050)


if __name__ == "__main__":
    # Define your base and output directories
    base_dir = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/voice"
    # base_dir = '/net/mraid20/export/genie/LabData/Analyses/davidkro/Voice/raw_recordings'
    output_dir = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/processed/voice_segments_10s"

    process_files_into_segments(
        base_dir, output_dir, _segment_duration=10, trim_start_seconds=0
    )
    # process_files(base_dir, output_dir)
