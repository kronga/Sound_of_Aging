import os
import multiprocessing as mp
from pydub import AudioSegment
from pathlib import Path

def process_single_file(args):
    """Process a single FLAC file"""
    input_path, output_folder, segment_length_seconds = args
    
    try:
        audio = AudioSegment.from_file(input_path, format="flac")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        segment_length_ms = segment_length_seconds * 1000
        
        segment_count = 0
        for start_ms in range(0, len(audio), segment_length_ms):
            end_ms = min(start_ms + segment_length_ms, len(audio))
            segment = audio[start_ms:end_ms]
            
            output_filename = f"{base_name}_segment_{segment_count}.flac"
            output_path = os.path.join(output_folder, output_filename)
            segment.export(output_path, format="flac")
            segment_count += 1
        
        return f"✓ {base_name}: {segment_count} segments"
        
    except Exception as e:
        return f"✗ {os.path.basename(input_path)}: {str(e)}"

def split_flac_files_parallel(input_folder, output_folder, segment_length_seconds=5):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel processing
    flac_files = [os.path.join(input_folder, f) 
                  for f in os.listdir(input_folder) 
                  if f.lower().endswith('.flac')]
    
    args_list = [(f, output_folder, segment_length_seconds) for f in flac_files]
    
    # Process files in parallel
    with mp.Pool(processes=18) as pool:
        results = pool.map(process_single_file, args_list)
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    input_folder = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/May25_rerun_tmp_data/trimmed_topDB30_aggregated_audio_full_length_2025May"
    output_folder = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/May25_rerun_tmp_data/trimmed_topDB30_aggregated_audio_full_length_2025May_5s_segments"     
    segment_length_seconds = 5
    # Usage for parallel version
    split_flac_files_parallel(input_folder, output_folder, segment_length_seconds)