import argparse
import json
import os
import base64
from pathlib import Path
import pandas as pd
import requests

OPENROUTER_URL = "https://llmfoundry.straive.com/openrouter/v1/chat/completions"
MODEL_NAME = "google/gemini-3-pro-preview"

def get_gems_prompt(n_listeners):
    return f"""You are analyzing a music audio clip.
This audio has been listened to and rated by N people.
Each person indicated whether they strongly felt each of the following emotions while listening.
The emotions are based on the Geneva Emotional Music Scales (GEMS):
- amazement (wonder, awe, happiness)
- solemnity (transcendence, inspiration, thrills)
- tenderness (sensuality, affect, feeling of love)
- nostalgia (dreamy, melancholic, sentimental)
- calmness (relaxation, serenity, meditative)
- power (strong, heroic, triumphant, energetic)
- joyful_activation (bouncy, animated, like dancing)
- tension (nervous, impatient, irritated)
- sadness (depressed, sorrowful)
Each listener gives a binary response (0 or 1) for each emotion.
Task:
Assume N = {n_listeners} listeners.
Based only on the audio content:
For each emotion, estimate:
1. The average rating (mean, between 0 and 1)
2. The standard deviation of the ratings (between 0 and 0.5)
Output requirements:
- Return JSON only
- Use exactly the emotion keys listed above
- Each emotion must contain:
  - "mean": float
  - "std": float
- Do NOT include explanations, comments, or extra text
- Do NOT add or remove emotions
- Do NOT round excessively (use up to 4 decimal places)
Output JSON schema:
{{
  "amazement": {{ "mean": 0.0, "std": 0.0 }},
  "solemnity": {{ "mean": 0.0, "std": 0.0 }},
  "tenderness": {{ "mean": 0.0, "std": 0.0 }},
  "nostalgia": {{ "mean": 0.0, "std": 0.0 }},
  "calmness": {{ "mean": 0.0, "std": 0.0 }},
  "power": {{ "mean": 0.0, "std": 0.0 }},
  "joyful_activation": {{ "mean": 0.0, "std": 0.0 }},
  "tension": {{ "mean": 0.0, "std": 0.0 }},
  "sadness": {{ "mean": 0.0, "std": 0.0 }}
}}"""

def analyze_audio(audio_path, n_listeners):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    
    with open(audio_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": get_gems_prompt(n_listeners)},
                {"type": "audio", "audio": {"data": audio_b64, "format": "audio/ogg"}}
            ]
        }]
    }
    
    headers = {"Authorization": f"Bearer {api_key}","Content-Type": "application/json"}
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")
    
    content = response.json()['choices'][0]['message']['content']
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {}

def load_ground_truth_and_listeners():
    truth_path = Path("truth.json")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth data file not found: {truth_path}")
    
    with open(truth_path, 'r') as f:
        truth_data = json.load(f)
    
    listeners = {"song_1": 48, "song_10": 17, "song_11": 50, "song_12": 45, "song_13": 27, "song_14": 11, "song_15": 11, "song_16": 12, "song_17": 11, "song_18": 11, "song_19": 11, "song_2": 47, "song_20": 11, "song_21": 53, "song_22": 45, "song_23": 19, "song_24": 12, "song_25": 24, "song_26": 12, "song_27": 12, "song_28": 28, "song_29": 12, "song_3": 45, "song_30": 12, "song_31": 50, "song_32": 46, "song_33": 26, "song_34": 14, "song_35": 13, "song_36": 14, "song_37": 14, "song_38": 13, "song_39": 14, "song_4": 46, "song_40": 14, "song_5": 17, "song_6": 17, "song_7": 17, "song_8": 18, "song_9": 17}
    
    return truth_data, listeners

def create_comparison_csvs(emotify_data, gemini_data):
    """Create two CSV files comparing emotify and gemini data"""
    emotions = ['amazement', 'solemnity', 'tenderness', 'nostalgia', 'calmness', 
                'power', 'joyful_activation', 'tension', 'sadness']
    
    means_data = []
    stds_data = []
    
    # Sort songs by number
    sorted_songs = sorted(emotify_data.keys(), key=lambda x: int(x.split('_')[1]))
    
    for song in sorted_songs:
        if song not in gemini_data:
            continue
            
        # Means - Gemini
        gemini_mean_row = {'song': song, 'metric': 'mean', 'source': 'gemini'}
        for emotion in emotions:
            gemini_mean_row[emotion] = gemini_data[song][emotion]['mean']
        means_data.append(gemini_mean_row)
        
        # Means - Emotify
        emotify_mean_row = {'song': song, 'metric': 'mean', 'source': 'emotify'}
        for emotion in emotions:
            emotify_mean_row[emotion] = emotify_data[song][emotion]['mean']
        means_data.append(emotify_mean_row)
        
        # Means - Diff (Gemini - Emotify)
        diff_mean_row = {'song': song, 'metric': 'mean', 'source': 'diff'}
        for emotion in emotions:
            diff = gemini_data[song][emotion]['mean'] - emotify_data[song][emotion]['mean']
            diff_mean_row[emotion] = round(diff, 4)
        means_data.append(diff_mean_row)
        
        # Stds - Gemini
        gemini_std_row = {'song': song, 'metric': 'std', 'source': 'gemini'}
        for emotion in emotions:
            gemini_std_row[emotion] = gemini_data[song][emotion]['std']
        stds_data.append(gemini_std_row)
        
        # Stds - Emotify
        emotify_std_row = {'song': song, 'metric': 'std', 'source': 'emotify'}
        for emotion in emotions:
            emotify_std_row[emotion] = emotify_data[song][emotion]['std']
        stds_data.append(emotify_std_row)
        
        # Stds - Diff (Gemini - Emotify)
        diff_std_row = {'song': song, 'metric': 'std', 'source': 'diff'}
        for emotion in emotions:
            diff = gemini_data[song][emotion]['std'] - emotify_data[song][emotion]['std']
            diff_std_row[emotion] = round(diff, 4)
        stds_data.append(diff_std_row)
    
    # Create DataFrames and save to CSV
    means_df = pd.DataFrame(means_data)
    stds_df = pd.DataFrame(stds_data)
    
    means_df.to_csv('means_comparison.csv', index=False)
    stds_df.to_csv('stds_comparison.csv', index=False)
    
    return means_df, stds_df

def main():
    data_dir = Path("data/raw")
    gemini_output_file = "gemini_output.json"
    
    try:
        truth_data, listeners = load_ground_truth_and_listeners()
        print(f"Loaded truth data for {len(truth_data)} songs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load existing Gemini data if available
    gemini_data = {}
    if Path(gemini_output_file).exists():
        with open(gemini_output_file, 'r') as f:
            gemini_data = json.load(f)
        print(f"Loaded existing Gemini data for {len(gemini_data)} songs")
    
    # Process audio files and get Gemini predictions
    for audio_file in data_dir.glob("song_*.opus"):
        song_id = audio_file.stem
        
        if song_id not in truth_data or song_id not in listeners:
            print(f"Skipping {song_id}: no truth data or listener count")
            continue
            
        if song_id in gemini_data:
            print(f"Skipping {song_id}: already processed")
            continue
        
        print(f"Processing {song_id}...")
        try:
            n_listeners = listeners[song_id]
            pred = analyze_audio(audio_file, n_listeners)
            gemini_data[song_id] = pred
            
            # Save after each successful prediction
            with open(gemini_output_file, 'w') as f:
                json.dump(gemini_data, f, indent=2)
            
            print(f"  Successfully processed {song_id}")
            
        except Exception as e:
            print(f"  Error processing {song_id}: {e}")
    
    # Create comparison CSV files
    if gemini_data:
        print(f"\nCreating comparison files with {len(gemini_data)} songs...")
        means_df, stds_df = create_comparison_csvs(truth_data, gemini_data)
        print(f"Created means_comparison.csv with {len(means_df)} rows")
        print(f"Created stds_comparison.csv with {len(stds_df)} rows")
    else:
        print("No Gemini data available to create comparison files")

if __name__ == "__main__":
    main()