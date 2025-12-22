import argparse
import json
import os
import base64
from pathlib import Path
import pandas as pd
import requests

# OpenRouter configuration
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
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
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
    # Load truth data from JSON file
    truth_path = Path("emotify_song_level_stats.json")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth data file not found: {truth_path}")
    
    with open(truth_path, 'r') as f:
        truth_data = json.load(f)
    
    # Load listener counts from CSV data (embedded)
    listeners = {"song_1": 48, "song_10": 17, "song_11": 50, "song_12": 45, "song_13": 27, "song_14": 11, "song_15": 11, "song_16": 12, "song_17": 11, "song_18": 11, "song_19": 11, "song_2": 47, "song_20": 11, "song_21": 53, "song_22": 45, "song_23": 19, "song_24": 12, "song_25": 24, "song_26": 12, "song_27": 12, "song_28": 28, "song_29": 12, "song_3": 45, "song_30": 12, "song_31": 50, "song_32": 46, "song_33": 26, "song_34": 14, "song_35": 13, "song_36": 14, "song_37": 14, "song_38": 13, "song_39": 14, "song_4": 46, "song_40": 14, "song_5": 17, "song_6": 17, "song_7": 17, "song_8": 18, "song_9": 17}
    
    return truth_data, listeners

def calculate_differences(truth, pred):
    """Calculate differences between human and Gemini for each emotion"""
    emotions = ["amazement", "solemnity", "tenderness", "nostalgia", "calmness", 
               "power", "joyful_activation", "tension", "sadness"]
    
    differences = {}
    
    for emotion in emotions:
        if emotion in truth and emotion in pred:
            diff_mean = abs(truth[emotion]["mean"] - pred[emotion]["mean"])
            diff_std = abs(truth[emotion]["std"] - pred[emotion]["std"])
            differences[emotion] = {
                "diff_mean": diff_mean,
                "diff_std": diff_std
            }
        else:
            differences[emotion] = {
                "diff_mean": 1.0,  # Max possible difference
                "diff_std": 0.5    # Max possible std difference
            }
    
    return differences

def create_overall_summary_table(all_differences):
    """Create Table 1: Overall summary across all songs and emotions"""
    emotions = ["amazement", "solemnity", "tenderness", "nostalgia", "calmness", 
               "power", "joyful_activation", "tension", "sadness"]
    
    all_mean_diffs = []
    all_std_diffs = []
    
    # Collect all differences
    for song_diffs in all_differences.values():
        for emotion in emotions:
            if emotion in song_diffs:
                all_mean_diffs.append(song_diffs[emotion]["diff_mean"])
                all_std_diffs.append(song_diffs[emotion]["diff_std"])
    
    # Calculate overall averages
    avg_mean_diff = sum(all_mean_diffs) / len(all_mean_diffs) if all_mean_diffs else 0
    avg_std_diff = sum(all_std_diffs) / len(all_std_diffs) if all_std_diffs else 0
    
    # Create summary table
    summary_data = {
        "Metric": ["Average Difference in Mean", "Average Difference in Std"],
        "Value": [f"{avg_mean_diff:.4f}", f"{avg_std_diff:.4f}"]
    }
    
    return pd.DataFrame(summary_data)

def create_song_level_breakdown_table(all_differences):
    """Create Table 2: Song-level breakdown with emotion-wise differences"""
    emotions = ["amazement", "solemnity", "tenderness", "nostalgia", "calmness", 
               "power", "joyful_activation", "tension", "sadness"]
    
    rows = []
    
    for song_id, song_diffs in all_differences.items():
        # Mean differences row
        mean_row = {"song_id": song_id, "metric": "diff_mean"}
        for emotion in emotions:
            if emotion in song_diffs:
                mean_row[emotion] = f"{song_diffs[emotion]['diff_mean']:.4f}"
            else:
                mean_row[emotion] = "N/A"
        rows.append(mean_row)
        
        # Std differences row
        std_row = {"song_id": song_id, "metric": "diff_std"}
        for emotion in emotions:
            if emotion in song_diffs:
                std_row[emotion] = f"{song_diffs[emotion]['diff_std']:.4f}"
            else:
                std_row[emotion] = "N/A"
        rows.append(std_row)
    
    return pd.DataFrame(rows)

def main():
    data_dir = Path("data/raw")
    
    try:
        truth_data, listeners = load_ground_truth_and_listeners()
        print(f"Loaded truth data for {len(truth_data)} songs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    all_differences = {}
    
    for audio_file in data_dir.glob("song_*.opus"):
        song_id = audio_file.stem
        
        if song_id not in truth_data or song_id not in listeners:
            print(f"Skipping {song_id}: no truth data or listener count")
            continue
        
        print(f"Processing {song_id}...")
        
        try:
            n_listeners = listeners[song_id]
            pred = analyze_audio(audio_file, n_listeners)
            truth = truth_data[song_id]
            
            # Calculate differences (not accuracy)
            differences = calculate_differences(truth, pred)
            all_differences[song_id] = differences
            
            # Show sample differences for this song
            print(f"  Sample diffs - amazement_mean: {differences['amazement']['diff_mean']:.4f}, "
                  f"calmness_std: {differences['calmness']['diff_std']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_differences:
        print(f"\nProcessed {len(all_differences)} songs")
        
        # Create Table 1: Overall Summary
        summary_table = create_overall_summary_table(all_differences)
        summary_table.to_csv("table1_overall_summary.csv", index=False)
        
        # Create Table 2: Song-level Breakdown
        breakdown_table = create_song_level_breakdown_table(all_differences)
        breakdown_table.to_csv("table2_song_breakdown.csv", index=False)
        
    else:
        print("No results to save")

if __name__ == "__main__":
    main()