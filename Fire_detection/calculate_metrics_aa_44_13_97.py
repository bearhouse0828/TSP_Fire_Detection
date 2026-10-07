#!/usr/bin/env python3
"""
Calculate Precision, Recall, and F1-score for aa-44-13-97 station
against Dae-Cheong Lake real fire labels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_dae_cheong_lake_real_fires():
    """Get real fire events for Dae-Cheong Lake (KST times)"""
    real_fires = [
        {
            'start': '2025-09-11 19:00:00',  # KST
            'end': '2025-09-11 19:25:00',    # KST
            'description': 'Sep 11, 19:00-19:25 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-12 08:15:00',  # KST
            'end': '2025-09-12 08:45:00',    # KST
            'description': 'Sep 12, 08:15-08:45 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-16 19:10:00',  # KST
            'end': '2025-09-16 20:00:00',    # KST
            'description': 'Sep 16, 19:10-20:00 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-17 09:20:00',  # KST
            'end': '2025-09-17 10:25:00',    # KST
            'description': 'Sep 17, 09:20-10:25 KST (Dae-Cheong Lake)'
        }
    ]
    
    # Convert KST to UTC (subtract 9 hours)
    utc_fires = []
    for fire in real_fires:
        start_kst = pd.to_datetime(fire['start'])
        end_kst = pd.to_datetime(fire['end'])
        
        start_utc = start_kst - timedelta(hours=9)
        end_utc = end_kst - timedelta(hours=9)
        
        # Ensure UTC timezone awareness
        start_utc = start_utc.tz_localize('UTC') if start_utc.tz is None else start_utc
        end_utc = end_utc.tz_localize('UTC') if end_utc.tz is None else end_utc
        
        utc_fires.append({
            'start': start_utc,
            'end': end_utc,
            'description': fire['description']
        })
    
    return utc_fires

def load_detection_events(station_id='aa-44-13-97'):
    """Load detection events for the specified station"""
    try:
        # Load single-station events
        single_file = f"outputs/{station_id}/{station_id}_pm_fire_events.csv"
        single_events = pd.read_csv(single_file)
        single_events['start'] = pd.to_datetime(single_events['start'])
        single_events['end'] = pd.to_datetime(single_events['end'])
        
        # Load TSPulse events
        tsp_df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
        flag_col = f"{station_id}_tspulse_flag"
        
        if flag_col in tsp_df.columns:
            # Convert TSPulse flags to events
            tsp_events = []
            in_event = False
            event_start = None
            
            for timestamp, row in tsp_df.iterrows():
                if row[flag_col] == 1 and not in_event:
                    # Start of event
                    in_event = True
                    event_start = timestamp
                elif row[flag_col] == 0 and in_event:
                    # End of event
                    in_event = False
                    if event_start is not None:
                        tsp_events.append({
                            'start': event_start,
                            'end': timestamp,
                            'method': 'TSPulse'
                        })
            
            # Handle case where event continues to end of data
            if in_event and event_start is not None:
                tsp_events.append({
                    'start': event_start,
                    'end': tsp_df.index[-1],
                    'method': 'TSPulse'
                })
            
            tsp_events_df = pd.DataFrame(tsp_events)
        else:
            tsp_events_df = pd.DataFrame()
        
        return single_events, tsp_events_df
        
    except Exception as e:
        print(f"❌ Error loading detection events: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_overlap(detected_start, detected_end, real_start, real_end, tolerance_minutes=30):
    """Calculate if detected event overlaps with real fire event within tolerance"""
    # Add tolerance window
    real_start_tol = real_start - timedelta(minutes=tolerance_minutes)
    real_end_tol = real_end + timedelta(minutes=tolerance_minutes)
    
    # Check for overlap
    return (detected_start <= real_end_tol) and (detected_end >= real_start_tol)

def calculate_metrics(single_events, tsp_events, real_fires, tolerance_minutes=30):
    """Calculate Precision, Recall, and F1-score for both methods"""
    
    print(f"📊 Calculating metrics with {tolerance_minutes}-minute tolerance window")
    print(f"📍 Real fire events: {len(real_fires)}")
    print(f"🔍 Single-station detections: {len(single_events)}")
    print(f"🔍 TSPulse detections: {len(tsp_events)}")
    print()
    
    results = {}
    
    for method_name, detected_events in [("Single-station", single_events), ("TSPulse", tsp_events)]:
        if detected_events.empty:
            print(f"⚠️  No {method_name} detections found")
            results[method_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(real_fires)
            }
            continue
        
        # Calculate True Positives (TP): detected events that overlap with real fires
        true_positives = 0
        matched_real_fires = set()
        
        for _, detected in detected_events.iterrows():
            for i, real_fire in enumerate(real_fires):
                if calculate_overlap(detected['start'], detected['end'], 
                                  real_fire['start'], real_fire['end'], tolerance_minutes):
                    true_positives += 1
                    matched_real_fires.add(i)
                    break  # Each detection can only match one real fire
        
        # Calculate False Positives (FP): detected events that don't overlap with any real fire
        false_positives = len(detected_events) - true_positives
        
        # Calculate False Negatives (FN): real fires that weren't detected
        false_negatives = len(real_fires) - len(matched_real_fires)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[method_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        print(f"📈 {method_name} Results:")
        print(f"   True Positives (TP): {true_positives}")
        print(f"   False Positives (FP): {false_positives}")
        print(f"   False Negatives (FN): {false_negatives}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print()
    
    return results

def print_detailed_analysis(single_events, tsp_events, real_fires, tolerance_minutes=30):
    """Print detailed analysis of detections vs real fires"""
    print("🔍 Detailed Analysis:")
    print("=" * 50)
    
    # Real fire events
    print("📍 Real Fire Events (Dae-Cheong Lake):")
    for i, fire in enumerate(real_fires):
        print(f"   {i+1}. {fire['description']}")
        print(f"      UTC: {fire['start']} - {fire['end']}")
    print()
    
    # Single-station detections
    if not single_events.empty:
        print("🔍 Single-station Detections:")
        for i, (_, event) in enumerate(single_events.iterrows()):
            print(f"   {i+1}. {event['start']} - {event['end']} (Score: {event['score']:.3f})")
            
            # Check which real fires this detection matches
            matches = []
            for j, real_fire in enumerate(real_fires):
                if calculate_overlap(event['start'], event['end'], 
                                  real_fire['start'], real_fire['end'], tolerance_minutes):
                    matches.append(f"Real Fire {j+1}")
            
            if matches:
                print(f"      ✅ Matches: {', '.join(matches)}")
            else:
                print(f"      ❌ No match with real fires")
        print()
    
    # TSPulse detections
    if not tsp_events.empty:
        print("🔍 TSPulse Detections:")
        for i, (_, event) in enumerate(tsp_events.iterrows()):
            print(f"   {i+1}. {event['start']} - {event['end']}")
            
            # Check which real fires this detection matches
            matches = []
            for j, real_fire in enumerate(real_fires):
                if calculate_overlap(event['start'], event['end'], 
                                  real_fire['start'], real_fire['end'], tolerance_minutes):
                    matches.append(f"Real Fire {j+1}")
            
            if matches:
                print(f"      ✅ Matches: {', '.join(matches)}")
            else:
                print(f"      ❌ No match with real fires")
        print()

def main():
    """Main function to calculate metrics"""
    print("🔥 aa-44-13-97 Station Performance Analysis")
    print("=" * 60)
    print("Target: Dae-Cheong Lake Real Fire Events")
    print()
    
    # Load data
    real_fires = get_dae_cheong_lake_real_fires()
    single_events, tsp_events = load_detection_events('aa-44-13-97')
    
    # Print detailed analysis
    print_detailed_analysis(single_events, tsp_events, real_fires)
    
    # Calculate metrics with different tolerance windows
    tolerance_windows = [15, 30, 60]  # minutes
    
    for tolerance in tolerance_windows:
        print(f"📊 Metrics with {tolerance}-minute tolerance window:")
        print("-" * 40)
        results = calculate_metrics(single_events, tsp_events, real_fires, tolerance)
        
        # Print summary table
        print("📋 Summary Table:")
        print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        for method, metrics in results.items():
            print(f"{method:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")
        print()
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
