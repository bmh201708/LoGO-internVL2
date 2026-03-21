import os
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf(sentence1, sentence2):
    if not sentence1.strip() and not sentence2.strip():
        return 1.0
    if not sentence1.strip() or not sentence2.strip():
        return 0.0
    if sentence1.strip().lower() == sentence2.strip().lower():
        return 1.0
        
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except ValueError:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Turn Inference Results")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the JSONL inference results file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_path):
        print(f"[ERROR] Results file not found: {args.results_path}")
        return
        
    all_step_results = []
    all_episode_results = []
    
    print(f"\n[INFO] Starting Evaluation on: {args.results_path}")
    
    mode = "UNKNOWN"
    total_episodes = 0
    
    with open(args.results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            entry = json.loads(line.strip())
            total_episodes += 1
            mode = entry.get("mode", "UNKNOWN")
            
            episode_id = entry.get("episode_id", "Unknown")
            gts = entry.get("ground_truths", [])
            preds = entry.get("predictions", [])
            
            if len(gts) != len(preds):
                print(f"[WARNING] Episode {episode_id} length mismatch! GTs: {len(gts)}, Preds: {len(preds)}")
                continue
                
            episode_correct = True
            
            for k, (gt, pred) in enumerate(zip(gts, preds)):
                similarity = calculate_tfidf(gt, pred)
                step_correct = (similarity > 0.6)
                all_step_results.append(step_correct)
                
                if not step_correct:
                    episode_correct = False
                    
            all_episode_results.append(episode_correct)
            
    if not all_step_results:
        print("[ERROR] No valid steps found in the results file.")
        return
        
    step_accuracy = sum(all_step_results) / len(all_step_results)
    episode_accuracy = sum(all_episode_results) / len(all_episode_results)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT | Original Inference Mode: {mode.upper()}")
    print(f"Total Episodes Processed: {total_episodes}")
    print(f"Total Steps Evaluated   : {len(all_step_results)}")
    print(f"{'-'*60}")
    
    if mode.lower() == "step":
        print(f"Step-level Accuracy (TF-IDF > 0.6)      : {step_accuracy:.2%}")
        print(f" * Computed independently per ground-truth-primed step.")
    elif mode.lower() == "episode":
        print(f"Episode-level Accuracy (All steps > 0.6): {episode_accuracy:.2%}")
        print(f" * Requires cascading success across the entire autoregressive chain.")
    else:
        print(f"Step-level Accuracy     : {step_accuracy:.2%}")
        print(f"Episode-level Accuracy  : {episode_accuracy:.2%}")
        
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
