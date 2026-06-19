import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

def format_prompt(question, context_docs):
    context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(context_docs)])
    prompt = f'''You are an advanced Evidence-Aware AI Assistant. 
Answer the following user question using ONLY the provided verified evidence.
If the evidence does not contain the answer, say "I don't have enough verified evidence to answer this."

Evidence:
{context}

Question:
{question}

Answer:'''
    return prompt

def generate_dataset(output_dir="data/finetune", num_samples=2000):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    print("Loading HotpotQA dataset for RAG Synthesis...")
    try:
        dataset = load_dataset('parquet', data_files='hf://datasets/hotpot_qa/distractor/train-00000-of-00001.parquet', split='train')
    except Exception as e:
        print(f"Failed to load HotpotQA parquet: {e}")
        print("Falling back to standard load_dataset...")
        dataset = load_dataset('hotpot_qa', 'distractor', split='train')
        
    # Shuffle and select subset
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    formatted_data = []
    print("Formatting prompts...")
    for item in tqdm(dataset):
        question = item['question']
        answer = item['answer']
        
        # HotpotQA context is a dict with 'title' and 'sentences'
        context_docs = []
        titles = item['context']['title']
        sentences_lists = item['context']['sentences']
        for title, sentences in zip(titles, sentences_lists):
            doc_text = f"{title}: " + " ".join(sentences)
            context_docs.append(doc_text)
            
        # Limit to 5 docs for context length
        context_docs = context_docs[:5]
        
        prompt = format_prompt(question, context_docs)
        
        # Occasionally simulate "no evidence" to teach the model to refuse hallucination
        if random.random() < 0.15:
            prompt = format_prompt(question, ["This document is completely unrelated and discusses photosynthesis."])
            answer = "I don't have enough verified evidence to answer this question."
            
        formatted_data.append({
            "instruction": prompt,
            "output": answer
        })
        
    # Split train/val
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")

if __name__ == "__main__":
    generate_dataset()
