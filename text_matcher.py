import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
from pathlib import Path
import re

EMBEDDINGS_DIR = Path('embeddings_cache')
VACANCY_EMB_FILE = EMBEDDINGS_DIR / 'vacancy_embeddings_bge_m3.pkl'
RESUME_EMB_FILE = EMBEDDINGS_DIR / 'resume_embeddings_bge_m3.pkl'
MODEL_NAME = 'BAAI/bge-m3'
MIN_WORD_LENGTH = 3
TEXT_SEARCH_TOP_K = 5

def extract_resume_title(row):
    try:
        title = str(row['Ищет работу на должность:'])
        if pd.notna(title) and title.strip() != '' and title.lower() != 'nan':
            return title.strip()
        return None
    except:
        return None

def extract_search_words(title):
    title_lower = title.lower()
    title_clean = re.sub(r'[^\w\s]', ' ', title_lower)
    words = title_clean.split()
    
    stop_words = {'и', 'в', 'на', 'по', 'для', 'от', 'до', 'из', 'с', 
                  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in'}
    
    meaningful_words = [
        word for word in words 
        if len(word) >= MIN_WORD_LENGTH and word not in stop_words
    ]
    
    return meaningful_words

def check_text_match(vacancy_title, search_words):
    vacancy_lower = vacancy_title.lower()
    matched_words = [word for word in search_words if word in vacancy_lower]
    return matched_words

def find_text_matches_with_scores(vacancies_df, search_words, top_k=5):
    matches = []
    
    for idx, row in vacancies_df.iterrows():
        vacancy_title = row['vacancy']
        matched_words = check_text_match(vacancy_title, search_words)
        
        if matched_words:
            text_score = len(matched_words) / len(search_words) if search_words else 0
            matches.append({
                'index': idx,
                'matched_words': matched_words,
                'text_score': text_score,
                'match_count': len(matched_words)
            })
    
    matches.sort(key=lambda x: (x['match_count'], x['text_score']), reverse=True)
    return matches[:top_k]

def load_or_create_embeddings(texts, model, cache_file):
    if cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print(f"Creating embeddings: {cache_file.name}")
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        EMBEDDINGS_DIR.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

def print_vacancy_details(vacancy_row, bert_score=None):
    print(f"\nVacancy: {vacancy_row['vacancy']}")
    print(f"Company: {vacancy_row['employer']}")
    print(f"Location: {vacancy_row['area']}")
    
    salary_parts = []
    if pd.notna(vacancy_row['salary_from']):
        salary_parts.append(f"from {int(vacancy_row['salary_from']):,}")
    if pd.notna(vacancy_row['salary_to']):
        salary_parts.append(f"to {int(vacancy_row['salary_to']):,}")
    
    if salary_parts:
        salary_str = " ".join(salary_parts)
        if pd.notna(vacancy_row['currency']):
            salary_str += f" {vacancy_row['currency']}"
        print(f"Salary: {salary_str}")
    
    print(f"Experience: {vacancy_row['experience']}")
    print(f"Schedule: {vacancy_row['schedule']}")
    
    if bert_score is not None:
        score_value = bert_score.item() if torch.is_tensor(bert_score) else bert_score
        print(f"BERT score: {score_value:.4f}")
    
    print(f"URL: {vacancy_row['url']}")

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on {device}")
    
    print("\nLoading data...")
    resumes_df = pd.read_csv(
        'resumes.csv',
        sep=';',
        on_bad_lines='skip',
        encoding='utf-8',
        low_memory=False
    )
    
    vacancies_df = pd.read_csv(
        'vacancies.csv',
        on_bad_lines='skip',
        encoding='utf-8',
        low_memory=False
    )
    
    print(f"Resumes: {len(resumes_df)}")
    print(f"Vacancies: {len(vacancies_df)}")
    
    resumes_df['resume_title'] = resumes_df.apply(extract_resume_title, axis=1)
    resumes_df = resumes_df[resumes_df['resume_title'].notna()].copy()
    resumes_df = resumes_df[resumes_df['resume_title'].str.strip() != '']
    resumes_df = resumes_df.reset_index(drop=True)
    
    vacancies_df = vacancies_df[vacancies_df['vacancy'].notna()].copy()
    vacancies_df = vacancies_df[vacancies_df['vacancy'].str.strip() != '']
    vacancies_df = vacancies_df.reset_index(drop=True)
    
    print(f"After filtering: {len(resumes_df)} resumes, {len(vacancies_df)} vacancies")
    
    resume_titles = resumes_df['resume_title'].tolist()
    vacancy_titles = vacancies_df['vacancy'].tolist()
    
    vacancy_embeddings = load_or_create_embeddings(vacancy_titles, model, VACANCY_EMB_FILE)
    resume_embeddings = load_or_create_embeddings(resume_titles, model, RESUME_EMB_FILE)
    
    print("\nReady")
    
    while True:
        print("\n" + "="*80)
        print("Example resumes:")
        for i in range(min(20, len(resumes_df))):
            print(f"[{i:4d}] {resume_titles[i]}")
        print(f"Total: {len(resumes_df)} resumes")
        
        try:
            user_input = input("\nEnter resume number (q to quit): ").strip()
            
            if user_input.lower() == 'q':
                break
            
            resume_idx = int(user_input)
            
            if resume_idx < 0 or resume_idx >= len(resumes_df):
                print(f"Error: number must be from 0 to {len(resumes_df)-1}")
                continue
            
            resume_title = resume_titles[resume_idx]
            search_words = extract_search_words(resume_title)
            
            print("\n" + "="*80)
            print(f"Resume #{resume_idx}: {resume_title}")
            print(f"Keywords: {', '.join(search_words)}")
            
            print(f"\nStep 1: Text search (top {TEXT_SEARCH_TOP_K})...")
            text_matches = find_text_matches_with_scores(vacancies_df, search_words, top_k=TEXT_SEARCH_TOP_K)
            
            if not text_matches:
                print("No vacancies with text matches found")
                continue
            
            print(f"Found {len(text_matches)} vacancies with text matches")
            
            resume_embedding = resume_embeddings[resume_idx].unsqueeze(0)
            
            candidates_with_bert = []
            for rank, match in enumerate(text_matches, 1):
                vacancy_idx = match['index']
                vacancy_row = vacancies_df.iloc[vacancy_idx]
                vacancy_embedding = vacancy_embeddings[vacancy_idx].unsqueeze(0)
                bert_score = util.cos_sim(resume_embedding, vacancy_embedding)[0][0]
                
                candidates_with_bert.append({
                    'rank': rank,
                    'index': vacancy_idx,
                    'row': vacancy_row,
                    'bert_score': bert_score
                })
            
            print(f"\nStep 2: BERT re-ranking...")
            best_candidate = max(candidates_with_bert, key=lambda x: x['bert_score'])
            
            print("\nBERT scores:")
            for cand in sorted(candidates_with_bert, key=lambda x: x['bert_score'], reverse=True):
                score_val = cand['bert_score'].item()
                marker = " <- BEST" if cand == best_candidate else ""
                print(f"Candidate #{cand['rank']}: {score_val:.4f}{marker}")
            
            print("\n" + "="*80)
            print("BEST VACANCY:")
            print_vacancy_details(best_candidate['row'], bert_score=best_candidate['bert_score'])
            print("="*80)
            
            continue_input = input("\nSearch more? (Enter - yes, q - quit): ").strip()
            if continue_input.lower() == 'q':
                break
            
        except ValueError:
            print("Error: enter a number")
        except KeyboardInterrupt:
            print("\nExiting")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()