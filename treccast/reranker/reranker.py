import pandas as pd
from regex import P
import torch
import torch.nn.functional as F
from collections import defaultdict
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict, Tuple


def run_reranker(
    queries: Dict[str, str],
    first_pass_rankings: Dict[str, Dict[str, float]],
    docs: Dict[str, str],
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'  # Fusion nDCG@3 = 0.222 SLOW
) -> Dict[str, Dict[str, float]]:

    model = CrossEncoder(model_name, max_length=512)
    
    rerankings = defaultdict(dict)
    for qid, query in tqdm(queries.items(), desc='Reranking'):

        docids = [docid for (docid, _) in first_pass_rankings[qid].items()]
        passages = [docs[docid] for docid in docids]
        queries_list = [[query, passage] for passage in passages]

        scores = model.predict(queries_list)

        # Combine docids and scores
        doc_score_pairs = list(
            zip(docids, scores.tolist()))

        # Sort by decreasing score
        doc_score_pairs = sorted(
            doc_score_pairs, key=lambda x: x[1], reverse=True)

        rerankings[qid] = {k: v for (k, v) in doc_score_pairs}

    return rerankings


def fusion(rankings: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:

    mvr_scores = defaultdict(lambda: defaultdict(float))
    for ranking in rankings:
        for qid, docs in ranking.items():
            for docid, score in docs.items():
                if mvr_scores[qid][docid]:
                    mvr_scores[qid][docid] += score
                else:
                    mvr_scores[qid][docid] = score

    # Sort accoring to descending score
    mvr_reranked = defaultdict(dict)
    for qid, docs in mvr_scores.items():
        doc_score_pairs = sorted(
            docs.items(), key=lambda x: x[1], reverse=True)
        mvr_reranked[qid] = {k: v for (k, v) in doc_score_pairs}

    return mvr_reranked
