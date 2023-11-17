import argparse
import confuse
import ir_measures
import pandas as pd
import pytz
import re
from collections import defaultdict
from datetime import datetime
from elasticsearch import Elasticsearch
from ir_measures import *
from operator import itemgetter
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, TypedDict, Tuple, Union

from core.utils import load_queries, load_qrels, write_to_trec
from reranker.reranker import run_reranker, fusion
from retriever.retriever import get_passages
from rewriter.rewriter import rewrite_queries
from term_selector.term_selector import term_selector


CONFIG_PATH = 'config.yaml'
QUERIES_PATH = 'data/queries_train.csv'
QUERIES_TEST_PATH = 'data/queries_test.csv'
QRELS_PATH = 'data/qrels_train.txt'
INDEX_NAME = 'ms_marco'


def get_es():
    return Elasticsearch()


def main(config):
    train = False
    if config['train'].get(bool):
        train = True
    if config['tz'].get():
        tz = pytz.timezone(config['tz'].get())
    run(
        train=train,
        tz=tz
    )


def run(
    train: bool,
    tz: pytz.timezone
) -> None:

    # Load queries and QRELS
    if train:
        input_queries = load_queries(QUERIES_PATH)
        qrels = load_qrels(QRELS_PATH)
    else:
        input_queries = load_queries(QUERIES_TEST_PATH)

    # Run MVR
    run_mvr(
        es=get_es(),
        queries=input_queries,
        tz=tz,
        train=train,
        qrels=qrels if train else None
    )

    return


def run_mvr(
    es: Elasticsearch,
    queries: Dict[str, str],
    tz: pytz.timezone,
    metrics: List[ir_measures] = [R(rel=2)@1000, nDCG@3, AP(rel=2), RR(rel=2)],
    train: bool = False,
    qrels: Union[Dict['str', Dict['str', 'int']], None] = None
):
    stage = 'TRAIN' if train else 'TEST'

    ##########################################################################
    # STEP 1
    # Input queries + CTS + BM25 Elasticsearch retrieval
    ##########################################################################
    cts_terms = term_selector(list(queries.values()))
    queries_cts = rewrite_queries(queries, cts_terms, n_previous_terms=3)
    first_pass_rankings, docs = get_passages(es, queries_cts, index=INDEX_NAME, k=1000)

    # Write rankings to file
    timestamp = datetime.now(tz).isoformat(timespec='seconds')
    filepath_out_trec = f'results/{timestamp}-BM25-first-pass-rankings-{stage}.trec'
    write_to_trec(filepath_out_trec, first_pass_rankings, train=train)

    # Print measures
    if train:
        measures = ir_measures.calc_aggregate(
            metrics,
            qrels,
            ir_measures.read_trec_run(filepath_out_trec)
        )
        print('First pass retrieval measures:')
        pprint(measures)

    ##########################################################################
    # STEP 2
    # Reranking queries+CTS with first-pass passages
    ##########################################################################
    mvr_1_rankings = run_reranker(queries_cts, first_pass_rankings, docs)

    # Write reranking results to file
    timestamp = datetime.now(tz).isoformat(timespec='seconds')
    filepath_out_trec = f'results/{timestamp}-MVR1-reranked-{stage}.trec'
    write_to_trec(filepath_out_trec, mvr_1_rankings, train=train)

    # Print measures
    if train:
        measures = ir_measures.calc_aggregate(
            metrics,
            qrels,
            ir_measures.read_trec_run(filepath_out_trec)
        )
        print('MVR1 reranking measures:')
        pprint(measures)

    ##########################################################################
    # STEP 3
    # Reranking based on queries + CTS terms from some of the passages
    ##########################################################################
    top_k_docs = 2
    docs_to_cts = []
    for qid, passages in mvr_1_rankings.items():
        for i, (docid, _) in enumerate(passages.items()):
            if i >= top_k_docs:
                break
            docs_to_cts.append(docs[docid])
    doc_cts = term_selector(docs_to_cts)

    merged_cts = []
    for i in range(0, len(doc_cts), top_k_docs):
        merged_cts.append(
            ' '.join(set(d for sublist in doc_cts[i:i+top_k_docs] for d in sublist)))

    queries_cts = rewrite_queries(queries, merged_cts)
    mvr_2_rankings = run_reranker(queries_cts, mvr_1_rankings, docs)

    # Write reranking results to file
    timestamp = datetime.now(tz).isoformat(timespec='seconds')
    filepath_out_trec = f'results/{timestamp}-MVR2-reranked-{stage}.trec'
    write_to_trec(filepath_out_trec, mvr_2_rankings, train=train)

    # Print measures
    if train:
        measures = ir_measures.calc_aggregate(
            metrics,
            qrels,
            ir_measures.read_trec_run(filepath_out_trec)
        )
        print('MVR2 reranking measures:')
        pprint(measures)

    ##########################################################################
    # STEP 4
    # Reranking based on query reformulation (GPT2)
    ##########################################################################
    # TODO

    ##########################################################################
    # STEP 5
    # Fuse results (simple addition of scores) and sort
    ##########################################################################
    mvr_rankings = fusion([mvr_1_rankings, mvr_2_rankings])

    # Write reranking results to file
    timestamp = datetime.now(tz).isoformat(timespec='seconds')
    filepath_out_trec = f'results/{timestamp}-MVR-reranked-fused-{stage}.trec'
    write_to_trec(filepath_out_trec, mvr_rankings, train=train)

    # Print measures
    if train:
        measures = ir_measures.calc_aggregate(
            metrics,
            qrels,
            ir_measures.read_trec_run(filepath_out_trec)
        )
        print('MVR fused reranked measures:')
        pprint(measures)


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    config = confuse.Configuration('dat640')
    config.set_file(CONFIG_PATH)
    config.set_args(args, dots=True)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='main.py')

    parser.add_argument(
        '-t',
        '--train',
        action='store_const',
        const=True,
        help='Score the train dataset. Defaults to False.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    main(config)
