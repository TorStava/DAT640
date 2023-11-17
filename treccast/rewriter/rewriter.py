import spacy
from collections import defaultdict
from typing import List, Dict

nlp = spacy.load('en_core_web_sm')


def rewrite_queries(
        queries: Dict[str, str],
        conversational_terms: List[List[str]],
        topic: str = None,
        n_previous_terms: int = 3
) -> List[Dict[str, str]]:

    if len(queries) != len(conversational_terms):
        raise ValueError('Queries and terms must be the same length.')

    rewritten_queries = defaultdict(str)
    current_topic_number = None
    
    for i, (qid, query) in enumerate(queries.items()):
        
        topic_number, turn_number = qid.split('_')
        
        if not current_topic_number == topic_number:
            current_topic_number = topic_number
            current_topic = conversational_terms[i]
            topic_start_idx = i

        add_terms = None
        if i > topic_start_idx:
            add_terms = current_topic[0] if current_topic else ''
            if i - topic_start_idx <= n_previous_terms:
                term_start_idx = topic_start_idx
            else:
                term_start_idx = i - n_previous_terms
            add_terms += ' ' + ' '.join([t for term in conversational_terms[term_start_idx:i] for t in term])
            add_terms = ' '.join(set([t for t in add_terms.split()]))

        rewritten_queries[qid] = ' '.join([query, add_terms]) if add_terms else query
    return rewritten_queries


def rewrite_queries_spacy(queries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rewritten_queries = defaultdict(str)
    current_topic = None

    for qid, query in queries.items():
        if not qid.split('_')[0] == current_topic:
            current_topic = qid.split('_')[0]
            rewritten_query = ''
        conversational_terms = []
        if len(rewritten_query) > 0:
            doc = nlp(rewritten_query)

            conversational_terms = [
                token.text for token in doc if token.tag_ in [
                    'NNP', 'NN', 'JJ', 'NNS'] or token.ent_type_ in [
                    'ORG', 'ORDINAL', 'NORP'
                ]
            ]

        rewritten_query = ' '.join([query] + list(set(conversational_terms)))
        rewritten_queries[qid] = rewritten_query

    return rewritten_queries

