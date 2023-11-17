import spacy
from collections import defaultdict
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from transformers.pipelines import AggregationStrategy
import numpy as np

from typing import List

nlp = spacy.load('en_core_web_sm')


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results]).tolist()


def term_selector(
        docs: List[str]
) -> List[str]:
    model_name = "ml6team/keyphrase-extraction-kbir-inspec"

    extractor = KeyphraseExtractionPipeline(model=model_name, device=0)
    keyphrases = extractor(docs)
    return keyphrases


def term_selector_spacy(docs: List[str]) -> List[str]:

    terms = []
    for doc in docs:
        tokens = nlp(doc)
        conversational_terms = [
            token.text for token in tokens if token.tag_ in [
                'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS', 'RBR', 'RBS', 'VBD'] or token.ent_type_ in [
                'ORG', 'ORDINAL', 'NORP'
            ]
        ]
        terms.append(conversational_terms)

    return terms