import pandas as pd
import pyterrier as pt
import re

if not pt.started():
    # Initialize PyTerrier with 8GB memory allocated
    pt.init(mem=8192)

index_path = './index'
index = pt.IndexFactory.of(index_path)
print('Using index at "%s"' % index_path)

queries_path = './data/queries_train.csv'
topics = pd.read_csv(queries_path, usecols=['qid', 'query'])
# Remove special characters
topics['query'] = topics['query'].apply(lambda x: re.sub(r'\W', ' ', x).lower())
print('Read %s queries from "%s"' % (len(topics), queries_path))

sdm = pt.rewrite.SequentialDependence()
bm25 = pt.BatchRetrieve(index, num_results=1000, verbose=True, wmodel='BM25')
pipe = sdm >> bm25
results = pipe.transform(topics)

results_path = './results.trec'
pt.io.write_results(results, results_path, format='trec', run_name='PM25')
print('Retrieval results written to "%s"' % results_path)

qrels = pd.read_csv('./data/qrels_train.txt', sep=' ', names=['qid', 'iter', 'docno', 'label'])

# Ensure same datatypes
results['docno'] = results['docno'].astype(str)
qrels['docno'] = qrels['docno'].astype(str)

eval = pt.Utils.evaluate(results, qrels, metrics=['recall.1000', 'map', 'recip_rank', 'ndcg_cut_3'])
print(eval)