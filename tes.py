import colbert
import torch
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from datasets import load_dataset


if __name__=='__main__':
    dataset = 'lifestyle'
    datasplit = 'dev'

    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    print(queries[24])
    print()
    print(collection[19929])
    print()


    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300 # truncate passages at 300 tokens
    max_id = 10000

    index_name = f'{dataset}.{datasplit}.{nbits}bits'

    answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
    filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]

    f'Filtered down to {len(filtered_queries)} queries'

    checkpoint = 'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)

    indexer.get_index() # You can get the absolute path of the index, if needed.