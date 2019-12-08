"""
This is a simple application for sentence embeddings: semantic search
We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.
This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="input_filename", required=True,
                    help="corpus input file", metavar="FILE")
parser.add_argument("-q", "--query", dest="query_filename", required=True,
                    help="corpus input file", metavar="FILE")
                    
# parser.add_argument("-q", "--query",dest="query", nargs='+', required=True,
#                     help="query")
args = parser.parse_args()

with open(args.input_filename, 'r', encoding='utf-8') as f:
    corpus = f.read().split('\n')
with open(args.query_filename, 'r', encoding='utf-8') as f:
    queries = f.read().split('\n')
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

corpus_embeddings = embedder.encode(corpus)

# Query sentences:
query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))