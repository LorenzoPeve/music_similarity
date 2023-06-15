# ATX Practical Datascience: Audio Similarity Search

_Adapted from https://docs.pinecone.io/docs/audio-search_

**Installs:**
```bash
conda install -c huggingface -c conda-forge datasets numpy tqdm python-annoy typer
pip install panns-inference
```

## Notebook
See `audio-similarity.ipynb` for the code to run.

## CLI

```bash
python audio_similarity.py ensure-index
python audio_similarity.py find-neighbors 400
```
