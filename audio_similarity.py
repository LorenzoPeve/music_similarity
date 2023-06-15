from pathlib import Path
import typing as t
from functools import cache

import numpy as np
from datasets import load_dataset
from panns_inference import AudioTagging
from tqdm.auto import tqdm
from annoy import AnnoyIndex
import typer

app = typer.Typer()

@cache
def load_model():
    # load the default model into the gpu.
    # change device to cpu if a gpu is not available
    model = AudioTagging(checkpoint_path=None, device='cuda')
    return model

@cache
def load_esc50_dataset():
    # load the dataset from huggingface model hub
    return load_dataset("ashraq/esc50", split="train")

@app.command()
def ensure_index():
    ndim = 2048

    if Path("./esc50-audio.ann").exists():
        annoy_index = AnnoyIndex(ndim, "angular")
        annoy_index.load("esc50-audio.ann")
    else:
        annoy_index = AnnoyIndex(ndim, "angular")
        annoy_index.on_disk_build("esc50-audio.ann")

        model = load_model()
        esc50_dataset = load_esc50_dataset()
        # select only the audio data from the dataset and store in a numpy array
        audios = np.array([a["array"] for a in esc50_dataset["audio"]])

        # we will use batches of 64
        batch_size = 64

        with tqdm(total=len(audios)) as pbar:
            for i in range(0, len(audios), batch_size):
                # find end of batch
                i_end = min(i+batch_size, len(audios))
                # extract batch
                batch = audios[i:i_end]
                # generate embeddings for all the audios in the batch
                _, emb = model.inference(batch)
                # create unique IDs
                ids = list(range(i, i_end))
                # upsert/insert these records to annoy
                for id, embedding in zip(ids, emb.tolist()):
                    annoy_index.add_item(id, embedding)
                    pbar.update()
        annoy_index.build(10)
    return annoy_index

def get_audio(audio_num: int) -> t.Tuple[np.array, str]:
    data = load_esc50_dataset()
    # get the audio data of the audio number
    query_audio = data[audio_num]["audio"]["array"]
    # get the category of the audio number
    category = data[audio_num]["category"]
    # print the category and play the audio
    return query_audio, category

def get_audio_embedding(audio_num: int):
    model = load_model()
    data = load_esc50_dataset()
    # get the audio data of the audio number
    query_audio = data[audio_num]["audio"]["array"]
    # reshape query audio
    query_audio = query_audio[None, :]
    # get the embeddings for the audio from the model
    _, embedding = model.inference(query_audio)
    return embedding.reshape((-1,))

@app.command()
def find_neighbors(audio_num: int) -> t.List[int]:
    annoy_index = ensure_index()
    neigbors = annoy_index.get_nns_by_vector(get_audio_embedding(400), 3)
    print(neigbors)
    return neigbors

if __name__ == "__main__":
    app()
