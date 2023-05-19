# music_similarity


#TODO

1. download: 

`

cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

`

2. unzip

`

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -

echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

unzip fma_metadata.zip

unzip fma_small.zip

`

3. view the data - https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb

5. LSH: https://pypi.org/project/lshashing/

7. ANN: https://github.com/facebookresearch/faiss

