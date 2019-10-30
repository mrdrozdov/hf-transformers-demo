```
wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz
tar xzf spanbert_hf_base.tar.gz
mkdir save-bert
mkdir save-spanbert

python demo-bert.py
cp save-bert/vocab.txt ./spanbert_hf_base/
python demo.py # will load spanbert!
```
