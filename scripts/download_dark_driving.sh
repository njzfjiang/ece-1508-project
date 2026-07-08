mkdir -p data
wget https://huggingface.co/datasets/chkaty/day2night-data/resolve/main/processed.zip -O data/processed.zip
cd data
unzip processed.zip
rm processed.zip