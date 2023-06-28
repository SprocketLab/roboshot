conda create -n roboshot -y python=3.7
conda activate roboshot
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install wilds
pip install transformers
pip install openai
pip install tqdm
pip install numpy