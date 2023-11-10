## Server for "Manipulated Interpolation of Anything"
We need to setup a server to serve segmenting tasks

### Download checkpoint
```
cd data/models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ../..
```

### Start Server
```
python app.py
```