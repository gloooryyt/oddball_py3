# oddball_py3
Python3 implementation of oddball

This is the Python3 implementation of oddball.  
For more detail, you can see: Akoglu L., McGlohon M., Faloutsos C.(2010) oddBall: Spotting Anomalies in Weighted Graphs.

## Environments
networkx (version: 2.1)  
numpy  
scikit_learn  
You can use the following command to download the environments directly.
```
pip install -r requirements.txt
```

## Run
The input is a weighted undirected graph which format is 'edge1 edge2 weight'.  
### Options:  
  --input: input file  
  --output: output file  
  --lof: Use LOF. 0: not use. 1: use. Default value is 0.  
  --anomaly_type: Anomaly Type. 1:star_or_clique. 2:heavy_vicinity. 3:dominant_edge.

You can use --help for more details.  
Here is a sample.
```
python main.py --input inputFile --output outputFile --lof 0 --anomaly_type 1
```

### Miscellaneous
Hope you will like this repository and have fun.  
Don't forget a star, lol~
