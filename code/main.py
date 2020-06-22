'''
Python3 implementation of oddball

@author:
Tao Yu (gloooryyt@gmail.com)

'''

import numpy as np
import networkx as nx
from loadData import *
from oddball import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Oddball.')
    parser.add_argument('--input', type=str, required=True, help='Path of input file.')
    parser.add_argument('--output', type=str, required=True, help='Path of output file.')
    parser.add_argument('--lof', type=int, default=0, help='Use LOF. 0: not use. 1: use. Default value is 0.')
    parser.add_argument('--anomaly_type', type=int, required=True, help='Anomaly Type. 1:star_or_clique. 2:heavy_vicinity. 3:dominant_edge.')
    args = parser.parse_args()

    #Input a weighted undirected graph which format is 'a b weight'
    input_path = args.input
    output_path = args.output
    G = load_data(input_path)
    featureDict = get_feature(G)
    if args.lof == 0:
        if args.anomaly_type == 1:
            star_or_clique_score = star_or_clique(featureDict)
            star_or_clique_array = []
            for key in star_or_clique_score.keys():
                star_or_clique_array.append(np.array([key, star_or_clique_score[key]]))
            star_or_clique_array = np.array(star_or_clique_array)
            star_or_clique_array = star_or_clique_array[star_or_clique_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in star_or_clique_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        elif args.anomaly_type == 2:
            heavy_vicinity_score = heavy_vicinity(featureDict)
            heavy_vicinity_array = []
            for key in heavy_vicinity_score.keys():
                heavy_vicinity_array.append(np.array([key, heavy_vicinity_score[key]]))
            heavy_vicinity_array = np.array(heavy_vicinity_array)
            heavy_vicinity_array = heavy_vicinity_array[heavy_vicinity_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in heavy_vicinity_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        elif args.anomaly_type == 3:
            dominant_edge_score = dominant_edge(featureDict)
            dominant_edge_array = []
            for key in dominant_edge_score.keys():
                dominant_edge_array.append(np.array([key, dominant_edge_score[key]]))
            dominant_edge_array = np.array(dominant_edge_array)
            dominant_edge_array = dominant_edge_array[dominant_edge_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in dominant_edge_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        else:
            print('parameter error!')
    elif args.lof == 1:
        if args.anomaly_type == 1:
            star_or_clique_withLOF_score = star_or_clique_withLOF(featureDict)
            star_or_clique_withLOF_array = []
            for key in star_or_clique_withLOF_score.keys():
                star_or_clique_withLOF_array.append(np.array([key, star_or_clique_withLOF_score[key]]))
            star_or_clique_withLOF_array = np.array(star_or_clique_withLOF_array)
            star_or_clique_withLOF_array = star_or_clique_withLOF_array[star_or_clique_withLOF_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in star_or_clique_withLOF_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        elif args.anomaly_type == 2:
            heavy_vicinity_withLOF_score = heavy_vicinity_withLOF(featureDict)
            heavy_vicinity_withLOF_array = []
            for key in heavy_vicinity_withLOF_score.keys():
                heavy_vicinity_withLOF_array.append(np.array([key, heavy_vicinity_withLOF_score[key]]))
            heavy_vicinity_withLOF_array = np.array(heavy_vicinity_withLOF_array)
            heavy_vicinity_withLOF_array = heavy_vicinity_withLOF_array[heavy_vicinity_withLOF_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in heavy_vicinity_withLOF_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        elif args.anomaly_type == 3:
            dominant_edge_withLOF_score = dominant_edge_withLOF(featureDict)
            dominant_edge_withLOF_array = []
            for key in dominant_edge_withLOF_score.keys():
                dominant_edge_withLOF_array.append(np.array([key, dominant_edge_withLOF_score[key]]))
            dominant_edge_withLOF_array = np.array(dominant_edge_withLOF_array)
            dominant_edge_withLOF_array = dominant_edge_withLOF_array[dominant_edge_withLOF_array[:, 1].argsort()[::-1]]  # Sort by score from large to small
            with open(output_path, 'w') as f:
                for key in dominant_edge_withLOF_array:
                    f.write(str(int(key[0])) + ' ' + str(key[1]))
                    f.write('\n')
            f.close()
        else:
            print('parameter error!')
    else:
        print('parameter error!')

if __name__ == "__main__":
    main()