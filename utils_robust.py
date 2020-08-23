import time
import pickle
import logging
import argparse

import sys
import os
import numpy as np
import pandas as pd
import functools
import seaborn as sns

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ProcessPool 

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Pick up local packages
sys.path.append('..')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

# Get rid of RuntimeWarnings, gross
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import trackml.dataset

sys.path.append('/global/homes/c/caditi97/exatrkx-ctd2020/MetricLearning/src/preprocess_with_dir/')
# from extract_dir import *
from preprocess import get_one_event, load_detector

# Pick up local packages
sys.path.append('..')
sys.path.append('/global/homes/c/caditi97/exatrkx-ctd2020/MetricLearning/src/metric_learning_adjacent/')

# Local imports
from build_graphs import *
from GraphLearning.src.trainers import get_trainer
from utils.data_utils import (get_output_dirs, load_config_file, load_config_dir, load_summaries,
                      save_train_history, get_test_data_loader,
                      compute_metrics, save_metrics, draw_sample_xy)

from build_graphs import *
from tqdm import tqdm
import statistics


feature_names = ['x', 'y', 'z', 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']



def get_data(noise_keep):
    artifact_storage_path = "/global/cfs/projectdirs/m3443/usr/dtmurnane/artifacts/adjacent/"
    #"/global/cfs/projectdirs/m3443/usr/aoka/data/artifacts/Training_Example_no_ptcut"
    best_emb_path = os.path.join(artifact_storage_path, 'metric_learning_emb', 'best_model.pkl')
    best_filter_path = os.path.join(artifact_storage_path, 'metric_learning_filter', 'best_model.pkl')    

    emb_model = load_embed_model(best_emb_path, DEVICE).to(DEVICE)
    filter_model = load_filter_model(best_filter_path, DEVICE).to(DEVICE)
    emb_model.eval()
    filter_model.eval()
    
    event_name = "event000001000.pickle"
    data_path = f"/global/cfs/cdirs/m3443/usr/aoka/data/classify/Classify_Example_{noise_keep}/preprocess_raw"
    hits, truth = load_event(data_path, event_name)
    print("noise:", noise_keep, "number of hits:", len(hits))
    return hits, truth, emb_model, filter_model


def plot_noise(noise_hits,noise_truth,noise_keep,index):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(noise_hits.shape)
    print("truth")
    print(noise_truth.shape)

    unique_ids = noise_truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = noise_truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = noise_hits[where_to_keep]
    noise = noise_hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))

    g3 = sns.jointplot(not_noise.x, not_noise.y, s=2, height=12, label = "not noise")
    g3.x = noise.x
    g3.y = noise.y
    g3.plot_joint(plt.scatter, c='r', s=1, label = "noise")



    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.title('Noise Distribution')
    plt.savefig('noise_[' +str(index)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
def plot_neighborhood(hits, truth, neighbors, noise_keep, k=None):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)
    
    hitidx = neighbors[k]
    hitids = hits.iloc[hitidx]['hit_id'].values
    print("len(neighbors[k]) = " +str(len(hitids)))
    sel_hits = hits[hits['hit_id'].isin(hitids)]
    # hits in a neighborhood
    print("Hits in the Neighborhood = " + str(len(sel_hits)))
    diff_n = len(hits) - len(sel_hits)
    print("Hits outside the Neighborhood = " + str(diff_n))
    g = sns.jointplot(sel_hits.x, sel_hits.y, s = 5, height = 12, label ='neighborhood')
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
#     noise_idx = truth[truth['particle_id'] == 0]
#     noise_ids = noise_idx[noise_idx['hit_id'].isin(hitids)]
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    g.x = noise_in.x
    g.y = noise_in.y
    g.plot_joint(plt.scatter, c = 'r', s=5, label='noise in neighborhood')
    print("Noise in Neighborhood = " + str(len(noise_in)))
#     diff = len(noise) - len(noise_in)
#     print("Noise outside Neibhorhood = " + str(diff))
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.savefig('neighbor[' +str(k)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    

def plot_allhits_with_neighborhood(hits, truth, neighbors, noise_keep, k):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)

    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = hits[where_to_keep]
    noise = hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))
    
    #noise vs not noise
    g = sns.jointplot(not_noise.x, not_noise.y, s=1, height=20, label = "not noise")
    g.x = noise.x
    g.y = noise.y
    g.plot_joint(plt.scatter, c='r', s=1, label = "noise")
    
    # vs neighborhood
    hitidx = neighbors[k]
    hitids = hits.iloc[hitidx]['hit_id'].values
    print("len(neighbors[k]) = " +str(len(hitids)))
    # hits in a neighborhood
    sel_hits = hits[hits['hit_id'].isin(hitids)]
    print("Hits in the Neighborhood = " + str(len(sel_hits)))
    diff_h = len(hits) - len(sel_hits)
    print("Hits outside the Neighborhood = " + str(diff_h))
    g.x = sel_hits.x
    g.y = sel_hits.y
    g.plot_joint(plt.scatter, c = 'k', s=2, label='neighborhood')
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    g.x = noise_in.x
    g.y = noise_in.y
    g.plot_joint(plt.scatter, c = 'y', s=3, label='noise in neighborhood')
    print("Noise in Neighborhood = " + str(len(noise_in)))
    diff_n = len(noise) - len(noise_in)
    print("Noise outside Neibhorhood = " + str(diff_n))
    
    if(len(noise) == 0):
        in_hits = len(sel_hits)/len(hits)
        out_hits = diff_h/len(hits)
        in_noise = 0
        out_noise = 0
    else:
        in_hits = len(sel_hits)/len(hits)
        out_hits = diff_h/len(hits)
        in_noise = len(noise_in)/len(noise)
        out_noise = diff_n/len(hits)
        
    
    
    print("----------------")
    print("% Hits inside = " +str(in_hits))
    print("% Hits outside = " +str(out_hits))
    print("% Noise inside = " +str(in_noise))
    print("% Noise outside = " +str(out_noise))
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.savefig('all_neighbor[' +str(k)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
    return in_hits, out_hits, in_noise, out_noise

def get_one_neighborhood(hits, truth, neighbors, index):
    vol = hits[['volume_id', 'layer_id']].values.T
#     print(vol)
    hit = hits.iloc[k]
#     print(hit)
    pid = truth[truth['hit_id'] == hit['hit_id']]['particle_id']
#     print(pid)
#     thit = truth[truth['hit_id'] == hit['hit_id']]['hit_id'] 
#     print(thit)
#     hitid = hit['hit_id']
#     print(hitid)
    print("Hit Number = " +str(index)+ " Particle ID = " +str(pid)+ " Hit ID = " +str(hit['hit_id']))
    one_n = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], neighbors[index], vol[0], vol[1])
    return one_n

def plot_one_neighborhood(hits, truth, neighbors, noise_keep):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)
    
    hitidx = neighbors
    hitids = hits.iloc[hitidx]['hit_id'].values
    print("len(neighbors[k]) = " +str(len(hitids)))
    sel_hits = hits[hits['hit_id'].isin(hitids)]
    # hits in a neighborhood
    print("Hits in the Neighborhood = " + str(len(sel_hits)))
    diff_n = len(hits) - len(sel_hits)
    print("Hits outside the Neighborhood = " + str(diff_n))
    g = sns.jointplot(sel_hits.x, sel_hits.y, s = 5, height = 12, label ='neighborhood')
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
#     noise_idx = truth[truth['particle_id'] == 0]
#     noise_ids = noise_idx[noise_idx['hit_id'].isin(hitids)]
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    g.x = noise_in.x
    g.y = noise_in.y
    g.plot_joint(plt.scatter, c = 'r', s=5, label='noise in neighborhood')
    print("Noise in Neighborhood = " + str(len(noise_in)))
#     diff = len(noise) - len(noise_in)
#     print("Noise outside Neibhorhood = " + str(diff))
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.savefig('one_n_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
def plots(hits, truth, noise_keep, feature_names, index, emb_model, radius = 0.4):
    neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)
    print("Total Neighborhoods/Hits = " + str(len(neighbors)))
    print("Chosen neighborhood/Hit = " + str(index))
#     print(neighbors[index])
    
    plot_noise(hits,truth,noise_keep, index)
    
    in_hits, out_hits, in_noise, out_noise = plot_allhits_with_neighborhood(hits, truth, neighbors, noise_keep, index)
    
    plot_neighborhood(hits,truth, neighbors, noise_keep, index)
    
    return in_hits, out_hits, in_noise, out_noise
    
def overall(index):
    feature_names = ['x', 'y', 'z', 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    noise_keeps = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    
    in_hits =[]
    out_hits =[]
    in_noise=[]
    out_noise =[]
    
    for noise_keep in noise_keeps:
        hits, truth, emb_model, filter_model = get_data(noise_keep)
        in_h, out_h, in_n, out_n = plots(hits, truth, noise_keep, feature_names, index, emb_model, radius=0.4)
        in_hits.append(in_h)
        out_hits.append(out_h)
        in_noise.append(in_n)
        out_noise.append(out_n)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    ax1.plot(x, in_hits)
    ax1.set_title("% Hits inside Neighborhood")
    ax1.set_xlabel("noise_keep")
    ax2.plot(x, out_hits)
    ax2.set_title("% Hits outside Neighborhood")
    ax2.set_xlabel("noise_keep")
    
    ax3.plot(x, in_noise)
    ax3.set_title("% Noise inside Neighborhood")
    ax3.set_xlabel("noise_keep")
    ax4.plot(x, out_noise)
    ax4.set_title("% Noise outside Neighborhood")
    ax4.set_xlabel("noise_keep")
    
    plt.savefig("overall_[" +str(index)+ "].png", bbox_inches='tight')
    plt.tight_layout()
    
def ratios(hits, truth, feature_names, noise_keep, emb_model,radius=0.4):
    neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)

    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = hits[where_to_keep]
    noise = hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))
    
    truth_np = np.array(truth.values)
    in_hits =[]
    out_hits =[]
    in_noise =[]
    out_noise =[]
        
    n_nbr = len(neighbors)
    for nbr in tqdm(range(n_nbr)):
        hood = neighbors[nbr]
        in_h = len(hood)/len(hits)
        out_h = (len(hits)-len(hood))/len(hits)
        in_hits.append(in_h)
        out_hits.append(out_h)
        noise_count = 0
        if (len(noise) == 0):
            in_noise =[]
            out_noise =[]
            in_noise_mean = 0 
            out_noise_mean = 0
        else:
            for hit in hood:
                if truth_np[hit, 1] == 0: noise_count+=1
            in_n = noise_count/len(hood)
            out_n = (len(noise) - noise_count)/len(hits)
            in_noise.append(in_n)
            out_noise.append(out_n)
            
    if(len(noise)!=0):
        in_noise_mean = statistics.mean(in_noise)
        out_noise_mean = statistics.mean(out_noise)
        
    return statistics.mean(in_hits), statistics.mean(out_hits), in_noise_mean, out_noise_mean
    
def overall_ratios():
    feature_names = ['x', 'y', 'z', 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    noise_keeps = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    
    in_hits =[]
    out_hits =[]
    in_noise=[]
    out_noise =[]
    
    for noise_keep in noise_keeps:
        hits, truth, emb_model, filter_model = get_data(noise_keep)
        in_h, out_h, in_n, out_n = ratios(hits, truth, feature_names,noise_keep, emb_model,0.4)
        in_hits.append(in_h)
        out_hits.append(out_h)
        in_noise.append(in_n)
        out_noise.append(out_n)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    ax1.plot(x, in_hits)
    ax1.set_title("% Hits inside Neighborhood")
    ax1.set_xlabel("noise_keep")
    ax2.plot(x, out_hits)
    ax2.set_title("% Hits outside Neighborhood")
    ax2.set_xlabel("noise_keep")
    
    in_noise.pop(0)
    out_noise.pop(0)
    noise_keepsn = ["0.2", "0.4", "0.6", "0.8", "1"]
    xn = [float(keep) for keep in noise_keepsn]
    ax3.plot(xn, in_noise)
    ax3.set_title("% Noise inside Neighborhood")
    ax3.set_xlabel("noise_keep")
    ax4.plot(xn, out_noise)
    ax4.set_title("% Noise outside Neighborhood")
    ax4.set_xlabel("noise_keep")
    
    plt.savefig("overall_allhits.png", bbox_inches='tight')
    plt.tight_layout()
    


def get_truth_pairs(hits, truth):
    vol = hits[['volume_id', 'layer_id']].values.T
    true_pairs = []
    pids = truth[truth['particle_id'] != 0]['particle_id'].unique()
    for pid in tqdm(pids):
        seed_hits = hits[truth['particle_id']==pid].index.values.astype(int)
        for i in seed_hits:
            hit = hits.iloc[i]
            true_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], seed_hits, vol[0], vol[1])
            true_pairs += [(i, n) for n in true_neighbors]
    return true_pairs

def apply_filter_model(hits, filter_model, neighbors, select = True, radius=0.4, threshold=0.95):
    vol = hits[['volume_id', 'layer_id']].values.T
   
    batch_size = 64
    num_workers = 12 if DEVICE=='cuda' else 0
    dataset = EdgeData(hits[feature_names].values, vol, neighbors)
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        collate_fn = my_collate)
    # apply filter model
    idx_pairs, scores = predict_pairs(loader, filter_model, batch_size)
    
    if (select):
        idx_pairs, scores = apply_filter(idx_pairs, scores, threshold)
        print("   {:6.5f}% neighbors after filter".format( (1.0 * len(scores)) / len(hits)) +" ---#pairs = {}".format(len(idx_pairs)))
    else:
        print("   {:6.5f}% neighbors before filter".format((1.0 * len(scores)) / len(hits)) +" ---#pairs = {}".format(len(idx_pairs)))
        
    return idx_pairs, scores

def get_noise_pairs(pairs,truth):
    truth_np = np.array(truth.values)
    n = 0
    for pair in tqdm(pairs):
        hit_a = truth_np[pair[0], 1]
        hit_b = truth_np[pair[1], 1]
        if hit_a == 0 or hit_b == 0: 
            n += 1
    return n