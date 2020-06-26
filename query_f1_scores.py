# -*- coding: utf-8 -*-
"""
Research Topics in AI - Assignment 2

Authors: David O'Callaghan and Arya Mohapatra

This script was used for generating the analysing
the results without needing to re-run the pipeline.
All of this code is contained within pipeline.py too.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


names=['Frame Number', 'Sedan Black', 'Sedan Silver','Sedan Red', 
       'Sedan White', 'Sedan Blue', 'Hatchback Black', 'Hatchback Silver',
       'Hatchback Red', 'Hatchback White', 'Hatchback Blue', 'Total Cars']

ground_truth = pd.read_excel("Ground Truth (Assignment 2).xlsx", 
                             skiprows=1,
                             index_col=0,
                             names=names)

predictions = pd.read_csv("results.csv",
                          skiprows=1,
                          index_col=0,
                          names=names)

def get_tp_fp_fn(y_true, y_pred):
    tp, fp, fn = 0, 0, 0
    for i in range(len(y_true)):
        # Compute TPs, FPs and FNs
        if y_pred[i] <= y_true[i]:
            tp += y_pred[i]
            fp += 0
            fn += y_true[i] - y_pred[i]
        else:
            tp += y_true[i]
            fp += y_pred[i] - y_true[i]
            fn += 0
    return tp, fp, fn
        
def compute_f1score(tp, fp, fn):
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * (pre * rec) / (pre + rec)

f1_scores = []
# Q1 
n_frames = len(ground_truth)
y_true = ground_truth.loc[:,names[-1]].values
y_pred = predictions.loc[:,names[-1]].values

tp, fp, fn = get_tp_fp_fn(y_true, y_pred)

# Calculate F1-score
f1_scores.append(compute_f1score(tp, fp, fn))

# Q2
tp, fp, fn = 0, 0, 0
true_values = ground_truth[ground_truth['Total Cars'].values != 0]
pred_values = predictions[ground_truth['Total Cars'].values != 0]

for cols in (names[1:6], names[6:11]): # Sedan then Hatchback
    y_true = true_values.loc[:,cols].sum(axis=1).values
    y_pred = pred_values.loc[:,cols].sum(axis=1).values
    
    tp_, fp_, fn_ = get_tp_fp_fn(y_true, y_pred)
    tp += tp_
    fp += fp_
    fn += fn_

# Calculate F1-score
f1_scores.append(compute_f1score(tp, fp, fn))

# Q3
tp, fp, fn = 0, 0, 0
true_values = ground_truth[ground_truth['Total Cars'].values != 0]
pred_values = predictions[ground_truth['Total Cars'].values != 0]

for i in range(1,11): # For each type and colour
    col = names[i]
    y_true = true_values.loc[:,col].values
    y_pred = pred_values.loc[:,col].values
    
    tp_, fp_, fn_ = get_tp_fp_fn(y_true, y_pred)
    tp += tp_
    fp += fp_
    fn += fn_

# Calculate F1-score
f1_scores.append(compute_f1score(tp, fp, fn))

for i, f1 in enumerate(f1_scores):
    print(f'Q{i+1} : {round(f1, 3)}\n')
    
plt.bar([f"Q{i}" for i in range(1,4)], (f1_scores))
plt.title('Query F1 Scores')
plt.grid(axis='y', alpha=0.5)
plt.ylim([0,1])
plt.show()

q1_stats = np.load("q1_stats.npy")
q2_stats = np.load("q2_stats.npy")
q3_stats = np.load("q3_stats.npy")

def make_piecewise(stats):
    N = stats.shape[0]
    previous = stats[0,0]
    j = 1
    for _ in range(1, N):
        current = stats[j,0]
        if current != previous + 1:
            # Insert NaN for gaps frame that don't have data
            stats = np.insert(stats, j, np.array([np.nan, np.nan]), axis=0)
            j += 1
        previous = current
        j += 1
    return stats
    
print('Average extraction times')
print('------------------------')
for i, q_stats in enumerate([q1_stats, q2_stats, q3_stats]):
    average_extraction = np.round(np.mean(q_stats[:,1]), 4)
    std_extraction = np.round(np.std(q_stats[:,1]) , 4)
    print(f'Q{i+1} : {average_extraction} +- {std_extraction} s')
    
q1_stats = make_piecewise(q1_stats)
q2_stats = make_piecewise(q2_stats)
q3_stats = make_piecewise(q3_stats)

plt.plot(q1_stats[:,0], q1_stats[:,1], 'g', alpha=1, linewidth=1)
plt.plot(q2_stats[:,0], q2_stats[:,1], 'b', alpha=0.7, linewidth=1)
plt.plot(q3_stats[:,0], q3_stats[:,1] + 0.001, 'r', alpha=0.7, linewidth=1)
plt.legend(['Q1', 'Q2', 'Q3'])
plt.xlabel('Frame Number')
plt.ylabel('Time (seconds)')
plt.title('Query Extraction Time')
plt.ylim([0.1,0.26])
plt.show()
