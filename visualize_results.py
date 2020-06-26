import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
def plot_final_metrics(all_final_results, colors, labels, metrics_to_show, add_to_title, size=(26,12), store_path=''):
    """
    """
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2)

    for final_r, color, label in zip(all_final_results, colors, labels):
        for i, metric_ts in enumerate(metrics_to_show):
            ax[i].plot(final_r['rank_at'], final_r[f'{metric_ts}_mean'], lw=2, label=label, color=color)
            ax[i].fill_between(final_r['rank_at'], 
                            final_r[f'{metric_ts}_mean']+final_r[f'{metric_ts}_std'], 
                            final_r[f'{metric_ts}_mean']-final_r[f'{metric_ts}_std'], 
                            facecolor=color, alpha=0.5)
            ax[i].set_xticks(final_r['rank_at'])
            ax[i].set_xlabel('Rank@', fontsize=30)
            ax[i].tick_params(axis='both', which='major', labelsize=25)

    ax[0].set_title('Recall ' + add_to_title, fontsize=30)
    ax[0].set_ylabel('Recall', fontsize=30)
    
    ax[1].set_title('NDCG ' + add_to_title, fontsize=30)
    ax[1].set_ylabel('NDCG', fontsize=30)
    
    fig.legend(labels, loc='lower center', ncol=len(labels), fontsize=25, bbox_to_anchor = [0.45,-0.012])
    
    if len(store_path) > 0:
        fig.savefig(store_path)
        
        
def plot_train_stats(dfs, color, size=(26,12), store_path=''):
    """
    """
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2)
    to_plot = ['loss', 'val_rec@10']
    ylabels = ['Loss', 'Validation Recall@10']
    i = 0
    
    for tp, ylabel in zip(to_plot, ylabels):
        for df in dfs:
            ax[i].plot(df[f'{tp}_mean'], lw=2, label=tp, color=color)
            ax[i].fill_between(np.arange(len(df)), 
                            df[f'{tp}_mean']+df[f'{tp}_std'], 
                            df[f'{tp}_mean']-df[f'{tp}_std'], 
                            facecolor=color, alpha=0.5)
            ax[i].set_xlabel('Epoch', fontsize=25)
            ax[i].set_ylabel(ylabel, fontsize=25)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
        i += 1
    
    if len(store_path) > 0:
        fig.savefig(store_path)
        
def plot_train_stats(dfs, color, size=(26,12), store_path=''):
    """
    """
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2)
    to_plot = ['loss', 'val_rec@10']
    ylabels = ['Loss', 'Validation Recall@10']
    i = 0
    
    for tp, ylabel in zip(to_plot, ylabels):
        for df in dfs:
            ax[i].plot(df[f'{tp}_mean'], lw=2, label=tp, color=color)
            ax[i].fill_between(np.arange(len(df)), 
                            df[f'{tp}_mean']+df[f'{tp}_std'], 
                            df[f'{tp}_mean']-df[f'{tp}_std'], 
                            facecolor=color, alpha=0.5)
            ax[i].set_xlabel('Epoch', fontsize=25)
            ax[i].set_ylabel(ylabel, fontsize=25)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
        i += 1
    
    if len(store_path) > 0:
        fig.savefig(store_path)
        
        
############################################################################################################
# def plot_metrics(metrics, legend_names, plot_title='', size=(10,8), store_path='', ndcg=True):
#     ranks_at = metrics[0]['rank_at']
    
#     figure, axes = plt.subplots(nrows=2, ncols=2, figsize=size)
#     if len(plot_title) > 0:
#         figure.suptitle(plot_title)
#     figure.subplots_adjust(wspace=0.4, hspace=0.4)
        
#     bar_width = 1.0
#     bar_dist = 1.0
#     line_width = 2
#     title_size = 'large'
    
#     #Plots
#     for i, m in enumerate(metrics):
#         axes[0,0].plot(ranks_at, m['recall'], linewidth=line_width)
#         if ndcg:
#             axes[0,1].plot(ranks_at, m['ndcg'], linewidth=line_width)
#         else:
#             axes[0,1].plot(ranks_at, m['precision'], linewidth=line_width)
#         axes[1,0].bar(ranks_at[1:] + i*bar_dist, m['hitcounts'][1:], width=bar_width, align='center')
#         axes[1,1].bar(ranks_at[0] + i*bar_dist, m['hitcounts'][0], width=bar_width, align='center')

#     ## Recall@1-20    
#     for i, rank in enumerate(ranks_at):
#         top = max([m['recall'][i] for m in metrics])
#         axes[0,0].vlines(rank, 0, top, linestyle = '--', color='gainsboro', linewidth=line_width/2) 
#     axes[0,0].set_title('Recall@1-20', fontsize=title_size)
#     axes[0,0].set_xlabel('Rank@')
#     axes[0,0].set_ylabel('Recall')
    
    
#     if ndcg:
#         ## NDCG@1-20    
#         for i, rank in enumerate(ranks_at):
#             top = max([m['ndcg'][i] for m in metrics])
#             axes[0,1].vlines(rank, 0, top, linestyle = '--', color='gainsboro', linewidth=line_width/2) 
#         axes[0,1].set_title('NDCG@1-20', fontsize=title_size)
#         axes[0,1].set_xlabel('Rank@')
#         axes[0,1].set_ylabel('NDCG')
#     else:
#         ## Precision@1-20    
#         for i, rank in enumerate(ranks_at):
#             top = max([m['precision'][i] for m in metrics])
#             axes[0,1].vlines(rank, 0, top, linestyle = '--', color='gainsboro', linewidth=line_width/2) 
#         axes[0,1].set_title('Precision@1-20', fontsize=title_size)
#         axes[0,1].set_xlabel('Rank@')
#         axes[0,1].set_ylabel('Precision')
     
#     ## Hitcounts@5-20
#     axes[1,0].set_title('Hitcounts@5-20', fontsize=title_size)
#     axes[1,0].set_xlabel('Rank@')
#     axes[1,0].set_ylabel('Hitcounts')
#     axes[1,0].set_xlim([1,24])
#     axes[1,0].set_xticks(ranks_at[1:])
    
#     ## Hitcounts@1
#     axes[1,1].set_title('Hitcounts@1', fontsize=title_size)
#     axes[1,1].set_xlabel('Rank@1')
#     axes[1,1].set_ylabel('Hitcounts')
#     axes[1,1].set_xticks([])
    
#     figure.legend(legend_names, loc='lower center', ncol=len(legend_names), fontsize='large')
    
#     if len(store_path) > 0:
#         figure.savefig(store_path)
#     plt.show()