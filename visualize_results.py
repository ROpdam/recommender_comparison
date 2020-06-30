import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
def plot_final_metrics(final_results,file_name, colors, metrics_to_show, y_labels, title, size=(26,12), store_path=''):
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=len(metrics_to_show))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

    for label, color in zip(final_results.keys(), colors):
        for i, metric_ts in enumerate(metrics_to_show):
            df = final_results[label]['metrics'][file_name]
            metric_mean = df[f'{metric_ts}_mean']
            metric_std = df[f'{metric_ts}_std']
            rank_at = df['rank_at']

            ax[i].plot(rank_at, metric_mean, lw=2, label=label, color=color)
            ax[i].set_title(file_name)
            ax[i].fill_between(rank_at, 
                            metric_mean + metric_std, 
                            metric_mean - metric_std, 
                            facecolor=color, alpha=0.5)

            ax[i].set_title(f'{y_labels[i]} {title}', fontsize=30)
            ax[i].set_xticks(rank_at)
            ax[i].set_xlabel('Rank@', fontsize=30)
            ax[i].set_ylabel(y_labels[i], fontsize=30)
            ax[i].tick_params(axis='both', which='major', labelsize=25)

    fig.legend(final_results.keys(), loc='lower center', ncol=len(final_results.keys()), fontsize=25, bbox_to_anchor = [0.5,-0.015])
    
    if len(store_path) > 0:
        fig.savefig(store_path)


def plot_train_stat(final_results, stat, ylabel, label, file_names, titles, color, size=(25,25), store_path=''):
    fig, ax = plt.subplots(figsize=size, nrows=len(file_names), ncols=1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#     plt.tight_layout()

    for i, file_name in enumerate(file_names):
        mean = final_results[label]['stats'][file_name][f'{stat}_mean']
        std = final_results[label]['stats'][file_name][f'{stat}_std']
        ax[i].plot(np.arange(1, len(mean)+1), mean, lw=2, label=label, color=color)
        ax[i].fill_between(np.arange(1, len(mean)+1), 
                        mean + std, 
                        mean - std,
                        facecolor=color, alpha=0.5)
        ax[i].set_xticks(np.arange(0, len(mean)+1, 5))
        ax[i].set_title(f'{label} {titles[i]}', fontsize=30)
        ax[i].set_ylabel(ylabel, fontsize=25)
        ax[i].tick_params(axis='both', which='major', labelsize=25)
    
    ax[i].set_xlabel('Epoch', fontsize=25)
    
    if len(store_path) > 0:
        fig.savefig(store_path)
        
        
# def plot_train_stats(dfs, color, size=(26,12), store_path=''):
#     """
#     """
#     to_plot = ['loss', 'val_rec@10', 'val_ndcg@10']
#     ylabels = ['Loss', 'Validation Recall@10', 'Validation NDCG@10']
#     fig, ax = plt.subplots(figsize=size, nrows=1, ncols=len(to_plot))
#     i = 0
    
#     for tp, ylabel in zip(to_plot, ylabels):
#         for df in dfs:
#             ax[i].plot(df[f'{tp}_mean'], lw=2, label=tp, color=color)
#             ax[i].fill_between(np.arange(len(df)), 
#                             df[f'{tp}_mean']+df[f'{tp}_std'], 
#                             df[f'{tp}_mean']-df[f'{tp}_std'], 
#                             facecolor=color, alpha=0.5)
#             ax[i].set_xlabel('Epoch', fontsize=25)
#             ax[i].set_ylabel(ylabel, fontsize=25)
#             ax[i].tick_params(axis='both', which='major', labelsize=25)
#         i += 1
    
#     if len(store_path) > 0:
#         fig.savefig(store_path)
        
        
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