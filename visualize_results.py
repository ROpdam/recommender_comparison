import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
def plot_final_metrics(final_results,file_name, colors, metrics_to_show, y_labels, title, size=(26,12), store_path='', show_legend=True):
    fig, ax = plt.subplots(figsize=size, nrows=1, ncols=len(metrics_to_show))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0.3, hspace=None)

    for label, color in zip(final_results.keys(), colors):
        for i, metric_ts in enumerate(metrics_to_show):
            df = final_results[label]['metrics'][file_name]
            metric_mean = df[f'{metric_ts}_mean']
            metric_std = df[f'{metric_ts}_std']
            rank_at = df['rank_at']

            ax[i].plot(rank_at, metric_mean, lw=4, label=label, color=color)
            ax[i].fill_between(rank_at, 
                            metric_mean + metric_std, 
                            metric_mean - metric_std, 
                            facecolor=color, alpha=0.5)

#             ax[i].set_title(f'{y_labels[i]} {title}', fontsize=40)
            ax[i].set_xticks(rank_at)
            ax[i].set_xlabel('Rank@', fontsize=35)
            ax[i].set_ylabel(y_labels[i], fontsize=35)
            ax[i].tick_params(axis='both', which='major', labelsize=30)
    if show_legend:
        leg = fig.legend(final_results.keys(), loc='lower center', ncol=len(final_results.keys()), fontsize=35, bbox_to_anchor = [0.45,-0.01])
        
        leg.get_lines()[0].set_linewidth(8)
        leg.get_lines()[1].set_linewidth(8)
        leg.get_lines()[2].set_linewidth(8)
    
    if len(store_path) > 0:
        fig.savefig(store_path + '.pdf', bbox_inches='tight')


def plot_train_stat(final_results, stat, ylabel, label, file_names, titles, color, size=(8,20), store_path=''):
    fig, ax = plt.subplots(figsize=size, nrows=len(file_names), ncols=1)
    plt.subplots_adjust(left=0.3, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
#     plt.tight_layout()
    
    for i, file_name in enumerate(file_names):
        if file_name == 'ml_1m' and label == 'CFRNN':
            step = 20
        else:
            step = 5
        mean = final_results[label]['stats'][file_name][f'{stat}_mean']
        std = final_results[label]['stats'][file_name][f'{stat}_std']
        ax[i].plot(np.arange(1, len(mean)+1), mean, lw=2, label=label, color=color)
        ax[i].fill_between(np.arange(1, len(mean)+1), 
                        mean + std, 
                        mean - std,
                        facecolor=color, alpha=0.5)
        ax[i].set_xticks(np.arange(0, len(mean)+1, step))
        ax[i].set_title(f'{label} {titles[i]}', fontsize=30)
        ax[i].set_ylabel(ylabel, fontsize=25)
        ax[i].tick_params(axis='both', which='major', labelsize=25)
    
    ax[i].set_xlabel('Epoch', fontsize=25)
    
    if len(store_path) > 0:
        fig.savefig(store_path + '.pdf')