import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def violin_plot(df, target_column, title, filter_func=None):

    fig, ax = plt.subplots(figsize=(19, 6))

    # Store number of targets found in generated text
    if filter_func is None:
        targets_length = [len(row[target_column]) for _, row in df.dropna().iterrows()]
    # If specified, a filtered is applied (eg: lambda x: len([a for a in x if a[1] in {'PROC', 'DISO', 'CHEM'}])
    else:
        targets_length = [filter_func(row[target_column]) for _, row in df.dropna().iterrows()]
    
    # Define a color for each violin depending on the proportion of targets found during the most successful try
    cmap = plt.cm.Reds
    colors = []
    data = []
    for i, (_, row) in enumerate(df.dropna().iterrows()):
        max_val = max(row[f'# {target_column} found each try'])
        colors.append(cmap(max_val / targets_length[i]))

        # Format the data before plotting
        for val in row[f'# {target_column} found each try']:
            data.append({'Row': i, 'Value': val})
    data_df = pd.DataFrame(data)

    sns.violinplot(data=data_df, x='Row', y='Value', ax=ax, inner=None, cut=0, palette=colors)

    # Add legend
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)), ax=ax)
    cbar.set_label(f'Percentage of {target_column} found')
    ax.set_xlabel('Row')
    ax.set_ylabel(f'Number of {target_column}')
    ax.set_title(title)

    return plt