import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_heatmap(df, title):
    # Create a heatmap from the correlation matrix
    corr=df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=df.columns, yticklabels=df.columns)
    plt.title(title)
    plt.show()
    
def barplot_multiple_dataframes(dataframes, labels, title, xlabel, ylabel, legend = True):
    """Create barplot for multiple dataframes and group their barplots together
    """
    # Set the positions for the bars
    bar_width = 0.15
    bar_positions = []
    bar_positions.append(np.arange(len(dataframes[0])))
    for i in range(1, len(dataframes)):
        bar_positions.append(bar_positions[i-1]+ bar_width)

    # Plotting bar plot
    for i in range(len(dataframes)):
        plt.bar(bar_positions[i], dataframes[i], width=bar_width, label=labels[i])
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add x-axis ticks and labels
    plt.xticks(bar_positions[-1], dataframes[0].index)

    # Add legend
    if legend:
        plt.legend()

    # Show the plot
    plt.show()
    
def boxplot(list_of_lists, labels, title, xlabel, ylabel):
    """
    Create a boxplot for a list of lists of values.

    Parameters:
    - list_of_lists (list of lists): Each inner list represents the values for a boxplot.
    - labels (list of str, optional): Labels for each boxplot.
    - title (str, optional): Title of the boxplot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.

    Returns:
    - None: The function displays the boxplot.
    """
    fig, ax = plt.subplots()

    n = len(list_of_lists)
    positions = np.arange(1, n*2, 2)  # Adjust positions to be side-by-side

    box = ax.boxplot(list_of_lists, positions=positions, widths=0.6, patch_artist=True, labels=labels, showfliers=False)

    # Set different colors for each boxplot
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(f"C{i}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
    
def bar_plot_many_datapoints(
    df ,
    x_column,
    y_column,
    hue,
    x_label,
    y_label,
    title,
    legend_title,
    ):
    # Set seaborn color palette for better color distinctions
    sns.set_palette("tab10")

    # Increase the width of the bars
    bar_width = 0.8

    # Plotting
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x=x_column, y=y_column, data=df, hue='leaning', dodge=False)
    bar_plot.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.xticks([])  # Remove x-axis ticks
    #plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend(title=legend_title, bbox_to_anchor=(1, 1))  # Add legend with a title
    plt.tight_layout()
    plt.show()