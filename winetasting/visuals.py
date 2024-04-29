import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def create_confusion_matrix(analysis_df, red_herring_wines):
    
    plot_df = analysis_df.copy().sort_values(['type_accuracy', 'wine_type'])
    labels = red_herring_wines + plot_df['wine_type'].unique().tolist()
    labels = list(set(plot_df['guess_type'].unique()) - set(labels)) + labels
    
    cm = confusion_matrix(plot_df['wine_type'], plot_df['guess_type'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation='vertical', ax = ax)
    return fig, ax


def create_accuracy_visuals(analysis_df):
    
    labels = {'correct_type': 'Correct', 'wine_type': 'Wine type', 'count': 'Count',
              'price': 'Price ($)', 'wine_type__brand': 'Wine type & brand'}
    
    plot_df = analysis_df.sort_values(['type_accuracy', 'wine_type'])
    px.bar(plot_df, x='wine_type', color='correct_type', labels=labels, title='Accuracy by wine type',
           color_discrete_map={True: px.colors.qualitative.Plotly[0], False: px.colors.qualitative.Plotly[1]}).show()
    
    plot_df = analysis_df.sort_values(['bottle_accuracy', 'wine_type'])
    plot_df['wine_type__brand'] = plot_df['wine_type'] + '__' + plot_df['brand']
    px.bar(plot_df, x='wine_type__brand', color='correct_type', labels=labels, title='Accuracy by wine bottle',
           color_discrete_map={True: px.colors.qualitative.Plotly[0], False: px.colors.qualitative.Plotly[1]}).show()
    
    plot_df = analysis_df.copy()
    plot_df['wine_type__brand'] = plot_df['wine_type'] + '__' + plot_df['brand']
    plot_df = plot_df.groupby(['wine_type__brand', 'price'])['correct_type'].mean().reset_index()
    plot_df['accuracy'] = plot_df['correct_type'] * 100
    plot_df = plot_df.sort_values(['accuracy'])
    fig = px.bar(plot_df, x='wine_type__brand', y='price', color='accuracy', labels=labels, color_continuous_scale=px.colors.sequential.Bluered_r)
    fig.show()