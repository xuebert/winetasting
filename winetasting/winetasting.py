import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class WineTasting:

    def __init__(self):

        self.answer_df, self.guess_df = pd.DataFrame(), pd.DataFrame()
        self.analysis_df, self.results_df = pd.DataFrame(), pd.DataFrame()

        self.red_herring_wines = []
        

    def load_example_data(self):
        
        answer_df = pd.DataFrame([
            [28, 'argentina',      'wapisa',             'malbec', 15],
            [54, 'italy',          'kirkland',           'pinot_grigio', 5],
            [86, 'spain',          'carrasvinas',        'verdejo', 8],
            [30, 'italy',          'kirkland',           'prosecco', 8],
            [92, 'usa_california', 'burgess',            'cabernet_sauvignon', 43],
            [49, 'france',         'cheateau_picadis',   'bordeaux', 7],
            [13, 'usa_oregon',     'a_to_z_wineworks',   'riesling', 13],
            [18, 'usa_california', 'taken_wine_company', 'red_blend', 16],
            [11, 'bulgaria',       'rough_day',          'cabernet_sauvignon', 9],
            [88, 'spain',          'esteban_martin',     'syrah', 5],
            [65, 'italy',          'bottega_vinaia',     'pinot_grigio', 12],
            [96, 'france',         'moet_&_chandon',     'champagne', 45],
        ], columns = ['wine_id', 'country', 'brand', 'wine_type', 'price'])
        
        self.answer_df = answer_df.convert_dtypes()
        guess_df = pd.read_csv(Path(__file__).parent.parent / 'winetasting' / 'example_guesses.csv').convert_dtypes()
        guess_df['guess_type'] = guess_df['guess_type'].fillna('na_guess_type')
        guess_df['guess_price'] = guess_df['guess_price'].fillna(-1)
        self.guess_df = guess_df.copy()

        self.red_herring_wines = ['zinfandel', 'moscato', 'grenache', 'pinot_noir', 'sauvignon_blanc']
    

    def get_results(self):
        
        analysis_df = self.guess_df.merge(self.answer_df)
        analysis_df['correct_type'] = analysis_df['wine_type'] == analysis_df['guess_type']
        analysis_df['price_diff'] = analysis_df['guess_price'] - analysis_df['price']
        analysis_df['abs_price_diff'] = analysis_df['price_diff'].abs()
        
        type_accuracy_df = analysis_df.groupby(['wine_type'])['correct_type'].sum().reset_index()
        type_accuracy_df = type_accuracy_df.merge(analysis_df['wine_type'].value_counts().reset_index().rename(columns={'index': 'wine_type', 'wine_type': 'total_guesses'}))
        type_accuracy_df['type_accuracy'] = (100 * type_accuracy_df['correct_type'] / type_accuracy_df['total_guesses']).astype(pd.Int64Dtype())
        
        bottle_accuracy_df = analysis_df.groupby(['wine_id'])['correct_type'].sum().reset_index()
        bottle_accuracy_df = bottle_accuracy_df.merge(analysis_df['wine_id'].value_counts().reset_index().rename(columns={'index': 'wine_id', 'wine_id': 'total_guesses'}))
        bottle_accuracy_df['bottle_accuracy'] = (100 * bottle_accuracy_df['correct_type'] / bottle_accuracy_df['total_guesses']).astype(pd.Int64Dtype())
        
        analysis_df = analysis_df.merge(type_accuracy_df[['wine_type', 'type_accuracy']]).merge(bottle_accuracy_df[['wine_id', 'bottle_accuracy']])
        analysis_df['percent_price_diff'] = (100 * analysis_df['price_diff'] / analysis_df['price']).round()
        self.analysis_df = analysis_df.copy()

        results_df = analysis_df.groupby(['name'])['abs_price_diff'].sum().rename('total_price_diff').reset_index()
        self.results_df = results_df.merge(analysis_df.groupby(['name'])['correct_type'].sum().rename('total_correct_type').reset_index())


def create_confusion_matrix(analysis_df, red_herring_wines):
    
    plot_df = analysis_df.copy().sort_values(['type_accuracy', 'wine_type'])
    labels = red_herring_wines + plot_df['wine_type'].unique().tolist()
    labels = list(set(plot_df['guess_type'].unique()) - set(labels)) + labels
    
    cm = confusion_matrix(plot_df['wine_type'], plot_df['guess_type'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation='vertical', ax = ax)
    return fig, ax
    