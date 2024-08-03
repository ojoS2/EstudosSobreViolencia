import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import dash_table as dt
from statsmodels.tsa.arima.model import ARIMA
#import matplotlib.pyplot as plt

dash.register_page(__name__)


df = pd.read_csv('tabela_De_resultados.csv').reset_index()
df['Ano'] = pd.to_datetime(df['Data']).dt.year

fig_test = px.line(df[df['Municipio'] == 'Belo Horizonte'],
                     x='Data',
                     y="Taxa de crimes violentos contra o patrimônio",
                     color="Tipo",
                     markers = True)
lista_de_opcoes_muni = list(df['Municipio'].unique())

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)
#app = Dash(__name__, external_stylesheets=external_stylesheets)
#app = Dash(external_stylesheets=external_stylesheets)
layout = [
    # Todos os elementos do topo
    html.Div(children=[
            html.H1(children='Estatisticas preditivas'),
            html.Div(children='''
                Com os dados disponiveis, produzimos uma regressão estocástica:
            ''')
    ]),
    html.Div(className='row', children=[
        dcc.Dropdown(options=lista_de_opcoes_muni,
                     value='Belo Horizonte',
                     id='dropdown_subtitulo')
    ]),
    html.Div(className='row', children=[
        html.Div(children='''''', id = 'sub_titulo'),
        html.Div(children=[
            dcc.Graph(id='lineplot_1', figure=fig_test)
        ]),
    ]),
]


@callback(
    Output('lineplot_1', 'figure'),
    Input('dropdown_subtitulo', 'value'))
def update_figure(value):
    fig = px.line(df[df['Municipio'] == value],
                  x='Data',
                  y="Taxa de crimes violentos contra o patrimônio",
                  color="Tipo",
                  title="série temporal de crimes violentos contra o patrimônio:\n\t inclusos dados usados para treinamento, teste e a curva predita pelo modelo",
                  markers=True)
    return fig