import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash import dash_table as dt


dash.register_page(__name__, path='/')


df = pd.read_csv('pages/FJP_data.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.fillna(np.nan) 
lista_de_opcoes_muni = list(df['Municipio'].unique()) + ["Belo Horizonte e entornos"]
lista_de_opcoes_ano = list(df['Ano'].unique()) + ["Todos os anos"]
fig = px.bar(df, x="Ano", y='Taxa de crimes violentos contra o patrimônio', color="Municipio", barmode="group")
fig_sc = px.scatter(df, x="Habitantes por policial militar", y="Taxa de crimes violentos contra o patrimônio",
            size="Habitantes por policial civil", color="Municipio", hover_name="Ano",
            log_x=True, size_max=55)

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
#app = Dash(__name__, external_stylesheets=external_stylesheets)
#app = Dash(external_stylesheets=external_stylesheets)

layout = [
    # Todos os elementos do topo
    html.Div([
            html.H1(children='Testes das visualizações. Página principal'),
            html.Div(children='''
                Os dados usados são publicos e disponíveis em https://imrs.fjp.mg.gov.br/Consultas.
            ''')
            ]),
    html.Div(className='row', children='Dados',
             style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),
    html.Div(className='row', children=[
        dcc.Dropdown(options=lista_de_opcoes_muni,
                       value='Belo Horizonte e entornos',
                       id='dropdown_subtitulo')
    ]),
    html.Div(className='row', children=[
        html.Div(children='''''', id = 'sub_titulo'),
        html.Div(className='six columns', children=[
            dcc.Graph(id='barplot', figure=fig)
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(id='scaterplot', figure=fig_sc)
        ]),  
    ]),
    html.Div(children=[
            dt.DataTable(data=df.to_dict('records'), page_size=11, style_table={'height': 200, 'overflowX': 'auto'})
        ])

]
  
@callback(
    Output('sub_titulo', 'children'),
    Input('dropdown_subtitulo', 'value')
)
def update_output(value):
    return f'Criminalidade em {value}'

@callback(
    Output('barplot', 'figure'),
    Input('dropdown_subtitulo', 'value'))
def update_figure(value):
    if value == "Belo Horizonte e entornos":
        fig = px.bar(df, x="Ano", y='Taxa de crimes violentos contra o patrimônio', color="Municipio", barmode="group")
    else:
        fig = px.bar(df[df['Municipio'] == value], x="Ano", y='Taxa de crimes violentos contra o patrimônio', color="Municipio", barmode="group")
    fig.update_layout(transition_duration=500)
    return fig

@callback(
    Output('scaterplot', 'figure'),
    Input('dropdown_subtitulo', 'value'))
def update_figure_sc(value):
    if value == "Belo Horizonte e entornos":
        fig = px.scatter(df, x="Habitantes por policial militar", y="Taxa de crimes violentos contra o patrimônio",
            size="Habitantes por policial civil", color="Municipio", hover_name="Ano",
            log_x=True, size_max=55)
    else:
        fig = px.scatter(df[df['Municipio'] == value], x="Habitantes por policial militar", y="Taxa de crimes violentos contra o patrimônio",
            size="Habitantes por policial civil", color="Municipio", hover_name="Ano",
            log_x=True, size_max=55)
    fig.update_layout(transition_duration=500)
    return fig


#if __name__ == '__main__':
#    app.run(debug=True)