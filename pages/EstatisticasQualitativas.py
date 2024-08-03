import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

dash.register_page(__name__)

#layout = html.Div([
#    html.H1('This is our Archive page'),
#    html.Div('This is our Archive page content.'),
#])



FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
df = pd.read_csv('pages/FJP_data.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.fillna(np.nan) 
lista_de_opcoes_muni = list(df['Municipio'].unique()) + ["Belo Horizonte e entornos"]
lista_de_opcoes_ano = list(df['Ano'].unique()) + ["Todos os anos"]

temp = df[['Municipio', 'Número de ocorrências de Roubo', 'Ano']]
table = pd.pivot_table(temp, values = 'Número de ocorrências de Roubo', index='Ano', columns='Municipio')
Spearman_Roubo = table.corr(method='spearman').reset_index().set_index('Municipio')

temp = df[['Municipio', 'Taxa de crimes violentos contra o patrimônio', 'Ano']]
table = pd.pivot_table(temp, values = 'Taxa de crimes violentos contra o patrimônio', index='Ano', columns='Municipio')
Spearman_Patrimonio = table.corr(method='spearman').reset_index().set_index('Municipio')


temp = df[['Municipio', 'Habitantes por policial militar', 'Ano']]
table = pd.pivot_table(temp, values = 'Habitantes por policial militar', index='Ano', columns='Municipio')
Spearman_Militar = table.corr(method='spearman').reset_index().set_index('Municipio')

temp = df[['Municipio', 'Habitantes por policial civil', 'Ano']]
table = pd.pivot_table(temp, values = 'Habitantes por policial civil', index='Ano', columns='Municipio')
Spearman_civil = table.corr(method='spearman').reset_index().set_index('Municipio')


#print(round(test['Belo Horizonte']['Betim'], 3))

lista_de_opcoes_muni = list(set(df['Municipio']) - set(["Belo Horizonte"]))


#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, FONT_AWESOME])

card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 25,
    "margin": "auto",
}

card1 = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(children='''Correlação (Spearman)
                            da ocorrencia de roubos entre
                            Belo Horizonte e Contagem''', 
                            className="card-title",
                            id='card1_title'),
                    html.P(children=f"{round(Spearman_Roubo['Belo Horizonte']['Contagem'], 3)}", className="card-text", id='card1_value'),
                ]
            )
        ),
        dbc.Card(
            html.Div(className="fa fa-list", style=card_icon),
            className="bg-primary",
            style={"maxWidth": 55},
        ),
    ],
    className="mt-4 shadow", id='card_1'
)

card2 = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(children='''Correlação (Spearman)
                            da ocorrencia de crimes violentos
                            contra o patrimônio entre os municipios
                            de Belo Horizonte e Contagem''', 
                            className="card-title",
                            id='card2_title'),
                    html.P(children=f"{round(Spearman_Patrimonio['Belo Horizonte']['Contagem'], 3)}", className="card-text", id='card2_value'),
                ]
            )
        ),
        dbc.Card(
            html.Div(className="fa fa-list", style=card_icon),
            className="bg-primary",
            style={"maxWidth": 55},
        ),
    ],
    className="mt-4 shadow", id='card_2'
)

card3 = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(children='''Correlação (Spearman)
                            do numero de habitantes por policial
                            militar entre os municipios
                            de Belo Horizonte e Contagem''', 
                            className="card-title",
                            id='card3_title'),
                    html.P(children=f"{round(Spearman_Militar['Belo Horizonte']['Contagem'], 3)}", className="card-text", id='card3_value'),
                ]
            )
        ),
        dbc.Card(
            html.Div(className="fa fa-list", style=card_icon),
            className="bg-primary",
            style={"maxWidth": 55},
        ),
    ],
    className="mt-4 shadow", id='card_3'
)

card4 = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(children='''Correlação (Spearman)
                            do numero de habitantes por policial
                            civil  entre os municipios de 
                            Belo Horizonte e Contagem''', 
                            className="card-title",
                            id='card4_title'),
                    html.P(children=f"{round(Spearman_civil['Belo Horizonte']['Contagem'], 3)}", className="card-text", id='card4_value'),
                ]
            )
        ),
        dbc.Card(
            html.Div(className="fa fa-list", style=card_icon),
            className="bg-primary",
            style={"maxWidth": 55},
        ),
    ],
    className="mt-4 shadow", id='card_4'
)

layout = html.Div(children=[
    html.H1(children='Estatisticas qualitativas'),
    dcc.Dropdown(lista_de_opcoes_muni, value = 'Contagem', id='dropdown_muni'),
    dbc.Container([dbc.Row(dbc.Col([card1, card2], md=4)), dbc.Container(dbc.Row(dbc.Col([card3, card4], md=4)))])
])



#callbacks
@callback(
    Output('card1_title','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f'''Coeficiente de correlação da
                distribuição de roubos entre
                Belo Horizonte e {value}'''


@callback(
    Output('card2_title','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f'''Correlação (Spearman)
                da ocorrencia de crimes violentos
                contra o patrimônio entre os municipios
                de Belo Horizonte e {value}'''

@callback(
    Output('card3_title','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f'''Correlação (Spearman)
                do numero de habitantes por policial
                militar entre os municipios
                de Belo Horizonte e {value}'''


@callback(
    Output('card4_title','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f'''Correlação (Spearman)
                do numero de habitantes por policial
                civil  entre os municipios de 
                Belo Horizonte e {value}'''



@callback(
    Output('card1_value','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f"{round(Spearman_Roubo['Belo Horizonte'][value], 3)}"

@callback(
    Output('card2_value','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f"{round(Spearman_Patrimonio['Belo Horizonte'][value], 3)}"

@callback(
    Output('card3_value','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f"{round(Spearman_Militar['Belo Horizonte'][value], 3)}"

@callback(
    Output('card4_value','children'),
    Input('dropdown_muni','value')
)
def update_cards(value):
    return f"{round(Spearman_civil['Belo Horizonte'][value], 3)}"



#if __name__ == "__main__":
#    app.run_server(debug=True)