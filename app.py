import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

#app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG])
app = Dash(__name__, use_pages=True)
server = app.server
app.layout = html.Div([
    html.H1('Estudo sobre segurança publica na micro-região de Belo Horizonte'),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
            #dcc.Link(f"{page['name']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)
