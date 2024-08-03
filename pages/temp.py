import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def equal_elements(A, B):
    set_A = set(A)
    set_B = set(B)
    differences = (set_A - set_B).union(((set_B - set_A)))
    if len(differences) == 0:
        print('Elementos iguais')
    else:
        print("Diferenças detectadas:\n")
        print(differences)

def load_data():

    df_BH = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_BH.csv', sep = ';')
    df_Be = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Betin.csv', sep = ';')
    df_Br = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Brumadinho.csv', sep = ';')
    df_Ca = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Caete.csv', sep = ';')
    df_Con = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Confins.csv', sep = ';')
    df_Co = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Contagem.csv', sep = ';')
    df_Es = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Esmeraldas.csv', sep = ';')
    df_Ib = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Ibirite.csv', sep = ';')
    df_Ig = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Igarape.csv', sep = ';')
    df_Ju = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Juatuba.csv', sep = ';')
    df_LS = pd.read_csv('/home/ricardo/Downloads/DadosConsulta_Lag_Santa.csv', sep = ';')

    df_BH['Municipio'] = ['Belo Horizonte' for n in range(df_BH.shape[0])]
    df_Be['Municipio'] = ['Betim' for n in range(df_Be.shape[0])]
    df_Br['Municipio'] = ['Brumadinho' for n in range(df_Br.shape[0])]
    df_Ca['Municipio'] = ['Caete' for n in range(df_Ca.shape[0])]
    df_Con['Municipio'] = ['Confins' for n in range(df_Con.shape[0])]
    df_Co['Municipio'] = ['Contagem' for n in range(df_Co.shape[0])]
    df_Es['Municipio'] = ['Esmeraldas' for n in range(df_Es.shape[0])]
    df_Ib['Municipio'] = ['Ibirité' for n in range(df_Ib.shape[0])]
    df_Ig['Municipio'] = ['Igarapé' for n in range(df_Ig.shape[0])]
    df_Ju['Municipio'] = ['Juatuba' for n in range(df_Ju.shape[0])]
    df_LS['Municipio'] = ['Lagoa Santa' for n in range(df_LS.shape[0])]
    cols_to_use = ['Municipio',
                'Ano',
                'Taxa de crimes violentos contra o patrimnio',
                'Nmero de ocorrncias de Latrocnio',
                'Nmero de ocorrncias de Roubo',
                'Nmero de ocorrncias de mortes acidentais no trnsito',
                'Habitantes por policial militar',
                'Habitantes por policial civil']
    df = pd.concat([df_BH[cols_to_use],
                    df_Be[cols_to_use],
                    df_Br[cols_to_use],
                    df_Ca[cols_to_use],
                    df_Con[cols_to_use],
                    df_Co[cols_to_use],
                    df_Es[cols_to_use],
                    df_Ib[cols_to_use],
                    df_Ig[cols_to_use],
                    df_Ju[cols_to_use],
                    df_LS[cols_to_use]])
    return df
#print(df.head())

def treat_data(df):
    df.columns = ['Municipio',
                'Ano',
                'Taxa de crimes violentos contra o patrimônio',
                'Número de ocorrências de Latrocínio',
                'Número de ocorrências de Roubo',
                'Número de ocorrências de mortes acidentais no trânsito',
                'Habitantes por policial militar',
                'Habitantes por policial civil']
    df = df[df['Ano'] != 2022] 
    df['Taxa de crimes violentos contra o patrimônio'] = [float(item.replace(',', '.')) for item in df['Taxa de crimes violentos contra o patrimônio']]
    df['Número de ocorrências de Roubo'] = df['Número de ocorrências de Roubo'].fillna(99)
    df['Número de ocorrências de Roubo'] = df['Número de ocorrências de Roubo'].fillna(0)
    df['Habitantes por policial militar'] = [float(item.replace(',', '.')) for item in df['Habitantes por policial militar']]
    df['Habitantes por policial civil'] = df['Habitantes por policial civil'].fillna('0')
    df['Habitantes por policial civil'] = [float(item.replace(',', '.')) for item in df['Habitantes por policial civil']]

    df.to_csv('FJP_data.csv')
    return df

def load_redy_data():
    try:
        df = pd.read_csv('FJP_data.csv')
    except:
        print("No ready data to load, Retunrning empty DataFrame")
        df = pd.DataFrame({})
    return df

def independency_test(df, municipio_1='Belo Horizonte', municipio_2='Contagem'):
    x = np.array(df[(df['Municipio'] == municipio_1) & (df['Ano'] > 2011)]['Número de ocorrências de Roubo']).reshape(-1, 1)
    y = np.array(df[(df['Municipio'] == municipio_2) & (df['Ano'] > 2011)]['Número de ocorrências de Roubo']).reshape(-1, 1)
    print(mutual_info_regression(x, y))

    #temp = df[((df['Municipio'] == 'Belo Horizonte') | (df['Municipio'] == 'Contagem')) & (df['Ano'] > 2011)][['Municipio', 'Número de ocorrências de Roubo', 'Ano']]
    temp = df[df['Ano'] > 2011][['Municipio', 'Número de ocorrências de Roubo', 'Ano']]
    table = pd.pivot_table(temp, values = 'Número de ocorrências de Roubo', index='Ano', columns='Municipio')
    test = table.corr(method='spearman').reset_index().set_index('Municipio')
    return test

def find_best_model(df, max_i = 7, max_j = 7, max_k = 7):
    municipio , data, dados, tipo = [], [], [], []
    data_table = pd.DataFrame({})
    for muni in set(df['Municipio']):
        temp = df[df['Municipio'] == muni][['Ano','Taxa de crimes violentos contra o patrimônio']]
        temp['Ano'] = pd.to_datetime(temp['Ano'].astype('str') + "-01-01")
        temp = temp.set_index('Ano')
        train = temp[temp.index < pd.to_datetime("2015-01-01", format='%Y-%m-%d')]
        test = temp[temp.index >= pd.to_datetime("2015-01-01", format='%Y-%m-%d')]
        x = train['Taxa de crimes violentos contra o patrimônio']
        ind_i, ind_j, ind_k, rmsq = [], [], [], []
        print(muni)
        for i in range(0,max_i):
            for j in range(0,max_j):
                for k in range(0,max_k):
                    ind_i.append(i)
                    ind_j.append(j)
                    ind_k.append(k)
                    try:
                        ARIMAmodel = ARIMA(x, order = (i, j, k))
                        ARIMAmodel = ARIMAmodel.fit()
                        y_pred = ARIMAmodel.get_forecast(len(test.index))
                        y_pred_df = y_pred.conf_int(alpha = 0.05) 
                        y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
                        y_pred_df.index = test.index
                        arma_rmse = np.sqrt(mean_squared_error(test['Taxa de crimes violentos contra o patrimônio'].values, y_pred_df["Predictions"]))
                        rmsq.append(arma_rmse)
                    except:
                        rmsq.append(np.NAN)
        res_tab = pd.DataFrame({'i': ind_i, 'j': ind_j, 'k': ind_k, 'value': rmsq}).dropna()
        escolhido = res_tab[res_tab['value'] == res_tab['value'].min()]
        print(muni)
        print(escolhido)
        try:
            ARIMAmodel = ARIMA(x, order = (escolhido.iloc[0][0], escolhido.iloc[0][1], escolhido.iloc[0][2])).fit()
            y_pred = ARIMAmodel.get_forecast(len(test.index))
            y_pred_df = y_pred.conf_int(alpha = 0.05) 
            y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
            y_pred_df.index = test.index
        except:
            y_pred_df = pd.DataFrame({"Predictions":[np.NaN for i in range(test.shape[0])]})
        dados = dados + list(x)
        tipo = tipo + ['dados de treino' for i in range(x.shape[0])]
        municipio = municipio + [muni for i in range(x.shape[0])]
        data = data + list(train.reset_index()['Ano'])

        dados = dados + list(test['Taxa de crimes violentos contra o patrimônio'])
        tipo = tipo + ['dados reais usados para o teste' for i in range(test.shape[0])]
        municipio = municipio + [muni for i in range(test.shape[0])]
        data = data + list(test.reset_index()['Ano'])

        dados = dados + list(y_pred_df["Predictions"])
        tipo = tipo + ['projeções obtidas' for i in range(test.shape[0])]
        data = data + list(test.reset_index()['Ano'])
        municipio = municipio + [muni for i in range(test.shape[0])]
        

    data_table['Data'] = data
    data_table['Municipio'] = municipio
    data_table['Taxa de crimes violentos contra o patrimônio'] = dados
    data_table['Tipo'] = tipo
    data_table.to_csv('tabela_De_resultados.csv')
    #res.to_csv('tabela_De_parametros.csv')
    return data_table

def build_predictions_results(df):
    
    df['Ano'] = pd.to_datetime(df['Ano'].astype('str') + "-01-01")
    coefs = pd.read_csv('tabela_De_parametros.csv')
    data_table = pd.DataFrame({'Data':[], 'Municipio':[], 'Taxa de crimes violentos contra o patrimônio':[], 'tipo':[]})
    data, municipio, real, predito = [], [], [], []

    for idx, row in coefs.iterrows():
        temp = df[df['Municipio'] == row['Municipio']][['Ano','Taxa de crimes violentos contra o patrimônio']].set_index('Ano')
        train = temp[temp.index < pd.to_datetime("2015-01-01", format='%Y-%m-%d')]
        test = temp[temp.index >= pd.to_datetime("2015-01-01", format='%Y-%m-%d')]
        x = train['Taxa de crimes violentos contra o patrimônio']
        ARIMAmodel = ARIMA(x, order = (row['i'], row['j'], row['k'])).fit()
        y_pred = ARIMAmodel.get_forecast(len(test.index))
        y_pred_df = y_pred.conf_int(alpha = 0.05) 
        y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
        y_pred_df.index = test.index

        
        real = real + list(test['Taxa de crimes violentos contra o patrimônio'])
        predito = predito + list(y_pred_df["Predictions"])
        data = data + list(test.reset_index()['Ano'])
        #print(test.shape[0])
        municipio = municipio + [row['Municipio'] for i in range(test.shape[0])]

    data_table['Data'] = data
    data_table['Municipio'] = municipio
    data_table['dados reais'] = real
    data_table['daods preditos'] = predito
    data_table.to_csv('tabela_De_resultados.csv')
    
    return data_table

def print_results(df):
    pass


df = load_redy_data()
print(find_best_model(df, max_i = 7, max_j = 7, max_k = 7))


'''
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.ylabel('Taxa de crimes violentos contra o patrimônio')
plt.xlabel('Data')
plt.xticks(rotation=45)
plt.title("Divisão dos dados em treino e teste")
plt.show()
'''