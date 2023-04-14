import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report

import pandas as pd
from ydata_profiling import ProfileReport

import numpy as np

import matplotlib
matplotlib.use('agg')
from tkinter import *
from mttkinter import *

import seaborn as sns

from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import AgglomerativeClustering

from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
import statsmodels.api as sm

import scipy.cluster.hierarchy as shc

from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

from gower import gower_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix

from scipy.spatial.distance import pdist, squareform

from datetime import datetime

import plotly.figure_factory as ff

from PIL import Image
import io
from io import BytesIO

st.set_page_config(page_title='Curso Cientista de Dados EBAC: Modulo 31 - Exercício 02', 
                   page_icon='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQlWkTPwbN4sIWlFBLTlDUnHXFNHRGqslRP6cTgzcOUdfFWxY_kS2rok5rUwCRbe3Hg-0&usqp=CAU', 
                   layout="wide", initial_sidebar_state="auto", menu_items=None)

with st.sidebar:
    
    image = Image.open('Por-Que-e-Como-Data-Science-e-Mais-do-Que-Apenas-Machine-Learning.jpg')
    st.sidebar.image(image)
    st.markdown("---")
    st.title('Mod31Ex02 - Cientista de Dados - EBAC')
    st.subheader('Aluno: Matheus Feldberg')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/matheus-feldberg-521a93259)')
    st.markdown('[GitHub](https://github.com/math-feldberg/Mod16Ex01)')
    st.markdown('[Kaggle](https://https://www.kaggle.com/matheusfeldberg)')
    st.markdown("---")
         
    df = pd.read_csv('online_shoppers_intention.csv')
    df.index.name='id'
   
    @st.cache_data
    def convert_df(df):
                                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8') 
    
    csv = convert_df(df)
    st.download_button(
                                label="📥 Download do Dataframe em CSV",
                                data=csv,
                                file_name='dataframe.csv',
                                mime='text/csv')                                   
    @st.cache_data
    def to_excel(df):
                                output = BytesIO()
                                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                df.to_excel(writer, index=False, sheet_name='Sheet1')
                                writer.save()
                                processed_data = output.getvalue()
                                return processed_data
    df_xlsx = to_excel(df)
                                
    st.download_button(
                                label='📥 Download do Dataframe em EXCEL',
                                data=df_xlsx,
                                file_name= 'dataframe.xlsx')         
with st.sidebar:
    selected = option_menu(
                        menu_title = 'Sumário',
                        options =  ['1. Agrupamento hierárquico',
                        '2. Análise descritiva',
                        '3. Variáveis de agrupamento',
                        '4. Número de grupos',
                        '5. Avaliação dos grupos',
                        '6. Avaliação de resultados'],
                        default_index=0)
    
if selected == '1. Agrupamento hierárquico':
        
    st.subheader('1. Agrupamento hierárquico')
                
    st.markdown('''Neste exercício vamos usar a base [online shoppers purchase intention](https://https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
    de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://link.springer.com/article/10.1007/s00521-018-3523-0).
                            
A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12
meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com
corportamento de navegação diferentes possuem propensão a compra diferente?

Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informções da data,
como a proximidade à uma data especial, fim de semana e o mês.''')
                
    st.markdown('''
|Variavel                |Descrição          | 
|------------------------|:-------------------| 
|Administrative          | Quantidade de acessos em páginas administrativas| 
|Administrative_Duration | Tempo de acesso em páginas administrativas | 
|Informational           | Quantidade de acessos em páginas informativas  | 
|Informational_Duration  | Tempo de acesso em páginas informativas  | 
|ProductRelated          | Quantidade de acessos em páginas de produtos | 
|ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
|BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
|ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
|PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
|SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
|Month                   | Mês  | 
|OperatingSystems        | Sistema operacional do visitante | 
|Browser                 | Browser do visitante | 
|Region                  | Região | 
|TrafficType             | Tipo de tráfego                  | 
|VisitorType             | Tipo de visitante: novo ou recorrente | 
|Weekend                 | Indica final de semana | 
|Revenue                 | Indica se houve compra ou não |

\* variávels calculadas pelo google analytics''')

    st.subheader('Carregando os pacotes')   

    st.markdown('É considerada uma boa prática carregar os pacotes que serão utilizados como a primeira coisa do programa.')

    st.markdown('Usaremos o seguinte código:')
                                            
    code = '''import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as shc

from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

from gower import gower_matrix
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix

from scipy.spatial.distance import pdist, squareform'''

    st.code(code, language='python')

    st.subheader('Carregando os dados') 
    st.markdown('O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto *dataframe* do pandas.')
    st.markdown('O comando *df.head()* retorna as cinco primeiras linhas do dataframe.')

    df = pd.read_csv('online_shoppers_intention.csv')
    df.index.name='id'
     
    st.dataframe(df.head())

    st.markdown('Ao lado você pode baixar o dataframe original em formato CSV e EXCEL.')
                                      
elif selected == '2. Análise descritiva':
    
    st.header('2. Análise descritiva')

    st.markdown('Nessa etapa faremos uma análise descritiva das variáveis do escopo:')

    st.markdown('- Verificaremos a distribuição das variáveis com o comando *df.info()*:')
    
      
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)
    
    st.markdown('- Verificaremos se existem valores missing com os comandos *df.isna().sum()* e *df.isnull().sum()* e, caso positivo, faremos o devido tratamento')

    st.text(df.isna().sum())

    st.text(df.isnull().sum())

elif selected == '3. Variáveis de agrupamento':
                            
    st.header('3. Variáveis de agrupamento')

    st.markdown('Nessa etapa listaremos as variáveis a serem utilizadas:')

    st.markdown(''' 
    - Seleção para o agrupamento das variáveis que descrevam o padrão de navegação na sessão.
    
    As variáveis do dataframe são:''')

    st.text(df.columns)   

    ('''Selecionaremos as seguintes: 
    
    *Administrative, Administrative_Duration, Informational, Informational_Duration, 
    ProductRelated e ProductRelated_Duration*
    
    ''')

    ('''- Seleção das variáveis que indiquem a característica da data.''')

    st.dataframe(df[['Month','Weekend']] )   

    ('''- Tratamento das variáveis qualitativas (convertendo em variáveis *dummies*).''')
    
    df1 = df[['Administrative', 'Administrative_Duration', 'Informational',
          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
          'Month','Weekend']]    
    df2 = pd.get_dummies(df1)
    df2.replace({False: 0, True: 1}, inplace=True)
    df2.astype(float)
    st.dataframe(df2.head()) 

    ('''- Tratamento dos valores faltantes.
                                    ''')
    st.text(df2.isna().sum())

    st.text(df2.isnull().sum())
 
elif selected == '4. Número de grupos':
                        
                st.header('4. Número de grupos')

                st.markdown('''Nesta atividade vamos adotar uma abordagem bem pragmática e avaliar agrupamentos 
                hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do 
                diretor da empresa.''')
                
                st.markdown('O código abaixo permite escolher o número de *clusters* (n) que no nosso caso serão 3 e 4:')

                code = '''
clus = AgglomerativeClustering(linkage='complete',
                               distance_threshold = None,
                               n_clusters = n)
clus.fit(df2_pad)
df2['3_grupos'] = clus.labels_
'''

                st.code(code, language='python') 
             
elif selected == '5. Avaliação dos grupos':

    st.header('5. Avaliação dos grupos')

    st.markdown('''Depois de tratadas as variáveis qualitativas através da conversão em *dummies* e da padronização 
    das variáveis qualitativas, construiremos os agrupamentos com através do *AgglomerativeClustering*''')

    st.markdown('- Dataframe original:')

    st.dataframe(df.head())

    st.markdown('- Dataframe após a seleção das variáveis de estudo:')

    df1 = df[['Administrative', 'Administrative_Duration', 'Informational',
          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
          'Month','Weekend']]  
    
    st.dataframe(df1.head())

    st.markdown('- Dataframe selecioando após o tratamento das variáveis:')

    df2 = pd.get_dummies(df1)
    df2.replace({False: 0, True: 1}, inplace=True)
    df2.astype(float)
    st.dataframe(df2.head())

    st.markdown('- Padronização:')

    code = '''
    padronizador = StandardScaler()
     data = padronizador.fit_transform()
'''
    st.code(code, language='python')    
    st.markdown('- Criando os grupos de aglomeração com 3 e 4 *clusters*:')

    code = '''
clus = AgglomerativeClustering(linkage='complete',
                               distance_threshold = None,
                               n_clusters=3)
clus.fit(df2_pad)
df2['3_grupos'] = clus.labels_
data = df.merge(df2['3_grupos'], how='left', on='id')'''
    st.code(code, language='python') 

    code = '''
clus = AgglomerativeClustering(linkage='complete',
                               distance_threshold = None,
                               n_clusters=4)
clus.fit(df2_pad)
df2['4_grupos'] = clus.labels_'''

    st.code(code, language='python') 
    
    padronizador = StandardScaler()
    df2_pad = padronizador.fit_transform(df2)
    clus = AgglomerativeClustering(linkage='complete',
                               distance_threshold = None,
                               n_clusters=3)
    clus.fit(df2_pad)
    df2['3_grupos'] = clus.labels_
    data = df.merge(df2['3_grupos'], how='left', on='id')

    st.markdown('- *Crosstab* do grupo com 3 *clusters* com as variáveis explicatias **BounceRates** e **Revenue**, considerando a variável de marcação de tempo **Weekend**:')

    st.text(pd.crosstab([data['3_grupos']], [data['Revenue'], data['Weekend']]))

    st.text(pd.crosstab([data['3_grupos']], [data['BounceRates'], data['Weekend']]))

    st.markdown('**Podemos perceber que o grupo 0 apresentou maior conversão de vendas (Revenue) em dias de semana. O segundo frame mostra a taxa de evasão (BounceRates) sem compras.**')

    st.markdown('- *Crosstab* do grupo com 3 *clusters* com as variáveis explicatias **BounceRates** e **Revenue**, considerando a variável de marcação de tempo **Month**:')

    st.text(pd.crosstab([data['3_grupos']], [data['Revenue'], data['Month']]))

    st.text(pd.crosstab([data['3_grupos']], [data['BounceRates'], data['Month']]))

    st.markdown('**Aqui temos a conversão de compras (Revenue) e de evasão (BounceRates) distribuída por mês.**')

    clus = AgglomerativeClustering(linkage='complete',
                               distance_threshold = None,
                               n_clusters=4)
    clus.fit(df2_pad)
    df2['4_grupos'] = clus.labels_
    data = df.merge(df2['4_grupos'], how='left', on='id')

    st.markdown('- *Crosstab* do grupo com 4 *clusters* com as variáveis explicatias **BounceRates** e **Revenue**, considerando a variável de marcação de tempo **Weekend**:')

    st.text(pd.crosstab([data['4_grupos']], [data['Revenue'], data['Weekend']]))

    st.text(pd.crosstab([data['4_grupos']], [data['Revenue'], data['BounceRates']]))

    st.markdown('**Podemos perceber que o grupo 1 apresentou maior conversão de vendas (Revenue) em dias de semana. O segundo frame mostra a taxa de evasão (BounceRates) sem compras.**')

    st.markdown('- *Crosstab* do grupo com 4 *clusters* com as variáveis explicatias **BounceRates** e **Revenue**, considerando a variável de marcação de tempo **Month**:')

    st.text(pd.crosstab([data['4_grupos']], [data['Revenue'], data['Month']]))

    st.text(pd.crosstab([data['4_grupos']], [data['BounceRates'], data['Month']]))

    st.markdown('**Aqui temos a conversão de compras (Revenue) e de evasão (BounceRates) distribuída por mês.**')

elif selected == '6. Avaliação de resultados':
                        
    st.header('6. Avaliação de resultados')

    st.markdown('Considerando a análise dos dados podemos concluir que, dos grupos acima, os que possuem clientes propensos à compra são:')

    st.markdown('- Para o modelo com 3 clusters o grupo 0 é o que possui clientes mais propensos à compra.')

    st.markdown('- Para o modelo com 4 clusters o grupo 1 é o que possui clientes mais propensos à compra.')
