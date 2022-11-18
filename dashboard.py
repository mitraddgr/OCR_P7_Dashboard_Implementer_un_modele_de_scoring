import pickle

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import shap
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.image import imread
from sklearn.neighbors import NearestNeighbors

#Titre de la page
st.set_page_config(page_title="Projet 7-Dashboard Pret à dépenser",
				   page_icon = "icon_pret_a_depenser.png",layout="wide")


components.html("""<body style="margin: 0">
                        <h1 style="margin: 0;font-family: Source Sans Pro, sans-serif;
                        font-weight: 700; color: #4F8BF9;font-size: 2.45rem;text-align: center;">
                        <span>Projet 7-Dashboard Pret à dépenser</span>
                        </h1>
                    <body>""",
                    width=None, height=50, scrolling=False)

def load_data():
	'''Fonction chargeant et calculant les données nécessaires au dashboard.
	Ne prend pas de paramètres en entrée
	'''


	#Chargement des données de test
	db_test=pd.read_csv('./Data/df_app.csv')
	db_test['YEARS_BIRTH']=(db_test['DAYS_BIRTH']/-365).apply(lambda x: int(x))
	db_test=db_test.reset_index(drop=True)
	df_test=pd.read_csv('./Data/df_test3.csv')
	df_test_sample = pd.read_csv('./Data/test_sample.csv', sep=';')
	logo=imread("./Data/logo.png")


	# Chargement du modèle pré-entrainé
	pickle_in = dvc.api.read('Data/lgbm_model2.pkl', mode='rb')
	lgbm=pickle.loads(pickle_in)

	#Calcul des SHAP value
	explainer = shap.TreeExplainer(lgbm)
	shap_values = explainer.shap_values(df_test)[1]
	exp_value=explainer.expected_value[1]
	return db_test,df_test,df_test_sample,shap_values,lgbm,exp_value,logo


def tab_client(db_test):

	'''Fonction pour afficher le tableau du portefeuille client avec un système de 6 champs de filtres
	permettant une recherche plus précise.
	Le paramètre est le dataframe des clients
	'''
	st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{"Tableau clientèle"}</p>',unsafe_allow_html=True)
	row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row_spacer5 = st.columns([1,.1,1,.1,1,.1,4])

	#Définition des filtres via selectbox
	sex=row0_1.selectbox("Sexe",['All']+db_test['CODE_GENDER'].unique().tolist())
	age=row0_1.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))
	fam=row0_2.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
	child=row0_2.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
	pro=row0_3.selectbox("Statut pro.",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
	stud=row0_3.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())

	#Affichage du dataframe selon les filtres définis
	db_display=db_test[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
	db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
	db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
	db_display['AMT_INCOME_TOTAL']=db_test['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
	db_display['AMT_CREDIT']=db_test['AMT_CREDIT'].apply(lambda x: int(x))
	db_display['AMT_ANNUITY']=db_test['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

	db_display=filter(db_display,'CODE_GENDER',sex)
	db_display=filter(db_display,'YEARS_BIRTH',age)
	db_display=filter(db_display,'NAME_FAMILY_STATUS',fam)
	db_display=filter(db_display,'CNT_CHILDREN',child)
	db_display=filter(db_display,'NAME_INCOME_TYPE',pro)
	db_display=filter(db_display,'NAME_EDUCATION_TYPE',stud)

	st.dataframe(db_display)
	st.markdown(
		f'<h1 style="color:#6ad46a;font-size:24px;">{"Total clients correspondants: " + str(len(db_display))}</h1>',
		unsafe_allow_html=True)

def filter(df,col,value):
	'''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
	if value!='All':
		db_filtered=df.loc[df[col]==value]
	else:
		db_filtered=df
	return db_filtered


def get_client(db_test):
	"""Sélection d'un client via une selectbox"""
	client=st.selectbox('Client',db_test['SK_ID_CURR'])
	idx_client=db_test.index[db_test['SK_ID_CURR']==client][0]
	return client,idx_client


def infos_client(db_test,client,idx_client):
	"""Affichage des infos du client sélectionné dans la barre latérale"""
	st.markdown("ID client: "+str(client))
	st.markdown("Sexe: "+db_test.loc[idx_client,'CODE_GENDER'])
	st.markdown("Age: " + str(db_test.loc[idx_client, 'YEARS_BIRTH']))
	st.markdown("Statut familial: "+db_test.loc[idx_client,'NAME_FAMILY_STATUS'])
	st.markdown("Enfants: "+str(db_test.loc[idx_client,'CNT_CHILDREN']))
	st.markdown("Statut professionnel : "+db_test.loc[idx_client,'NAME_INCOME_TYPE'])
	st.markdown("Niveau d'études: "+db_test.loc[idx_client,'NAME_EDUCATION_TYPE'])

def comparaison(df_test,db_test,idx_client):
	"""Fonction principale de l'onglet 'Comparaison clientèle' """
	st.title('Dashboard Pret à dépenser')
	st.subheader('Comparaison clientèle')
	idx_neigh,total=get_neigh(df_test,idx_client)
	db_neigh=db_test.loc[idx_neigh,['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','TARGET']]
	db_neigh['AMT_INCOME_TOTAL']=db_neigh['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
	db_neigh['AMT_CREDIT']=db_neigh['AMT_CREDIT'].apply(lambda x: int(x))
	db_neigh['AMT_ANNUITY']=db_neigh['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

	if total:
		display_charts(db_test,idx_client)

	else:
		display_charts(db_neigh,idx_client)


def get_neigh(df,idx_client):
	"""Calcul des voisins les plus proches du client sélectionné
	Sélection du nombre de voisins par un slider.
	Retourne les proches voisins et un booléen indiquant la clientèle globale ou non"""
	row1,row_spacer1,row2,row_spacer2 = st.columns([1,.1,.3,3])
	size=row1.slider("Taille du groupe de comparaison",min_value=10,max_value=1000,value=500)
	row2.write('')
	total=row2.button(label="Clientèle globale")
	neigh= NearestNeighbors(n_neighbors=size)
	neigh.fit(df)
	k_neigh=neigh.kneighbors(df.loc[idx_client].values.reshape(1,-1),return_distance=False)[0]
	k_neigh=np.sort(k_neigh)
	return k_neigh,total

def display_charts(df,client):
	"""Affichae des graphes de comparaison pour le client sélectionné """
	row1_1,row1_2,row1_3 = st.columns(3)
	st.write('')
	row2_10,row2_2,row2_3 = st.columns(3)

	chart_kde("Répartition de l'age",row1_1,df,'YEARS_BIRTH',client)
	chart_kde("Répartition des revenus",row1_2,df,'AMT_INCOME_TOTAL',client)
	chart_bar("Répartition du nombre d'enfants",row1_3,df,'CNT_CHILDREN',client)

	chart_bar("Répartition du statut professionel",row2_10,df,'NAME_INCOME_TYPE',client)
	chart_bar("Répartition du niveau d'études",row2_2,df,'NAME_EDUCATION_TYPE',client)
	chart_bar("Répartition du type de logement",row2_3,df,'NAME_HOUSING_TYPE',client)
	st.dataframe(df[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']])


def chart_kde(title,row,df,col,client):
	"""Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
	with row:
		st.subheader(title)
		fig,ax = plt.subplots()
		sns.kdeplot(df.loc[df['TARGET']==0,col],color='#fcba03', label = 'Target == 0')
		sns.kdeplot(df.loc[df['TARGET']==1,col],color='#3145c4', label = 'Target == 1')
		plt.axvline(x=df.loc[client,col],ymax=0.95,color='black')
		plt.legend()
		st.pyplot(fig)


def chart_bar(title, row, df, col, client):
	"""Définition des graphes barres avec une ligne horizontale indiquant la position du client"""
	with row:
		st.subheader(title)
		fig, ax = plt.subplots()
		data = df[['TARGET', col]]
		if data[col].dtypes != 'object':
			data[col] = data[col].astype('str')

			data1 = round(data[col].loc[data['TARGET'] == 1].value_counts() / data[col].loc[
				data['TARGET'] == 1].value_counts().sum() * 100, 2)
			data0 = round(data[col].loc[data['TARGET'] == 0].value_counts() / data[col].loc[
				data['TARGET'] == 0].value_counts().sum() * 100, 2)
			data = pd.concat([pd.DataFrame({"Pourcentage": data0, 'TARGET': 0}),
							  pd.DataFrame({'Pourcentage': data1, 'TARGET': 1})]).reset_index().rename(
				columns={'index': col})
			sns.barplot(data=data, x='Pourcentage', y=col, hue='TARGET', palette=['#fcba03', '#3145c4'],
						order=sorted(data[col].unique()));

			data[col] = data[col].astype('int64')

			plt.axhline(y=sorted(data[col].unique()).index(df.loc[client, col]), xmax=0.95, color='black', linewidth=4)
			st.pyplot(fig)
		else:

			data1 = round(data[col].loc[data['TARGET'] == 1].value_counts() / data[col].loc[
				data['TARGET'] == 1].value_counts().sum() * 100, 2)
			data0 = round(data[col].loc[data['TARGET'] == 0].value_counts() / data[col].loc[
				data['TARGET'] == 0].value_counts().sum() * 100, 2)
			data = pd.concat([pd.DataFrame({"Pourcentage": data0, 'TARGET': 0}),
							  pd.DataFrame({'Pourcentage': data1, 'TARGET': 1})]).reset_index().rename(
				columns={'index': col})
			sns.barplot(data=data, x='Pourcentage', y=col, hue='TARGET', palette=['#fcba03', '#3145c4'],
						order=sorted(data[col].unique()));

			plt.axhline(y=sorted(data[col].unique()).index(df.loc[client, col]), xmax=0.95, color='black', linewidth=4)
			st.pyplot(fig)


def score_viz(df_test,client,exp_value,shap_values):
	"""Fonction principale de l'onglet 'Score visualisation' """

	score,result=prediction(client)


	fig = go.Figure(go.Indicator(
	mode = "gauge+number+delta",
    value = score,
    number = {'font':{'size':48}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': result.tolist(), 'font': {'size': 28, 'color':color(result)}},
    delta = {'reference': 0.48, 'increasing': {'color': "#3145c4"},'decreasing':{'color':'#fcba03'}},
    gauge = {
        'axis': {'range': [0,1], 'tickcolor': color(result)},
        'bar': {'color': color(result)},
        'steps': [
            {'range': [0,0.48], 'color': 'lightgreen'},
            {'range': [0.48,1], 'color': 'lightcoral'}],
        'threshold': {
            'line': {'color': "black", 'width': 5},
            'thickness': 1,
            'value': 0.48}}))


	st.plotly_chart(fig)

	st_shap(shap.force_plot(exp_value, shap_values[client], features = df_test.iloc[client], feature_names=df_test.columns, figsize=(12,5)))

def prediction(id):
	y_pred=get_proba_for_client(id)
	decision=np.where(y_pred>0.48,"Rejected","Approved")
	return y_pred,decision

def color(pred):
	'''Définition de la couleur selon la prédiction'''
	if pred=='Approved':
		col='#fcba03'
	else :
		col='#3145c4'
	return col

def st_shap(plot, height=None):
	"""Fonction permettant l'affichage de graphique shap values"""
	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
	components.html(shap_html, height=height)


def get_proba_for_client(client_id:str):
    #url = f'https://ocr-p7-api.herokuapp.com/predict?id_client={client_id}'
    url = f'https://ocr-p7-api.herokuapp.com/predict?id_client=322225'
    x = requests.post(url)
    return x.json()['probability']

def main():
	"""Fonction principale permettant l'affichage de la fenêtre latérale avec les 3 onglets.
	"""
	db_test,df_test,df_test_sample,shap_values,lgbm,exp_value,logo=load_data()
	st.sidebar.image(logo)
	PAGES = [
	    "Tableau clientèle",
	    "Comparaison clientèle",
		"Visualisation score"

	]

	st.sidebar.write('')
	st.sidebar.write('')
	st.sidebar.write('')

	st.sidebar.title('Pages')
	selection = st.sidebar.radio("Go to", PAGES)

	if selection=="Tableau clientèle":
		tab_client(db_test)
	if selection=="Comparaison clientèle":
		st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{"Comparaison clientèle"}</p>',unsafe_allow_html=True)
		client,idx_client=get_client(db_test)
		infos_client(db_test,client,idx_client)
		comparaison(df_test,db_test,idx_client)
	if selection=="Visualisation score":
		st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{"Visualisation score"}</p>',unsafe_allow_html=True)
		st.markdown('Veuillez sélectionner un numéro de demande de prêt')
		client,idx_client=get_client(db_test)
		infos_client(db_test,client,idx_client)
		score_viz(df_test,idx_client,exp_value,shap_values)


if __name__ == '__main__':
	main()



