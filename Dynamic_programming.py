# -*- coding: utf-8 -*-
"""
Devoir 4 Exercise 1
Construisez l'enveloppe supérieure de líoption de vente Bermudéenne Ben-Ameur et al. (2013)
"""

import time 
import datetime
import numpy as np 
import pandas as pd

from math import sqrt 
from scipy.stats import norm

def lecture_inputs(filename):
	inputs = pd.read_excel(
		filename,sheet_name = "Inputs", header=None).values

	inputs = {key: value for (key, value) in inputs}

	return inputs


def programmation_dynamique(inputs):
	"""
	tarfication de l'option  par programmation dynamique et éléments finis 
	"""
	X0 = inputs["X0"]
	K = inputs["K"]
	T = inputs["T"]
	r = inputs["r"]
	sigma = inputs["sigma"]
	type_option = inputs["option type"]

	N = int(inputs["N"]) # nombre de dates de décision
	p = int(inputs["p"])
	dt = T / N

	debut = time.perf_counter()	

	# drift et diffusion 
	drift = (r - 0.5 * sigma ** 2) * T
	diffusion = sigma * sqrt(T)

	# créer une grille de points 
	grille = np.zeros(p+1)

	# les autres points 
	grille[1] = X0 * np.exp(drift - 7 * diffusion)
	grille[-1] = X0 * np.exp(drift + 7 * diffusion)

	grille[2] = X0 * np.exp(drift - 5 * diffusion)
	grille[-2] = X0 * np.exp(drift + 5 * diffusion)
	
	delta = np.arange(3,p-1) / p
	grille[3:-2] = X0 * np.exp(drift + norm.ppf(delta) * diffusion)

	if type_option == "call":
		payoff_final = np.maximum(grille - K, 0)
	elif type_option == "put":
		payoff_final = np.maximum(K - grille, 0)		

	facteur_actualisation = np.exp(-r * dt)
	facteur_cumulation = np.exp( r * dt)
	valeur_global = payoff_final

	constant1 = (r - 0.5 * sigma ** 2) * dt
	constant2 = sigma * sqrt(dt)



	# calcul des tableaux de transition
	T1_k = np.zeros((p,p+1))
	T2_k = np.zeros((p,p+1))
	# ak
	matrice_grille = np.repeat(np.reshape(grille,(p+1,1)),p+1,axis=1)
	# ai
	matrice_grille2 = np.repeat(np.reshape(grille,(1,p+1)),p+1,axis=0)

	d1 = (np.log(matrice_grille2[1:,1:] / matrice_grille[1:,1:]) - constant1) / constant2
	phi1 = norm.cdf(d1)
	T1_k = np.concatenate((phi1[:,0].reshape(p,1), phi1[:,1:] - phi1[:,:-1], (1-phi1[:,-1]).reshape(p,1)),axis=1)

	d2 = d1 - constant2
	phi2 = norm.cdf(d2)
	T2_k = (facteur_cumulation * matrice_grille[1:,:]
		* np.concatenate((phi2[:,0].reshape(p,1), phi2[:,1:] - phi2[:,:-1], (1-phi2[-1]).reshape(p,1)),axis=1)
		)



	for i in range(N):
		# interpolation linéaire par morceaux
		beta = np.diff(valeur_global[1:]) / np.diff(grille[1:])
		beta = np.append(np.append(beta[0],beta),beta[-1])

		alpha = ((grille[2:] * valeur_global[1:-1] - grille[1:-1] * valeur_global[2:])
			/ np.diff(grille[1:])
			)
		alpha = np.append(np.append(alpha[0],alpha),alpha[-1])
		# on suppose quand Xt = 0, valeur_detention = 0 
		valeur_detention = 	np.append(0,facteur_actualisation * np.sum(alpha * T1_k + beta * T2_k, axis=1))
		# valeur_detention = np.zeros(p+1)
		# for j in range(1,p+1):
		# 	# taille p - 1 
		# 	d1 = (np.log(grille[1:] / grille[j]) - constant1) / constant2
		# 	phi1 = norm.cdf(d1)
		# 	# phi1_0 = phi1_1 
		# 	phi1 = np.append(phi1[0],phi1)
		# 	# taille p + 1
		# 	T1_k = np.append(np.append(phi1[0],phi1[2:] - phi1[1:-1]),1-phi1[-1])

		# 	d2 = d1 - constant2
		# 	phi2 = norm.cdf(d2)
		# 	phi2 = np.append(phi2[0],phi2)

		# 	T2_k = (facteur_cumulation * grille[j]
		# 		* np.append(np.append(phi2[0],phi2[2:] - phi2[1:-1]),1-phi2[-1])
		# 		)
		# 	# calcul de la valeur de détention 
		# 	valeur_detention[j] = facteur_actualisation * np.sum(alpha * T1_k + beta * T2_k)

		# calcul de la valeur global de l'option
		valeur_global = np.maximum(payoff_final, valeur_detention)


	# Calcul du prix de l'option
	# argmax -- first occuremce is returned 
	index = np.argmax(grille >= X0)
	if grille[index] == X0:
		prix = valeur_detention[index]
	else:
		beta_X0 = (valeur_detention[index] - valeur_detention[index-1]) / (grille[index] - grille[index-1])
		alpha_X0 = valeur_detention[index] - beta_X0 * grille[index]
    
		prix = alpha_X0 + beta_X0 * X0
	temps = time.perf_counter() - debut


	return (prix, temps)

def impression_outputs(inputs, outputs):
	"""
	Génération du fichier output
	"""
	nom_de_fichier = "Outputs.xlsx"
	
	# Ajout des résultats
	liste_valeurs = []
	liste_temps = []
	for value in outputs.values():
		liste_valeurs.append(round(value[0],5))
		liste_temps.append(round(value[1],2))


	# Création des dataframes
	df_file = pd.DataFrame(["Tarification de l'option Bermudéenne"])
	df_date = pd.DataFrame([datetime.datetime.now()])
	df_inputs = pd.DataFrame(
		{"Inputs": [key for key in inputs],
		"Valeurs": [value for value in inputs.values()]}
		)
	df_outputs = pd.DataFrame(
		{"Resultats": [key for key in outputs],
		"Valeurs": liste_valeurs,
		"Temps d'exécution (en secondes)": liste_temps}
		)

	# Création de l'objet excel
	writer = pd.ExcelWriter(nom_de_fichier, 
		engine = "xlsxwriter",
		datetime_format = 'mmm d yyyy hh:mm')
	workbook = writer.book
	worksheet = workbook.add_worksheet("Outputs")
	writer.sheets["Outputs"] = worksheet

	# Ajustement de la taille des colonnes 
	worksheet.set_column("A:E", 30)

	# Écriture des dataframes
	df_file.to_excel(
		writer,
		sheet_name = "Outputs",
		startrow = 0,
		startcol = 0,
		index = False,
		header = None
		)
	df_date.to_excel(
		writer,
		sheet_name = "Outputs",
		startrow = 0,
		startcol = 1,
		index = False,
		header = None
		)
	df_inputs.to_excel(
		writer,
		sheet_name = "Outputs",
		startrow = 2,
		startcol = 0,
		index = False
		)
	df_outputs.to_excel(
		writer,
		sheet_name = "Outputs",
		startrow = 13,
		startcol = 0,
		index = False
		)

	# Sauvegarde du fichier
	writer.save()
	writer.close()

	


if __name__ == "__main__":
	inputs = lecture_inputs("Inputs.xlsx") 
	resultats = {}
	resultats['programmation dynamique'] = programmation_dynamique(inputs)
	impression_outputs(inputs, resultats)

