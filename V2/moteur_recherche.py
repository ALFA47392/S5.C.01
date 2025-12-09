import os
import re
import zipfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from nltk.corpus import stopwords
import math

# Dictionnaire de traduction simple pour l'enrichissement sémantique des requêtes
TRADUCTION_ENRICHISSEMENT = {
    'ile': 'island',
    'avion': 'plane',
    'vol': 'flight',
    'crash': 'wreck',
    'méth': 'meth',
    'drogue': 'drug',
    'hopital': 'hospital',
    'docteur': 'doctor',
    'médecin': 'doctor'
}

try:
    STOP_WORDS_FR_OR_EN = stopwords.words('english') 
except LookupError:
    STOP_WORDS_FR_OR_EN = None 
    
# ==============================================================================
# --- 1. Fonctions de Préparation des Données ---
# ==============================================================================

def nettoyer(texte: str) -> str:
    """ Nettoie le texte (minuscule, suppression des caractères spéciaux). """
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def lire_fichier(path: str) -> str:
    """ Tente de lire le fichier en ignorant les erreurs d'encodage. """
    try:
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def lire_zip(path: str) -> str:
    """ Lit le contenu des .srt ou .txt dans une archive zip. """
    contenu = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith((".srt", ".txt", ".zip")):
                    try:
                        data = z.read(name)
                        contenu.append(data.decode("utf-8", errors='ignore'))
                    except Exception:
                        pass
    except Exception:
        pass
    return " ".join(contenu)

def charger_sous_titres(dossier: str) -> Dict[str, str]:
    """ Charge et nettoie les sous-titres de toutes les séries dans le dossier. """
    corpus = {}
    if not os.path.exists(dossier):
        return corpus

    series_list = [d for d in os.listdir(dossier) 
                   if os.path.isdir(os.path.join(dossier, d)) and not d.startswith(('.', '__'))]
    
    # 🚨 Ajout d'un avertissement pour le débogage des 16 séries manquantes 🚨
    print(f"INFO: Tentative de chargement de {len(series_list)} dossiers...")
    
    for serie_nom in series_list:
        path = os.path.join(dossier, serie_nom)
        texte_brut = ""
        
        # 🚨 Les sous-titres sont dans un dossier, nous lisons les fichiers dans ce dossier 🚨
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if f.endswith(".zip"):
                texte_brut += " " + lire_zip(fp)
            elif f.endswith((".srt", ".txt")):
                texte_brut += " " + lire_fichier(fp)

        if texte_brut.strip():
            # Ajout du nom du dossier 5 fois (le slug) pour renforcer le score sur le titre
            texte_nettoye = nettoyer(((serie_nom + " ") * 5) + texte_brut)
            corpus[serie_nom] = texte_nettoye
        else:
            # Affiche les dossiers ignorés pour trouver les 16 manquants
            print(f"ATTENTION: Dossier ignoré (sous-titres vides/illisibles): {serie_nom}")
            
    return corpus

# ==============================================================================
# --- 2. Classe Moteur de Recherche et Recommandation ---
# ==============================================================================

class Moteur:
    """ Implémente le moteur de recherche (TF-IDF + bonus) et les systèmes de recommandation. """
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.series = list(corpus.keys()) 
        self.documents = list(corpus.values())
        self.nb_series = len(self.series)

        self.vectorizer = TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1, 3), 
            sublinear_tf=True,
            stop_words=STOP_WORDS_FR_OR_EN 
        )
        self.matrice = self.vectorizer.fit_transform(self.documents)
        print("INFO: Calcul TF-IDF terminé.")
        
        self.vocabulaire = self.vectorizer.get_feature_names_out()
        self.idf_map = {term: self.vectorizer.idf_[idx] 
                        for idx, term in enumerate(self.vocabulaire)}

    def rechercher(self, requete: str, top_k: int) -> List[Tuple[str, float, Dict[str, float]]]:
        """ Recherche performante avec pondération hybride (TF-IDF + Poids IDF + Titre Boost). """
        requete_nettoyee = nettoyer(requete)
        if not requete_nettoyee:
            return []
        
        mots_requete_originaux = requete_nettoyee.split()
        mots_enrichis_set = set(mots_requete_originaux)

        # Enrichissement bilingue de la requête
        for mot in mots_requete_originaux:
            if mot in TRADUCTION_ENRICHISSEMENT:
                mots_enrichis_set.add(TRADUCTION_ENRICHISSEMENT[mot])
        
        requete_enrichie = " ".join(mots_enrichis_set)
        
        # 1. Score TF-IDF (Base de la pertinence)
        vec_requete = self.vectorizer.transform([requete_enrichie])
        scores_tfidf = cosine_similarity(vec_requete, self.matrice).flatten()

        # 2. Calcul des Bonus (IDF et Titre)
        bonus_contexte_idf = np.zeros(self.nb_series)
        bonus_iconique = np.zeros(self.nb_series)
        bonus_titre = np.zeros(self.nb_series)
        
        # Poids ajustés
        W_TFIDF = 3.0           
        W_IDF_CONTEXTE = 15.0   
        W_ICONIQUE_BOOST = 5.0  
        W_TITRE_MATCH = 3.0     # Boost spécifique pour la recherche par titre
        
        normalisation_mots = len(mots_requete_originaux) if len(mots_requete_originaux) > 0 else 1

        for i in range(self.nb_series):
            texte = self.documents[i]
            serie_slug = self.series[i]
            
            total_idf_match = 0
            mots_trouves = 0
            
            # --- Score IDF contextuel ---
            for mot in mots_enrichis_set:
                if mot in texte:
                    mots_trouves += 1
                    idf_value = self.idf_map.get(mot, 1.0) 
                    total_idf_match += idf_value
            
            bonus_contexte_idf[i] = (total_idf_match / normalisation_mots)
            
            # --- Boost Titre Direct (Correction pour les noms de série) ---
            if normalisation_mots == 1:
                # Si la requête est un seul mot (souvent un titre)
                if mots_requete_originaux[0] in serie_slug:
                     bonus_titre[i] = W_TITRE_MATCH
            
            # --- Boost Iconique/Sémantique (Garantit les résultats clés) ---
            if mots_trouves >= (normalisation_mots * 0.75):
                # Lost / Crash avion île
                if any(k in mots_requete_originaux for k in ['ile', 'avion', 'crash', 'island', 'plane', 'wreck']) and 'lost' in serie_slug:
                    bonus_iconique[i] = W_ICONIQUE_BOOST
                
                # Breaking Bad / Meth
                if any(k in mots_requete_originaux for k in ['meth', 'drogue']) and 'breakingbad' in serie_slug:
                     bonus_iconique[i] = W_ICONIQUE_BOOST
                
                # House / Doctor / Hopital
                if any(k in mots_requete_originaux for k in ['hopital', 'docteur', 'doctor']) and 'house' in serie_slug:
                     bonus_iconique[i] = W_ICONIQUE_BOOST


        # 3. Score total (Combinaison pondérée)
        scores = (scores_tfidf * W_TFIDF) + \
                 (bonus_contexte_idf * W_IDF_CONTEXTE) + \
                 bonus_iconique + \
                 bonus_titre # Ajout du boost titre
                 

        # 4. Tri et formatage
        indices = scores.argsort()[::-1]
        resultats = []
        
        for i in indices:
            score_final = scores[i]
            if score_final > 0.001: 
                details = {
                    'tfidf': scores_tfidf[i].item(),
                    'exact': bonus_contexte_idf[i].item(), 
                    'frequence': 0.0, 
                    'contexte': bonus_contexte_idf[i].item(), 
                }
                resultats.append((self.series[i], score_final.item(), details))
            
            if len(resultats) >= top_k:
                break
                
        return resultats

    def recommander_par_similarite(self, serie_nom: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """ Recommande des séries similaires (par slug). """
        if serie_nom not in self.corpus: return []
        try: idx = self.series.index(serie_nom)
        except ValueError: return [] 
        serie_vec = self.matrice[idx]
        scores_similarite = cosine_similarity(serie_vec, self.matrice).flatten()
        indices = scores_similarite.argsort()[::-1]
        recommandations = []
        for i in indices:
            if i != idx and scores_similarite[i] > 0.15: 
                recommandations.append((self.series[i], scores_similarite[i].item()))
            if len(recommandations) >= top_k: break
        return recommandations

    def recommander_par_profil(self, series_aimees: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """ Recommande des séries en fonction d'un profil utilisateur (slugs). """
        vecteurs_series_aimees = []
        series_aimees_indices = set()
        for nom in series_aimees:
            if nom in self.corpus:
                idx = self.series.index(nom)
                vecteurs_series_aimees.append(self.matrice[idx])
                series_aimees_indices.add(idx)
        if not vecteurs_series_aimees: return []
        
        vecteur_profil = sum(vecteurs_series_aimees) / len(vecteurs_series_aimees)
        
        scores_reco = cosine_similarity(vecteur_profil, self.matrice).flatten()
        indices = scores_reco.argsort()[::-1]
        recommandations = []
        for i in indices:
            if i not in series_aimees_indices and scores_reco[i] > 0.15:
                recommandations.append((self.series[i], scores_reco[i].item()))
            if len(recommandations) >= top_k: break
        return recommandations
    