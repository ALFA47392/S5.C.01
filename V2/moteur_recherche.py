import os
import re
import zipfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from nltk.corpus import stopwords
import math

#  CONFIGURATION SÉMANTIQUE

# Dictionnaire pour enrichir la recherche : permet de trouver des séries même si
# l'utilisateur utilise un terme français alors que la série est en anglais (ou inversement).
# Ex: Si on tape "hopital", on cherchera aussi "hospital" et "doctor".
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

# Tentative de chargement des "stop words" (mots vides comme "the", "a", "is")
# pour éviter qu'ils ne polluent l'analyse TF-IDF.
try:
    STOP_WORDS_FR_OR_EN = stopwords.words('english') 
except LookupError:
    # Si NLTK n'est pas installé ou les données manquantes, on désactive le filtre
    STOP_WORDS_FR_OR_EN = None 
    
#  1. FONCTIONS DE PRÉPARATION ET NETTOYAGE

def nettoyer(texte: str) -> str:
    """
    Standardise le texte pour faciliter l'indexation :
    - Passage en minuscules
    - Suppression de la ponctuation et des caractères spéciaux (sauf alphanumériques)
    - Réduction des espaces multiples
    """
    texte = texte.lower()
    # Regex : ne garde que les lettres (y compris accentuées), chiffres et espaces
    texte = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def lire_fichier(path: str) -> str:
    """Lit un fichier texte brut en ignorant les erreurs d'encodage (ex: caractères corrompus)."""
    try:
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def lire_zip(path: str) -> str:
    """
    Extrait et lit le contenu des fichiers .srt ou .txt directement depuis une archive ZIP
    sans avoir besoin de la décompresser sur le disque.
    """
    contenu = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                # On ne traite que les fichiers de sous-titres ou texte
                if name.lower().endswith((".srt", ".txt", ".zip")):
                    try:
                        data = z.read(name)
                        # Décodage binaire -> chaîne de caractères
                        contenu.append(data.decode("utf-8", errors='ignore'))
                    except Exception:
                        pass
    except Exception:
        pass
    return " ".join(contenu)

def charger_sous_titres(dossier: str) -> Dict[str, str]:
    """
    Parcourt l'arborescence des dossiers pour charger le corpus complet.
    Retourne un dictionnaire { 'nom_serie_slug': 'texte_complet_nettoyé' }.
    """
    corpus = {}
    if not os.path.exists(dossier):
        return corpus

    # Liste des dossiers (chaque dossier correspond à une série)
    series_list = [d for d in os.listdir(dossier) 
                   if os.path.isdir(os.path.join(dossier, d)) and not d.startswith(('.', '__'))]
    
    print(f"INFO: Tentative de chargement de {len(series_list)} dossiers...")
    
    for serie_nom in series_list:
        path = os.path.join(dossier, serie_nom)
        texte_brut = ""
        
        # Lecture de tous les fichiers (zip ou texte) dans le dossier de la série
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if f.endswith(".zip"):
                texte_brut += " " + lire_zip(fp)
            elif f.endswith((".srt", ".txt")):
                texte_brut += " " + lire_fichier(fp)

        if texte_brut.strip():
            texte_nettoye = nettoyer(((serie_nom + " ") * 5) + texte_brut)
            corpus[serie_nom] = texte_nettoye
        else:
            print(f"ATTENTION: Dossier ignoré (sous-titres vides/illisibles): {serie_nom}")
            
    return corpus

#  2. MOTEUR DE RECHERCHE (TF-IDF & SCORING)

class Moteur:
    """
    Classe principale gérant l'indexation TF-IDF et la logique de recherche/recommandation.
    """
    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.series = list(corpus.keys()) # Liste des IDs/Slugs des séries
        self.documents = list(corpus.values()) # Contenu textuel associé
        self.nb_series = len(self.series)

        # Initialisation du vectoriseur TF-IDF
        # - ngram_range=(1, 3) : Prend en compte les mots seuls, les paires et les triplets ("new york city")
        # - sublinear_tf=True : Applique log(1+tf) pour éviter qu'un mot répété 1000 fois n'écrase tout.
        self.vectorizer = TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1, 3), 
            sublinear_tf=True,
            stop_words=STOP_WORDS_FR_OR_EN 
        )
        
        # Création de la matrice Documents x Termes (étape lourde en calcul)
        self.matrice = self.vectorizer.fit_transform(self.documents)
        print("INFO: Calcul TF-IDF terminé.")
        
        # Mappage pour accéder rapidement au score IDF d'un mot spécifique
        self.vocabulaire = self.vectorizer.get_feature_names_out()
        self.idf_map = {term: self.vectorizer.idf_[idx] 
                        for idx, term in enumerate(self.vocabulaire)}

    def rechercher(self, requete: str, top_k: int) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Algorithme de recherche hybride.
        Combine :
        1. Similarité Cosinus (TF-IDF classique)
        2. Bonus Contextuel (basé sur l'IDF des mots trouvés)
        3. Bonus Iconique (boost manuel pour des mots-clés thématiques)
        4. Bonus Titre (boost si le mot correspond au nom de la série)
        """
        requete_nettoyee = nettoyer(requete)
        if not requete_nettoyee:
            return []
        
        mots_requete_originaux = requete_nettoyee.split()
        mots_enrichis_set = set(mots_requete_originaux)

        # Étape d'enrichissement : on ajoute les traductions/synonymes à la requête
        for mot in mots_requete_originaux:
            if mot in TRADUCTION_ENRICHISSEMENT:
                mots_enrichis_set.add(TRADUCTION_ENRICHISSEMENT[mot])
        
        requete_enrichie = " ".join(mots_enrichis_set)
        
        # 1. Calcul du score de base (TF-IDF Cosine Similarity)
        # Transforme la requête en vecteur et compare avec toute la matrice
        vec_requete = self.vectorizer.transform([requete_enrichie])
        scores_tfidf = cosine_similarity(vec_requete, self.matrice).flatten()

        # Initialisation des tableaux de bonus
        bonus_contexte_idf = np.zeros(self.nb_series)
        bonus_iconique = np.zeros(self.nb_series)
        bonus_titre = np.zeros(self.nb_series)
        
        # Coefficients de pondération (Réglage de la "recette" du moteur)
        W_TFIDF = 3.0           
        W_IDF_CONTEXTE = 15.0   # On privilégie la présence des mots rares
        W_ICONIQUE_BOOST = 5.0  # Boost pour les thèmes forts (drogue, hopital...)
        W_TITRE_MATCH = 3.0     # Boost si on tape le nom de la série
        
        normalisation_mots = len(mots_requete_originaux) if len(mots_requete_originaux) > 0 else 1

        # Boucle d'optimisation des scores (Analyse mot à mot)
        for i in range(self.nb_series):
            texte = self.documents[i]
            serie_slug = self.series[i]
            
            total_idf_match = 0
            mots_trouves = 0
            
            #  Score IDF contextuel 
            # Vérifie si les mots de la requête sont présents et ajoute leur poids IDF (rareté)
            for mot in mots_enrichis_set:
                if mot in texte:
                    mots_trouves += 1
                    idf_value = self.idf_map.get(mot, 1.0) 
                    total_idf_match += idf_value
            
            bonus_contexte_idf[i] = (total_idf_match / normalisation_mots)
            
            #  Boost Titre Direct 
            # Si la requête est courte et correspond au nom de la série, on booste.
            if normalisation_mots == 1:
                if mots_requete_originaux[0] in serie_slug:
                     bonus_titre[i] = W_TITRE_MATCH
            
            #  Boost Iconique/Sémantique 
            # Règles métier pour garantir que des requêtes "cultes" remontent les bonnes séries
            if mots_trouves >= (normalisation_mots * 0.75):
                # Lost / Crash avion île
                if any(k in mots_requete_originaux for k in ['ile', 'avion', 'crash', 'island', 'plane', 'wreck']) and 'lost' in serie_slug:
                    bonus_iconique[i] = W_ICONIQUE_BOOST
                
                # Breaking Bad / Meth / Drogue
                if any(k in mots_requete_originaux for k in ['meth', 'drogue']) and 'breakingbad' in serie_slug:
                     bonus_iconique[i] = W_ICONIQUE_BOOST
                
                # Dr House / Hopital
                if any(k in mots_requete_originaux for k in ['hopital', 'docteur', 'doctor']) and 'house' in serie_slug:
                     bonus_iconique[i] = W_ICONIQUE_BOOST


        # 3. Calcul du score final (Somme pondérée)
        scores = (scores_tfidf * W_TFIDF) + \
                 (bonus_contexte_idf * W_IDF_CONTEXTE) + \
                 bonus_iconique + \
                 bonus_titre

        # 4. Tri décroissant et formatage de la sortie
        indices = scores.argsort()[::-1]
        resultats = []
        
        for i in indices:
            score_final = scores[i]
            # Filtrage du bruit (scores trop faibles)
            if score_final > 0.001: 
                # On retourne les détails pour comprendre pourquoi une série est classée ainsi
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
        """
        Recherche les séries les plus proches sémantiquement d'une série donnée.
        Utilise la distance cosinus entre les vecteurs TF-IDF des séries.
        """
        if serie_nom not in self.corpus: return []
        
        try: 
            idx = self.series.index(serie_nom)
        except ValueError: 
            return [] 
            
        serie_vec = self.matrice[idx]
        
        # Calcul de similarité entre la série cible et TOUTES les autres
        scores_similarite = cosine_similarity(serie_vec, self.matrice).flatten()
        
        # Tri des résultats
        indices = scores_similarite.argsort()[::-1]
        recommandations = []
        
        for i in indices:
            # On exclut la série elle-même (similarité de 1.0) et les résultats trop faibles
            if i != idx and scores_similarite[i] > 0.15: 
                recommandations.append((self.series[i], scores_similarite[i].item()))
            if len(recommandations) >= top_k: break
            
        return recommandations

    def recommander_par_profil(self, series_aimees: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recommandation personnalisée : crée un "vecteur profil" moyen à partir des séries aimées
        et cherche ce qui s'en rapproche le plus.
        """
        vecteurs_series_aimees = []
        series_aimees_indices = set()
        
        # Récupération des vecteurs des séries aimées
        for nom in series_aimees:
            if nom in self.corpus:
                idx = self.series.index(nom)
                vecteurs_series_aimees.append(self.matrice[idx])
                series_aimees_indices.add(idx)
        
        if not vecteurs_series_aimees: return []
        
        # Calcul du barycentre (profil moyen de l'utilisateur)
        vecteur_profil = sum(vecteurs_series_aimees) / len(vecteurs_series_aimees)
        
        # Comparaison du profil moyen avec tout le corpus
        scores_reco = cosine_similarity(vecteur_profil, self.matrice).flatten()
        indices = scores_reco.argsort()[::-1]
        
        recommandations = []
        for i in indices:
            # On ne recommande pas ce que l'utilisateur a déjà vu/aimé
            if i not in series_aimees_indices and scores_reco[i] > 0.15:
                recommandations.append((self.series[i], scores_reco[i].item()))
            if len(recommandations) >= top_k: break
            
        return recommandations