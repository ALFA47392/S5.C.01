import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import Counter
import math

# =======================
# 🔹 1. Prétraitement avancé
# =======================

def nettoyer_texte(texte):
    """Nettoyage avancé : minuscules, suppression ponctuation, normalisation."""
    texte = texte.lower()
    # Garder les caractères accentués et espaces
    texte = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", texte)
    # Supprimer les espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def extraire_termes_importants(texte, min_freq=3, max_ratio=0.1):
    """Extrait les termes les plus discriminants d'un texte."""
    mots = texte.split()
    # Filtrer les mots trop courts (moins informatifs)
    mots = [mot for mot in mots if len(mot) > 3]
    
    compteur = Counter(mots)
    total_mots = len(mots)
    
    # Garder les termes avec une fréquence significative mais pas trop élevée
    termes_importants = []
    for mot, freq in compteur.items():
        ratio = freq / total_mots
        if freq >= min_freq and ratio <= max_ratio:
            termes_importants.append(mot)
    
    return termes_importants

def charger_sous_titres(dossier):
    """Charge tous les fichiers de sous-titres avec extraction de termes clés."""
    corpus = {}
    termes_series = {}
    
    for serie in os.listdir(dossier):
        serie_path = os.path.join(dossier, serie)
        if os.path.isdir(serie_path):
            contenu = ""
            for f in os.listdir(serie_path):
                if f.endswith(".srt") or f.endswith(".txt"):
                    with open(os.path.join(serie_path, f), "r", encoding="utf-8", errors="ignore") as file:
                        contenu += " " + file.read()
            
            texte_nettoye = nettoyer_texte(contenu)
            corpus[serie] = texte_nettoye
            termes_series[serie] = extraire_termes_importants(texte_nettoye)
    
    return corpus, termes_series

# =======================
# 🔹 2. Moteur de recherche ultra-précis
# =======================

class MoteurRechercheSeries:
    def __init__(self, corpus, termes_series):
        self.series = list(corpus.keys())
        self.termes_series = termes_series
        
        # Configuration TF-IDF optimisée pour la précision
        french_stopwords = stopwords.words("french")
        
        self.vectorizer = TfidfVectorizer(
            stop_words=french_stopwords,
            max_features=15000,         # Plus de vocabulaire pour la précision
            ngram_range=(1, 3),         # Uni/bi/trigrammes
            min_df=1,                   # Garder même les termes rares
            max_df=0.85,                # Exclure les termes trop fréquents
            sublinear_tf=True,          # Scaling logarithmique
            use_idf=True,               # Importance de IDF
            smooth_idf=True,            # Lissage IDF
            norm='l2'                   # Normalisation L2
        )
        
        self.matrice = self.vectorizer.fit_transform(corpus.values())
        
        # Créer un index inversé pour la recherche rapide
        self._creer_index_inverse()
    
    def _creer_index_inverse(self):
        """Crée un index inversé terme -> [séries] pour optimiser la recherche."""
        self.index_inverse = {}
        for i, serie in enumerate(self.series):
            termes = self.termes_series[serie]
            for terme in termes:
                if terme not in self.index_inverse:
                    self.index_inverse[terme] = []
                self.index_inverse[terme].append(i)
    
    def _calculer_score_exact_match(self, requete_mots, serie_idx):
        """Calcule un score basé sur les correspondances exactes de mots."""
        serie = self.series[serie_idx]
        termes_serie = set(self.termes_series[serie])
        requete_set = set(requete_mots)
        
        # Correspondances exactes
        matches_exacts = len(requete_set.intersection(termes_serie))
        
        # Correspondances partielles (inclusion)
        matches_partiels = 0
        for mot_req in requete_mots:
            if len(mot_req) > 3:
                for terme_serie in termes_serie:
                    if mot_req in terme_serie or terme_serie in mot_req:
                        matches_partiels += 0.5
                        break  # Une seule correspondance par mot de requête
        
        # Score normalisé
        score_exact = matches_exacts / len(requete_mots) if requete_mots else 0
        score_partiel = matches_partiels / len(requete_mots) if requete_mots else 0
        
        return score_exact + (score_partiel * 0.3)
    
    def _calculer_score_densite(self, requete_mots, serie_idx):
        """Calcule un score basé sur la densité des termes dans la série."""
        serie = self.series[serie_idx]
        texte_serie = self.termes_series[serie]
        compteur_serie = Counter(texte_serie)
        
        score_densite = 0
        for mot in requete_mots:
            # Correspondance exacte
            if mot in compteur_serie:
                freq_relative = compteur_serie[mot] / len(texte_serie)
                score_densite += freq_relative
            else:
                # Correspondance partielle
                for terme_serie, freq in compteur_serie.items():
                    if mot in terme_serie or terme_serie in mot:
                        freq_relative = freq / len(texte_serie)
                        score_densite += freq_relative * 0.2
                        break
        
        return score_densite
    
    def _calculer_score_position(self, requete_mots, serie_idx):
        """Bonus si plusieurs mots de la requête apparaissent proches dans le texte."""
        # Simplification : bonus si tous les termes de la requête sont présents
        serie = self.series[serie_idx]
        termes_serie = set(self.termes_series[serie])
        
        termes_trouves = 0
        for mot in requete_mots:
            if any(mot in terme or terme in mot for terme in termes_serie):
                termes_trouves += 1
        
        # Bonus croissant avec le nombre de termes trouvés
        if len(requete_mots) > 1:
            return (termes_trouves / len(requete_mots)) ** 2
        return 0
    
    def rechercher(self, requete, top_k=5):
        """Recherche ultra-précise combinant plusieurs métriques."""
        requete_clean = nettoyer_texte(requete)
        requete_mots = [mot for mot in requete_clean.split() if len(mot) > 2]
        
        if not requete_mots:
            return []
        
        # 1. Score TF-IDF classique
        vecteur_requete = self.vectorizer.transform([requete_clean])
        scores_tfidf = cosine_similarity(vecteur_requete, self.matrice).flatten()
        
        # 2. Pré-filtrage par index inversé pour optimiser
        series_candidates = set()
        for mot in requete_mots:
            if mot in self.index_inverse:
                series_candidates.update(self.index_inverse[mot])
        
        # Si pas de candidats exacts, utiliser toutes les séries
        if not series_candidates:
            series_candidates = set(range(len(self.series)))
        
        # 3. Calcul des scores combinés
        scores_finaux = []
        for i in series_candidates:
            # Score TF-IDF
            score_tfidf = scores_tfidf[i]
            
            # Score correspondances exactes/partielles
            score_exact = self._calculer_score_exact_match(requete_mots, i)
            
            # Score densité
            score_densite = self._calculer_score_densite(requete_mots, i)
            
            # Score position/proximité
            score_position = self._calculer_score_position(requete_mots, i)
            
            # Combinaison pondérée des scores
            score_final = (
                score_tfidf * 0.4 +          # TF-IDF : base solide
                score_exact * 0.35 +         # Correspondances exactes : très important
                score_densite * 0.15 +       # Densité : contexte
                score_position * 0.1         # Position : bonus
            )
            
            scores_finaux.append((i, score_final, {
                'tfidf': score_tfidf,
                'exact': score_exact,
                'densite': score_densite,
                'position': score_position
            }))
        
        # Trier par score final
        scores_finaux.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les top_k résultats
        resultats = []
        for i, (serie_idx, score_final, details) in enumerate(scores_finaux[:top_k]):
            resultats.append((
                self.series[serie_idx], 
                score_final,
                details
            ))
        
        return resultats

# =======================
# 🔹 3. Recommandation (inchangée)
# =======================

class SystemeRecommandation:
    def __init__(self, moteur):
        self.moteur = moteur
        self.similarites = cosine_similarity(moteur.matrice)
        self.notes = {serie: [] for serie in moteur.series}

    def noter(self, serie, note):
        """Ajoute une note utilisateur pour une série (entre 1 et 5)."""
        if serie in self.notes:
            self.notes[serie].append(note)

    def recommander(self, serie_preferee, top_k=5):
        """Recommande des séries similaires à une série donnée."""
        if serie_preferee not in self.moteur.series:
            return []
        idx = self.moteur.series.index(serie_preferee)
        scores = self.similarites[idx]
        indices = np.argsort(scores)[::-1][1:top_k+1]
        return [(self.moteur.series[i], scores[i]) for i in indices]

# =======================
# 🔹 4. Exemple d'utilisation
# =======================

if __name__ == "__main__":
    dossier_sous_titres = "sous-titres"
    
    print("📂 Chargement des sous-titres...")
    corpus, termes_series = charger_sous_titres(dossier_sous_titres)
    
    print("⚡ Indexation TF-IDF ultra-précise...")
    moteur = MoteurRechercheSeries(corpus, termes_series)
    reco = SystemeRecommandation(moteur)
    
    # Test avec l'exemple des consignes
    print("\n🔍 Recherche ultra-précise: 'avion'")
    resultats = moteur.rechercher("avion", top_k=5)
    
    for i, (serie, score_final, details) in enumerate(resultats, 1):
        print(f"{i}. {serie} ({score_final:.4f})")
        print(f"   ├─ TF-IDF: {details['tfidf']:.3f}")
        print(f"   ├─ Exact: {details['exact']:.3f}")  
        print(f"   ├─ Densité: {details['densite']:.3f}")
        print(f"   └─ Position: {details['position']:.3f}")
        print()
    
    # Autres tests
    print("🔍 Recherche: 'meth'")
    resultats = moteur.rechercher("meth", top_k=3)
    for serie, score, _ in resultats:
        print(f"{serie} ({score:.4f})")
    
    print("\n🔍 Recherche: 'zombie apocalypse'")
    resultats = moteur.rechercher("zombie apocalypse", top_k=3)
    for serie, score, _ in resultats:
        print(f"{serie} ({score:.4f})")