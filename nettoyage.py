import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher

# =======================
# 🔹 1. Prétraitement avancé amélioré
# =======================

def nettoyer_texte(texte):
    """Nettoyage avancé optimisé pour la recherche de films."""
    texte = texte.lower()
    # Garder les caractères accentués, espaces et apostrophes (importantes pour les films)
    texte = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s]", " ", texte)
    # Supprimer les espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def extraire_termes_importants(texte, min_freq=2, max_ratio=0.2):
    """Extrait les termes discriminants avec meilleure granularité."""
    mots = texte.split()
    # Filtrer les mots trop courts mais garder certains mots clés courts importants
    mots_utiles = ["cia", "fbi", "usa", "nyc", "web", "app", "gps", "sos", "dna", "tv", "pc"]
    mots = [mot for mot in mots if len(mot) > 3 or mot in mots_utiles]
    
    compteur = Counter(mots)
    total_mots = len(mots)
    
    termes_importants = []
    for mot, freq in compteur.items():
        ratio = freq / total_mots
        # Critères plus flexibles pour capturer plus de variété
        if freq >= min_freq and ratio <= max_ratio:
            termes_importants.append(mot)
    
    return termes_importants

def extraire_mots_cles_specifiques(texte):
    """Extrait des mots-clés spécifiques aux genres et thématiques de films."""
    patterns_genres = {
        'action': r'\b(combat|arme|guerre|militaire|soldat|bataille|explosion|poursuite|mission)\b',
        'romance': r'\b(amour|couple|mariage|divorce|relation|coeur|sentiments|embrasser)\b',
        'thriller': r'\b(mystère|enquête|suspense|danger|menace|tuer|mort|criminel|police|detective)\b',
        'sci-fi': r'\b(robot|alien|espace|futur|technologie|science|expérience|laboratoire|mutation)\b',
        'horror': r'\b(peur|terreur|monstre|sang|vampire|zombie|fantôme|démone|possession|exorcisme)\b',
        'comedy': r'\b(rire|drôle|humour|blague|comédie|amusant|ridicule|bizarre|fou)\b',
        'drama': r'\b(famille|enfant|père|mère|frère|soeur|problème|conflit|émotion|larmes)\b',
        'crime': r'\b(vol|voleur|banque|argent|dealer|drogue|gang|mafia|prison|procès)\b'
    }
    
    mots_cles = []
    for genre, pattern in patterns_genres.items():
        matches = re.findall(pattern, texte, re.IGNORECASE)
        mots_cles.extend(matches)
    
    return list(set(mots_cles))

def charger_sous_titres(dossier):
    """Charge tous les fichiers avec extraction améliorée de métadonnées."""
    corpus = {}
    termes_series = {}
    mots_cles_series = {}
    statistiques_series = {}
    
    for serie in os.listdir(dossier):
        serie_path = os.path.join(dossier, serie)
        if os.path.isdir(serie_path):
            contenu = ""
            nb_fichiers = 0
            
            for f in os.listdir(serie_path):
                if f.endswith(".srt") or f.endswith(".txt"):
                    try:
                        with open(os.path.join(serie_path, f), "r", encoding="utf-8", errors="ignore") as file:
                            contenu_fichier = file.read()
                            contenu += " " + contenu_fichier
                            nb_fichiers += 1
                    except Exception as e:
                        print(f"Erreur lecture {f}: {e}")
            
            if contenu.strip():
                texte_nettoye = nettoyer_texte(contenu)
                corpus[serie] = texte_nettoye
                termes_series[serie] = extraire_termes_importants(texte_nettoye)
                mots_cles_series[serie] = extraire_mots_cles_specifiques(texte_nettoye)
                
                # Statistiques pour pondération
                statistiques_series[serie] = {
                    'nb_fichiers': nb_fichiers,
                    'nb_mots': len(texte_nettoye.split()),
                    'longueur_texte': len(texte_nettoye)
                }
    
    return corpus, termes_series, mots_cles_series, statistiques_series

# =======================
# 🔹 2. Moteur de recherche ultra-précis amélioré
# =======================

class MoteurRechercheSeries:
    def __init__(self, corpus, termes_series, mots_cles_series, statistiques_series):
        self.series = list(corpus.keys())
        self.termes_series = termes_series
        self.mots_cles_series = mots_cles_series
        self.statistiques_series = statistiques_series
        self.corpus = corpus
        
        # Stopwords français étendus
        french_stopwords = set(stopwords.words("french") + 
                             ["alors", "après", "avant", "avec", "avoir", "bien", "chez", "comme", 
                              "donc", "être", "faire", "tout", "tous", "toute", "toutes", "très",
                              "aussi", "autre", "autres", "même", "mêmes", "cette", "cette"])
        
        # Configuration TF-IDF optimisée
        self.vectorizer = TfidfVectorizer(
            stop_words=list(french_stopwords),
            max_features=20000,         # Plus de vocabulaire
            ngram_range=(1, 2),         # Bi-grammes pour capturer les expressions
            min_df=1,                   # Garder les termes rares (important pour les films spécifiques)
            max_df=0.7,                 # Plus strict sur les termes fréquents
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        self.matrice = self.vectorizer.fit_transform(corpus.values())
        self.termes_vocabulaire = self.vectorizer.get_feature_names_out()
        
        # Index inversé amélioré
        self._creer_index_inverse_ameliore()
        
        # Matrice de similarité entre séries (pour le contexte)
        self.similarites_series = cosine_similarity(self.matrice)
    
    def _creer_index_inverse_ameliore(self):
        """Index inversé avec poids et métadonnées."""
        self.index_inverse = defaultdict(list)
        self.index_mots_cles = defaultdict(list)
        
        for i, serie in enumerate(self.series):
            # Index des termes normaux
            for terme in self.termes_series[serie]:
                self.index_inverse[terme].append((i, 1.0))
            
            # Index des mots-clés spécifiques (poids plus élevé)
            for mot_cle in self.mots_cles_series[serie]:
                self.index_mots_cles[mot_cle].append((i, 2.0))
    
    def _calculer_score_exact_match(self, requete_mots, serie_idx):
        """Score de correspondance exacte amélioré."""
        serie = self.series[serie_idx]
        termes_serie = set(self.termes_series[serie])
        mots_cles_serie = set(self.mots_cles_series[serie])
        requete_set = set(requete_mots)
        
        # Correspondances exactes dans les termes
        matches_exacts_termes = len(requete_set.intersection(termes_serie))
        
        # Correspondances exactes dans les mots-clés (bonus important)
        matches_exacts_cles = len(requete_set.intersection(mots_cles_serie))
        
        # Correspondances floues avec ratio de similarité
        matches_flous = 0
        for mot_req in requete_mots:
            if len(mot_req) > 3:
                best_ratio = 0
                for terme_serie in termes_serie:
                    ratio = SequenceMatcher(None, mot_req, terme_serie).ratio()
                    if ratio > 0.8 and ratio > best_ratio:  # Seuil de similarité
                        best_ratio = ratio
                
                if best_ratio > 0:
                    matches_flous += best_ratio * 0.7  # Pondération des matches flous
        
        # Score final normalisé
        if len(requete_mots) > 0:
            score_termes = matches_exacts_termes / len(requete_mots)
            score_cles = (matches_exacts_cles / len(requete_mots)) * 1.5  # Bonus mots-clés
            score_flou = matches_flous / len(requete_mots)
            return score_termes + score_cles + score_flou
        
        return 0
    
    def _calculer_score_frequence_relative(self, requete_mots, serie_idx):
        """Score basé sur la fréquence relative des mots dans la série."""
        serie = self.series[serie_idx]
        texte_serie = self.corpus[serie]
        mots_serie = texte_serie.split()
        compteur_serie = Counter(mots_serie)
        total_mots = len(mots_serie)
        
        score_freq = 0
        for mot in requete_mots:
            # Fréquence exacte
            freq_exacte = compteur_serie.get(mot, 0)
            if freq_exacte > 0:
                # Score basé sur TF mais avec normalisation logarithmique
                tf = freq_exacte / total_mots
                score_freq += math.log(1 + tf * 1000)  # Amplification logarithmique
            
            # Fréquence partielle
            else:
                freq_partielle = sum(freq for terme, freq in compteur_serie.items() 
                                   if mot in terme and len(mot) > 3)
                if freq_partielle > 0:
                    tf_partielle = freq_partielle / total_mots
                    score_freq += math.log(1 + tf_partielle * 500) * 0.3
        
        return score_freq
    
    def _calculer_score_densite_contextuelle(self, requete_mots, serie_idx):
        """Score de densité avec prise en compte du contexte."""
        serie = self.series[serie_idx]
        texte_serie = self.corpus[serie].split()
        
        if not texte_serie:
            return 0
        
        # Recherche de séquences de mots proches
        score_contexte = 0
        taille_fenetre = 50  # Fenêtre de contexte
        
        for i, mot_req in enumerate(requete_mots):
            positions = [j for j, mot in enumerate(texte_serie) 
                        if mot == mot_req or (len(mot_req) > 3 and mot_req in mot)]
            
            for pos in positions:
                # Chercher d'autres mots de la requête dans la fenêtre
                debut = max(0, pos - taille_fenetre)
                fin = min(len(texte_serie), pos + taille_fenetre)
                fenetre = texte_serie[debut:fin]
                
                autres_mots_trouves = 0
                for autre_mot in requete_mots:
                    if autre_mot != mot_req and autre_mot in fenetre:
                        autres_mots_trouves += 1
                
                if autres_mots_trouves > 0:
                    # Bonus pour la co-occurrence
                    score_contexte += autres_mots_trouves * 0.5
        
        return score_contexte / len(requete_mots) if requete_mots else 0
    
    def _calculer_score_specificite(self, requete_mots, serie_idx):
        """Score basé sur la spécificité des termes (rareté globale)."""
        score_spec = 0
        
        for mot in requete_mots:
            # Nombre de séries contenant ce mot
            nb_series_avec_mot = sum(1 for s in self.series 
                                   if mot in self.corpus[s])
            
            if nb_series_avec_mot > 0:
                # IDF manuel : plus le mot est rare, plus il a de poids
                idf = math.log(len(self.series) / nb_series_avec_mot)
                
                # Vérifier si le mot est dans cette série
                if mot in self.corpus[self.series[serie_idx]]:
                    score_spec += idf
        
        return score_spec / len(requete_mots) if requete_mots else 0
    
    def rechercher(self, requete, top_k=5, mode_debug=False):
        """Recherche ultra-précise avec scoring multicritères optimisé."""
        requete_clean = nettoyer_texte(requete)
        requete_mots = [mot for mot in requete_clean.split() if len(mot) > 2]
        
        if not requete_mots:
            return []
        
        # 1. Score TF-IDF de base
        vecteur_requete = self.vectorizer.transform([requete_clean])
        scores_tfidf = cosine_similarity(vecteur_requete, self.matrice).flatten()
        
        # 2. Pré-filtrage intelligent
        series_candidates = set()
        poids_candidats = defaultdict(float)
        
        for mot in requete_mots:
            # Candidats directs
            if mot in self.index_inverse:
                for serie_idx, poids in self.index_inverse[mot]:
                    series_candidates.add(serie_idx)
                    poids_candidats[serie_idx] += poids
            
            # Candidats par mots-clés
            if mot in self.index_mots_cles:
                for serie_idx, poids in self.index_mots_cles[mot]:
                    series_candidates.add(serie_idx)
                    poids_candidats[serie_idx] += poids * 1.5  # Bonus mots-clés
        
        # Si pas de candidats directs, utiliser tous (mais avec malus)
        if not series_candidates:
            series_candidates = set(range(len(self.series)))
        
        # 3. Calcul des scores combinés optimisés
        scores_finaux = []
        
        for i in series_candidates:
            # Scores individuels
            score_tfidf = scores_tfidf[i]
            score_exact = self._calculer_score_exact_match(requete_mots, i)
            score_freq = self._calculer_score_frequence_relative(requete_mots, i)
            score_contexte = self._calculer_score_densite_contextuelle(requete_mots, i)
            score_spec = self._calculer_score_specificite(requete_mots, i)
            
            # Bonus pour les candidats pré-filtrés
            bonus_prefiltre = min(poids_candidats[i] * 0.1, 0.3)
            
            # Pondération optimisée pour la précision
            score_final = (
                score_tfidf * 0.25 +          # TF-IDF : base mais moins dominant
                score_exact * 0.35 +          # Correspondances exactes : crucial
                score_freq * 0.20 +           # Fréquence relative : important
                score_contexte * 0.10 +       # Contexte : bonus qualité
                score_spec * 0.10 +           # Spécificité : boost termes rares
                bonus_prefiltre               # Pré-filtrage : léger bonus
            )
            
            scores_finaux.append((i, score_final, {
                'tfidf': score_tfidf,
                'exact': score_exact,
                'frequence': score_freq,
                'contexte': score_contexte,
                'specificite': score_spec,
                'prefiltre': bonus_prefiltre
            }))
        
        # Tri par score final
        scores_finaux.sort(key=lambda x: x[1], reverse=True)
        
        # Retour des résultats
        resultats = []
        for i, (serie_idx, score_final, details) in enumerate(scores_finaux[:top_k]):
            resultats.append((
                self.series[serie_idx], 
                score_final,
                details
            ))
        
        return resultats

# =======================
# 🔹 3. Système de recommandation amélioré
# =======================

class SystemeRecommandation:
    def __init__(self, moteur):
        self.moteur = moteur
        self.similarites = cosine_similarity(moteur.matrice)
        self.notes = {serie: [] for serie in moteur.series}
        self.historique_recherches = []

    def noter(self, serie, note):
        """Ajoute une note utilisateur."""
        if serie in self.notes and 1 <= note <= 5:
            self.notes[serie].append(note)

    def recommander_par_similarite(self, serie_preferee, top_k=5):
        """Recommandations basées sur la similarité textuelle."""
        if serie_preferee not in self.moteur.series:
            return []
        
        idx = self.moteur.series.index(serie_preferee)
        scores = self.similarites[idx]
        
        # Exclure la série elle-même et trier
        indices_scores = [(i, scores[i]) for i in range(len(scores)) if i != idx]
        indices_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(self.moteur.series[i], score) for i, score in indices_scores[:top_k]]

    def recommander_par_profil(self, series_aimees, top_k=5):
        """Recommandations basées sur un profil utilisateur."""
        if not series_aimees:
            return []
        
        # Calculer le profil moyen
        indices_aimees = [self.moteur.series.index(s) for s in series_aimees 
                         if s in self.moteur.series]
        
        if not indices_aimees:
            return []
        
        profil_moyen = np.mean(self.moteur.matrice[indices_aimees], axis=0)
        scores = cosine_similarity(profil_moyen, self.moteur.matrice).flatten()
        
        # Exclure les séries déjà aimées
        indices_scores = [(i, scores[i]) for i in range(len(scores)) 
                         if self.moteur.series[i] not in series_aimees]
        indices_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(self.moteur.series[i], score) for i, score in indices_scores[:top_k]]

# =======================
# 🔹 4. Interface utilisateur améliorée
# =======================

def interface_recherche_interactive():
    """Interface interactive pour tester le moteur."""
    print("🎬 Chargement du moteur de recherche de films...")
    
    try:
        dossier_sous_titres = "sous-titres"
        corpus, termes_series, mots_cles_series, stats_series = charger_sous_titres(dossier_sous_titres)
        
        print(f"📊 {len(corpus)} films chargés avec succès!")
        
        moteur = MoteurRechercheSeries(corpus, termes_series, mots_cles_series, stats_series)
        reco = SystemeRecommandation(moteur)
        
        while True:
            print("\n" + "="*50)
            requete = input("🔍 Recherche (ou 'quit' pour quitter): ").strip()
            
            if requete.lower() in ['quit', 'q', 'exit']:
                break
            
            if not requete:
                continue
            
            print(f"\n📋 Résultats pour '{requete}':")
            resultats = moteur.rechercher(requete, top_k=8)
            
            if not resultats:
                print("❌ Aucun résultat trouvé")
                continue
            
            for i, (film, score, details) in enumerate(resultats, 1):
                print(f"\n{i:2d}. 🎬 {film}")
                print(f"    Score global: {score:.4f}")
                print(f"    ├─ TF-IDF: {details['tfidf']:.3f}")
                print(f"    ├─ Exact: {details['exact']:.3f}")
                print(f"    ├─ Fréq: {details['frequence']:.3f}")
                print(f"    ├─ Contexte: {details['contexte']:.3f}")
                print(f"    ├─ Spécific: {details['specificite']:.3f}")
                print(f"    └─ Préfiltre: {details['prefiltre']:.3f}")
            
            # Option de recommandation
            choix = input("\n💡 Voir des recommandations pour un film? (numéro ou 'n'): ").strip()
            if choix.isdigit() and 1 <= int(choix) <= len(resultats):
                film_choisi = resultats[int(choix)-1][0]
                print(f"\n🎯 Films similaires à '{film_choisi}':")
                recommandations = reco.recommander_par_similarite(film_choisi, top_k=5)
                
                for j, (film_sim, score_sim) in enumerate(recommandations, 1):
                    print(f"{j}. {film_sim} (similarité: {score_sim:.3f})")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")

# =======================
# 🔹 5. Tests et exemples
# =======================

if __name__ == "__main__":
    # Lancement de l'interface interactive
    interface_recherche_interactive()