import sys
import os
import time
import pickle

# On suppose que moteur_recherche.py est à la racine, donc on l'ajoute au path si besoin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import DOSSIER_SOUS_TITRES, CACHE_FILE

try:
    from moteur_recherche import Moteur, charger_sous_titres
except ImportError:
    print("FATAL: Le fichier moteur_recherche.py est introuvable.", file=sys.stderr)
    sys.exit(1)

# Variables globales du moteur
moteur = None
systeme_reco = None

def initialiser_moteur():
    """Charge le moteur depuis le cache ou le reconstruit."""
    global moteur, systeme_reco
    
    print(" Initialisation du moteur de recherche...")
    
    if os.path.exists(CACHE_FILE):
        start_time = time.time()
        print(" Chargement depuis le cache...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                moteur = cache_data['moteur']
                systeme_reco = cache_data['systeme_reco']
            return
        except Exception as e:
            print(f" Erreur cache: {e}. Re-création forcée.", file=sys.stderr)
            try: os.remove(CACHE_FILE)
            except OSError: pass
    
    start_time = time.time()
    print("Première initialisation (chargement des sous-titres et TF-IDF)...")
    corpus = charger_sous_titres(DOSSIER_SOUS_TITRES)
    if not corpus:
        print(f" FATAL: Aucun sous-titre chargé depuis {DOSSIER_SOUS_TITRES}.", file=sys.stderr)
        sys.exit(1)
    
    moteur = Moteur(corpus)
    systeme_reco = moteur
    
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'moteur': moteur, 'systeme_reco': systeme_reco}, f)
        print(f" Moteur initialisé et mis en cache en {time.time() - start_time:.2f}s!")
    except Exception as e:
        print(f" Attention: Impossible de sauvegarder le cache: {e}", file=sys.stderr)

def get_moteur():
    return moteur

def get_systeme_reco():
    return systeme_reco