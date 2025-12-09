import re
import os
import sys
import time
import pickle
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor

# Import des classes du moteur de recherche
try:
    from moteur_recherche import Moteur, charger_sous_titres
except ImportError:
    print("FATAL: Le fichier moteur_recherche.py est introuvable. Veuillez vérifier que le fichier est bien présent.", file=sys.stderr)
    sys.exit(1)


app = Flask(__name__)
CORS(app)

# =======================
# 🔹 Configuration (À AJUSTER)
# =======================

DB_CONFIG = {
    "dbname": "flavien",
    "user": "postgres",
    "password": "flavien",
    "host": "localhost",
    "port": 5432
}

# Chemin absolu vers le dossier 'sous-titres' (VÉRIFIEZ LE CHEMIN FINAL)
DOSSIER_SOUS_TITRES = "/Users/flavien/Library/CloudStorage/OneDrive-Toulouse3/Semestre5/S5.C.01/V2/sous-titres" 
CACHE_FILE = "moteur_cache.pkl"

# =======================
# 🔹 Fonction d'aide pour l'alignement des noms (Slugify)
# =======================

def aligner_nom_bdd(nom_serie: str) -> str:
    """
    Convertit un nom de série de BDD en un format 'slug' pour la comparaison
    aux noms de corpus. Ex: 'Breaking Bad' -> 'breakingbad'
    """
    if not nom_serie:
        return ""
    nom = nom_serie.lower()
    nom = nom.replace(' ', '')
    # Supprime les caractères spéciaux (y compris les apostrophes, tirets, etc.)
    nom = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ]", "", nom)
    return nom

# =======================
# 🔹 Connexion BDD
# =======================

def get_db_connection():
    """Crée une connexion à la base de données avec gestion d'erreurs."""
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.OperationalError as e:
        print(f"FATAL BDD ERROR: Impossible de se connecter à PostgreSQL. Détails: {e}", file=sys.stderr)
        raise ConnectionError("Problème de connexion à la base de données.") from e


# =======================
# 🔹 Initialisation du moteur (avec cache)
# =======================

moteur: Moteur = None
systeme_reco: Moteur = None

def initialiser_moteur():
    """Charge le moteur depuis le cache ou le construit si nécessaire."""
    global moteur, systeme_reco
    
    print("🚀 Initialisation du moteur de recherche...")
    
    # 1. Tente de charger le cache
    if os.path.exists(CACHE_FILE):
        start_time = time.time()
        print("📦 Chargement depuis le cache...")
        try:
            with open(CACHE_FILE, 'rb') as f:
               cache_data = pickle.load(f)
               moteur = cache_data['moteur']
               systeme_reco = cache_data['systeme_reco']
            print(f"✅ Moteur chargé depuis le cache en {time.time() - start_time:.2f}s!")
            return
        except Exception as e:
            print(f"❌ Erreur lors du chargement du cache: {e}. Re-création forcée.", file=sys.stderr)
            try:
                os.remove(CACHE_FILE)
            except OSError:
                pass 
    
    # 2. Construction si pas de cache
    start_time = time.time()
    print("🔄 Première initialisation (chargement des sous-titres et TF-IDF)...")
    corpus = charger_sous_titres(DOSSIER_SOUS_TITRES)
    if not corpus:
        print(f"❌ FATAL: Aucun sous-titre chargé depuis {DOSSIER_SOUS_TITRES}.", file=sys.stderr)
        sys.exit(1)
    
    moteur = Moteur(corpus)
    systeme_reco = moteur 
   
    # Sauvegarde du cache
    try:
        with open(CACHE_FILE, 'wb') as f:
           pickle.dump({'moteur': moteur, 'systeme_reco': systeme_reco}, f)
        print(f"✅ Moteur initialisé et mis en cache en {time.time() - start_time:.2f}s!")
    except Exception as e:
         print(f"❌ Attention: Impossible de sauvegarder le cache. Détails: {e}", file=sys.stderr)

initialiser_moteur()

# =======================
# 🔹 Fonction d'aide pour le mapping BDD
# =======================

def preparer_mapping_bdd() -> Dict[str, Any]:
    """ Récupère toutes les séries de la BDD et les mappe par slug. """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, nom, resume, affiche_url, langue_originale
        FROM series
    """)
    all_series_bdd = cur.fetchall()
    cur.close()
    conn.close()
    
    bdd_map = {}
    for serie in all_series_bdd:
        slug = aligner_nom_bdd(serie['nom'])
        bdd_map[slug] = serie
    return bdd_map


# =======================
# 🔹 1. RECHERCHE DE SÉRIES (TF-IDF + Bonus)
# =======================

@app.route('/api/recherche', methods=['GET'])
def rechercher_series():
    """ Recherche des séries par mots-clés dans les sous-titres. """
    if moteur is None:
        return jsonify({"error": "Le moteur de recherche n'a pas pu être initialisé."}), 503

    try:
        requete = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 20)) # Limite par défaut à 20
        
        if not requete:
            return jsonify({"error": "Paramètre 'q' requis"}), 400
        
        start_search_time = time.time()
        # 1. Recherche avec le moteur (retourne des SLUGS)
        resultats = moteur.rechercher(requete, top_k=limit)
        search_time = time.time() - start_search_time

        # 2. Enrichir avec les infos BDD via mapping rapide
        bdd_map = preparer_mapping_bdd()
        series_enrichies = []
        
        for serie_slug, score, details in resultats:
            if serie_slug in bdd_map:
                serie_info = bdd_map[serie_slug] 

                series_enrichies.append({
                    "id": serie_info['id'],
                    "nom": serie_info['nom'], # NOM COMPLET DE LA BDD (Ex: Breaking Bad)
                    "resume": serie_info['resume'],
                    "affiche_url": serie_info['affiche_url'],
                    "langue_originale": serie_info['langue_originale'],
                    "score": round(score, 4),
"details_score": {
    "tfidf": round(details.get('tfidf', 0), 4),
    "couverture": round(details.get('couverture', 0), 4),
    "densite": round(details.get('densite', 0), 4),
    "proximite": round(details.get('proximite', 0), 4),
    # Anciens noms pour rétrocompatibilité frontend
    "exact": round(details.get('exact', details.get('couverture', 0)), 4),
    "frequence": round(details.get('frequence', details.get('densite', 0)), 4),
    "contexte": round(details.get('contexte', details.get('proximite', 0)), 4)
}
                })
        
        return jsonify({
            "requete": requete,
            "temps_recherche_ms": round(search_time * 1000, 2),
            "nombre_resultats": len(series_enrichies),
            "resultats": series_enrichies
        })
    
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503 
    except Exception as e:
        print(f"Erreur inattendue dans /api/recherche: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur. Détails: {type(e).__name__}"}), 500

# =======================
# 🔹 2. RECOMMANDATIONS
# =======================

@app.route('/api/recommandations/similarite', methods=['GET'])
def recommandations_similarite():
    """ Recommander des séries similaires (par contenu TF-IDF) """
    if systeme_reco is None:
        return jsonify({"error": "Le moteur de recommandation n'a pas pu être initialisé."}), 503

    try:
        serie_nom_slug = request.args.get('serie', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not serie_nom_slug:
            return jsonify({"error": "Paramètre 'serie' requis"}), 400
        
        if serie_nom_slug not in systeme_reco.series:
            return jsonify({"error": f"Série '{serie_nom_slug}' introuvable dans le corpus (essayez un slug comme 'lostvf' ou 'breakingbad')"}), 404
        
        start_reco_time = time.time()
        recommandations = systeme_reco.recommander_par_similarite(serie_nom_slug, top_k=limit)
        reco_time = time.time() - start_reco_time
        
        bdd_map = preparer_mapping_bdd()
        series_recommandees = []
        for nom_slug, score in recommandations:
            if nom_slug in bdd_map:
                serie_info = bdd_map[nom_slug]
                series_recommandees.append({
                    "id": serie_info['id'],
                    "nom": serie_info['nom'],
                    "affiche_url": serie_info['affiche_url'],
                    "resume": serie_info['resume'],
                    "langue_originale": serie_info['langue_originale'],
                    "score_similarite": round(score, 4)
                })

        return jsonify({
            "serie_reference": serie_nom_slug,
            "temps_reco_ms": round(reco_time * 1000, 2),
            "nombre_recommandations": len(series_recommandees),
            "recommandations": series_recommandees
        })
    
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503 
    except Exception as e:
        print(f"Erreur inattendue dans /api/recommandations/similarite: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur. Détails: {type(e).__name__}"}), 500

@app.route('/api/recommandations/profil', methods=['POST'])
def recommandations_profil():
    """ Recommander des séries basées sur un profil utilisateur """
    if systeme_reco is None:
        return jsonify({"error": "Le moteur de recommandation n'a pas pu être initialisé."}), 503
        
    try:
        data = request.get_json()
        series_aimees_slugs = data.get('series_aimees', [])
        limit = data.get('limit', 5)
        
        if not isinstance(series_aimees_slugs, list) or len(series_aimees_slugs) == 0:
            return jsonify({"error": "'series_aimees' doit être une liste non vide de slugs"}), 400
        
        start_reco_time = time.time()
        recommandations = systeme_reco.recommander_par_profil(series_aimees_slugs, top_k=limit)
        reco_time = time.time() - start_reco_time

        bdd_map = preparer_mapping_bdd()
        series_recommandees = []
        for nom_slug, score in recommandations:
            if nom_slug in bdd_map:
                serie_info = bdd_map[nom_slug]
                series_recommandees.append({
                    "id": serie_info['id'],
                    "nom": serie_info['nom'],
                    "affiche_url": serie_info['affiche_url'],
                    "resume": serie_info['resume'],
                    "langue_originale": serie_info['langue_originale'],
                    "score_profil": round(score, 4)
                })
        
        return jsonify({
            "series_reference": series_aimees_slugs,
            "temps_reco_ms": round(reco_time * 1000, 2),
            "nombre_recommandations": len(series_recommandees),
            "recommandations": series_recommandees
        })
    
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503 
    except Exception as e:
        print(f"Erreur inattendue dans /api/recommandations/profil: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur. Détails: {type(e).__name__}"}), 500


# =======================
# 🔹 3. GESTION DES SÉRIES (Listing et Notation)
# =======================

@app.route('/api/series', methods=['GET'])
def lister_series():
    """ Lister toutes les séries disponibles (BDD et Corpus) """
    if moteur is None:
        return jsonify({"error": "Le moteur de recherche n'a pas pu être initialisé."}), 503
        
    try:
        # Récupère le mapping BDD (plus rapide que de faire des requêtes individuelles)
        bdd_map = preparer_mapping_bdd()
        
        # Filtrer uniquement les séries du corpus (qui sont des slugs)
        series_enrichies = []
        for serie_slug in moteur.series:
            if serie_slug in bdd_map:
                series_enrichies.append(bdd_map[serie_slug])
        
        return jsonify({
            "nombre_series": len(series_enrichies),
            "series": series_enrichies
        })
    
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Erreur inattendue dans /api/series: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/series/<int:serie_id>', methods=['GET'])
def details_serie(serie_id):
    """ Obtenir les détails d'une série par ID """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Infos générales
        cur.execute("""
            SELECT id, nom, resume, affiche_url, langue_originale
            FROM series WHERE id = %s
        """, (serie_id,))
        serie = cur.fetchone()
        
        if not serie:
            return jsonify({"error": "Série introuvable"}), 404
        
        # Note moyenne (nécessite la table 'recommandations' ou 'evaluations')
        cur.execute("""
            SELECT AVG(note) as note_moyenne, COUNT(*) as nb_notes
            FROM recommandations WHERE id_series = %s
        """, (serie_id,))
        
        notes_info = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return jsonify({
            **serie,
            "note_moyenne": round(float(notes_info['note_moyenne']), 2) if notes_info['note_moyenne'] else None,
            "nb_notes": notes_info['nb_notes']
        })
    
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Erreur inattendue dans /api/series/<id>: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/utilisateur/<int:user_id>/noter', methods=['POST'])
def noter_serie(user_id):
    """
    Noter une série ou mettre à jour une note existante.
    Body JSON: {"serie_id": 1, "note": 4}
    """
    try:
        data = request.get_json()
        
        if not data or 'serie_id' not in data or 'note' not in data:
            return jsonify({"error": "Champs 'serie_id' et 'note' requis"}), 400
        
        serie_id = data['serie_id']
        note = data['note']
        commentaire = data.get('commentaire', '') # Optionnel
        
        if not (1 <= note <= 5):
            return jsonify({"error": "La note doit être entre 1 et 5"}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Vérifier si une note existe déjà (Méthode propre : SELECT avant INSERT/UPDATE)
        cur.execute("""
            SELECT id FROM recommandations
            WHERE id_utilisateur = %s AND id_series = %s
        """, (user_id, serie_id))
        
        existing = cur.fetchone()
        
        if existing:
            # 2. Mise à jour (UPDATE)
            cur.execute("""
                UPDATE recommandations
                SET note = %s, commentaire = %s
                WHERE id_utilisateur = %s AND id_series = %s
            """, (note, commentaire, user_id, serie_id))
            message = "Note mise à jour avec succès"
        else:
            # 3. Insertion (INSERT)
            cur.execute("""
                INSERT INTO recommandations (id_utilisateur, id_series, note, commentaire)
                VALUES (%s, %s, %s, %s)
            """, (user_id, serie_id, note, commentaire))
            message = "Note enregistrée avec succès"

        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            "message": message,
            "user_id": user_id,
            "serie_id": serie_id,
            "note": note
        })
    
    except psycopg2.IntegrityError as e:
        # 🚨 Cette erreur gère les clés étrangères manquantes 🚨
        conn.rollback() 
        print(f"Erreur d'intégrité de la BDD (Vérifiez les IDs des séries/utilisateurs): {e}", file=sys.stderr)
        return jsonify({"error": "Erreur d'intégrité: La serie ou l'utilisateur n'existe pas."}), 409
        
    except Exception as e:
        print(f"Erreur dans noter_serie: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500
@app.route('/api/utilisateur/<int:user_id>/series', methods=['GET'])
def series_utilisateur(user_id):
    """Obtenir les séries notées par un utilisateur (pour l'onglet 'Séries Vues')"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT s.id, s.nom, s.affiche_url, s.resume, r.note, r.commentaire
            FROM recommandations r
            JOIN series s ON r.id_series = s.id
            WHERE r.id_utilisateur = %s
            ORDER BY r.note DESC
        """, (user_id,))
        
        series_notees = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return jsonify({
            "user_id": user_id,
            "nombre_series": len(series_notees),
            "series": series_notees
        })
    
    except Exception as e:
        print(f"Erreur dans series_utilisateur: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# =======================
# 🔹 4. LANCEMENT
# =======================

if __name__ == '__main__':
    if moteur is None:
        print("L'API ne peut pas démarrer sans une initialisation correcte du moteur.")
    else:
        print("\n" + "="*50)
        print("🎬 API Moteur de Recherche de Séries TV")
        print("="*50)
        print("🌐 API disponible sur http://localhost:5001")
        print("INFO: N'OUBLIEZ PAS DE MODIFIER DB_CONFIG ET LE CHEMIN DES SOUS-TITRES.")
        print("="*50 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5001)
