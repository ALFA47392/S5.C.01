import re
import os
import sys
import time
import pickle
from typing import Dict, Any, List, Tuple
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import bcrypt # Utilisé pour le hachage sécurisé des mots de passe

# Import des classes du moteur de recherche (TF-IDF et recommandation)
try:
    # On suppose que Moteur et charger_sous_titres sont définis dans moteur_recherche.py
    from moteur_recherche import Moteur, charger_sous_titres
except ImportError:
    # Erreur fatale si le moteur est manquant, l'API ne peut pas fonctionner
    print("FATAL: Le fichier moteur_recherche.py est introuvable.", file=sys.stderr)
    sys.exit(1)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app) # Autorise les requêtes cross-origin (pour le front-end)


# 1. Configuration Globale


# Paramètres de connexion à la base de données PostgreSQL
DB_CONFIG = {
    "dbname": "flavien",
    "user": "postgres",
    "password": "flavien",
    "host": "localhost",
    "port": 5432
}

# Chemins des ressources du projet
DOSSIER_SOUS_TITRES = "/Users/flavien/Library/CloudStorage/OneDrive-Toulouse3/Semestre5/S5.C.01/S5.C.01/V2/sous-titres"
CACHE_FILE = "moteur_cache.pkl" # Cache binaire pour le moteur de recherche


# 2. Connection Pool pour PostgreSQL (Optimisation I/O BDD)


connection_pool = None

def init_connection_pool():
    """
    Initialise un pool de connexions (SimpleConnectionPool) pour éviter l'overhead
    de l'établissement d'une nouvelle connexion pour chaque requête.
    """
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1,  # Minimum de connexions (toujours prêtes)
            10, # Maximum de connexions
            cursor_factory=RealDictCursor, # Permet aux requêtes de retourner des dictionnaires Python
            **DB_CONFIG
        )
        print(" Pool de connexions PostgreSQL initialisé.")
    except psycopg2.OperationalError as e:
        # Sortie du programme si la BDD n'est pas accessible au démarrage
        print(f"FATAL: Impossible de créer le pool de connexions. Détails: {e}", file=sys.stderr)
        sys.exit(1)

def get_db_connection():
    """Récupère une connexion disponible depuis le pool."""
    if connection_pool:
        return connection_pool.getconn()
    raise ConnectionError("Pool de connexions non initialisé.")

def release_db_connection(conn):
    """Libère la connexion vers le pool pour qu'elle soit réutilisée."""
    if connection_pool:
        connection_pool.putconn(conn)


#  3. Cache en mémoire pour le mapping BDD (Optimisation RAM)


bdd_mapping_cache = None
bdd_mapping_timestamp = 0
BDD_CACHE_TTL = 300  # Durée de vie du cache: 5 minutes

@lru_cache(maxsize=1)
def aligner_nom_bdd(nom_serie: str) -> str:
    """Convertit un nom de série en slug (minuscule sans caractères spéciaux) pour l'alignement Moteur/BDD."""
    if not nom_serie:
        return ""
    nom = nom_serie.lower().replace(' ', '')
    # Nettoyage des caractères spéciaux (conforme à la logique front-end/moteur)
    nom = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ]", "", nom) 
    return nom

def preparer_mapping_bdd(force_refresh=False) -> Dict[str, Any]:
    """Récupère le mapping complet {slug: données_série} depuis la BDD ou le cache si valide."""
    global bdd_mapping_cache, bdd_mapping_timestamp
    
    current_time = time.time()
    
    # Vérification du cache : si valide (non expiré selon TTL), on retourne les données en mémoire
    if not force_refresh and bdd_mapping_cache and (current_time - bdd_mapping_timestamp) < BDD_CACHE_TTL:
        return bdd_mapping_cache
    
    # Sinon, reconstruction du cache
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, nom, resume, affiche_url, langue_originale
        FROM series
    """)
    all_series_bdd = cur.fetchall()
    cur.close()
    release_db_connection(conn)
    
    bdd_map = {}
    # Création du dictionnaire {slug: données}
    for serie in all_series_bdd:
        slug = aligner_nom_bdd(serie['nom'])
        bdd_map[slug] = serie
    
    bdd_mapping_cache = bdd_map
    bdd_mapping_timestamp = current_time
    
    return bdd_map


#  4. Initialisation du moteur TF-IDF


moteur: Moteur = None
systeme_reco: Moteur = None

def initialiser_moteur():
    """Charge le moteur depuis le cache binaire (pickle) ou le reconstruit si nécessaire."""
    global moteur, systeme_reco
    
    print(" Initialisation du moteur de recherche...")
    
    # Tenter de charger le moteur depuis le cache existant
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
            # En cas d'échec de lecture du cache, on force la reconstruction
            print(f" Erreur lors du chargement du cache: {e}. Re-création forcée.", file=sys.stderr)
            try:
                os.remove(CACHE_FILE)
            except OSError:
                pass
    
    # Reconstruction complète (opération longue)
    start_time = time.time()
    print("Première initialisation (chargement des sous-titres et TF-IDF)...")
    corpus = charger_sous_titres(DOSSIER_SOUS_TITRES)
    if not corpus:
        print(f" FATAL: Aucun sous-titre chargé depuis {DOSSIER_SOUS_TITRES}.", file=sys.stderr)
        sys.exit(1)
    
    moteur = Moteur(corpus) # Instanciation du moteur (calcul TF-IDF)
    systeme_reco = moteur
    
    # Sauvegarde du moteur dans le cache pour les prochains démarrages
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'moteur': moteur, 'systeme_reco': systeme_reco}, f)
        print(f" Moteur initialisé et mis en cache en {time.time() - start_time:.2f}s!")
    except Exception as e:
        print(f" Attention: Impossible de sauvegarder le cache. Détails: {e}", file=sys.stderr)


#  5. AUTHENTIFICATION (Routes)


@app.route('/api/utilisateur/inscription', methods=['POST'])
def inscription():
    """Route pour enregistrer un nouvel utilisateur avec hachage bcrypt du mot de passe."""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        # Définition du pseudo (par défaut, la partie avant le @)
        pseudo = data.get('pseudo', email.split('@')[0])
        
        if not email or not password:
            return jsonify({"error": "Email et mot de passe requis"}), 400

        # Hachage sécurisé
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO utilisateurs (pseudo, email, mdp_hash)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (pseudo, email, hashed_password.decode('utf-8')))
            
            user_id = cur.fetchone()['id']
            conn.commit() # Validation de la transaction
            
            return jsonify({
                "message": "Inscription réussie",
                "user_id": user_id,
                "pseudo": pseudo
            }), 201 # Code 201 Created
            
        except psycopg2.IntegrityError:
            conn.rollback() # Annulation en cas d'email déjà utilisé (contrainte unique)
            return jsonify({"error": "Cet email est déjà utilisé."}), 409
        finally:
            cur.close()
            release_db_connection(conn) # Très important : libérer la connexion
            
    except Exception as e:
        print(f"Erreur inscription: {e}", file=sys.stderr)
        return jsonify({"error": "Erreur interne lors de l'inscription."}), 500

@app.route('/api/utilisateur/connexion', methods=['POST'])
def connexion():
    """Route pour authentifier l'utilisateur en comparant le mot de passe fourni au hash stocké."""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email et mot de passe requis"}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Récupération du hash
        cur.execute("""
            SELECT id, mdp_hash, pseudo
            FROM utilisateurs
            WHERE email = %s
        """, (email,))
        
        user = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        
        if user:
            stored_hash = user['mdp_hash'] 
            
            # Vérification bcrypt : comparaison sécurisée
            if isinstance(stored_hash, str) and bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                return jsonify({
                    "message": "Connexion réussie",
                    "user_id": user['id'],
                    "pseudo": user['pseudo']
                }), 200
            else:
                return jsonify({"error": "Email ou mot de passe incorrect."}), 401
        else:
            return jsonify({"error": "Email ou mot de passe incorrect."}), 401
            
    except Exception as e:
        print(f"Erreur connexion fatale: {e}", file=sys.stderr)
        return jsonify({"error": "Erreur interne lors de la connexion."}), 500


#  6. RECHERCHE DE SÉRIES


@app.route('/api/recherche', methods=['GET'])
def rechercher_series():
    """Route de recherche principale, interroge le moteur TF-IDF et enrichit les résultats avec la BDD."""
    if moteur is None:
        return jsonify({"error": "Le moteur de recherche n'a pas pu être initialisé."}), 503

    try:
        requete = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 20))
        
        if not requete:
            return jsonify({"error": "Paramètre 'q' requis"}), 400
        
        # 1. Exécution de la recherche sur le moteur (calcul TF-IDF/scores)
        start_search_time = time.time()
        resultats = moteur.rechercher(requete, top_k=limit)
        search_time = time.time() - start_search_time

        # 2. Enrichissement des résultats avec les données stockées en BDD (via cache)
        bdd_map = preparer_mapping_bdd()
        series_enrichies = []
        
        for serie_slug, score, details in resultats:
            if serie_slug in bdd_map:
                serie_info = bdd_map[serie_slug]

                # Formatage du résultat final pour le front-end
                series_enrichies.append({
                    "id": serie_info['id'],
                    "nom": serie_info['nom'],
                    "resume": serie_info['resume'],
                    "affiche_url": serie_info['affiche_url'],
                    "langue_originale": serie_info['langue_originale'],
                    "score": round(score, 4),
                    "details_score": {
                        # Assure l'affichage de tous les critères du moteur
                        "tfidf": round(details.get('tfidf', 0), 4),
                        "couverture": round(details.get('couverture', 0), 4),
                        # ... autres scores du moteur (exact, frequence, contexte)
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
        return jsonify({"error": f"Erreur interne du serveur."}), 500


#  7. RECOMMANDATIONS


@app.route('/api/recommandations/similarite', methods=['GET'])
def recommandations_similarite():
    """Recommande des séries similaires en utilisant la similarité cosinus (TF-IDF) entre les documents."""
    if systeme_reco is None:
        return jsonify({"error": "Le moteur de recommandation n'a pas pu être initialisé."}), 503

    try:
        serie_nom_slug = request.args.get('serie', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not serie_nom_slug:
            return jsonify({"error": "Paramètre 'serie' requis"}), 400
        
        if serie_nom_slug not in systeme_reco.series:
            return jsonify({"error": f"Série '{serie_nom_slug}' introuvable dans le corpus"}), 404
        
        start_reco_time = time.time()
        recommandations = systeme_reco.recommander_par_similarite(serie_nom_slug, top_k=limit)
        reco_time = time.time() - start_reco_time
        
        # Enrichissement BDD
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
    
    except Exception as e:
        print(f"Erreur inattendue dans /api/recommandations/similarite: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur."}), 500

@app.route('/api/recommandations/profil', methods=['POST'])
def recommandations_profil():
    """Recommande des séries en calculant un profil moyen basé sur l'historique de l'utilisateur."""
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

        # Enrichissement BDD
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
    
    except Exception as e:
        print(f"Erreur inattendue dans /api/recommandations/profil: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur."}), 500


#  8. GESTION DES SÉRIES (CRUD et Listes)


@app.route('/api/series', methods=['GET'])
def lister_series():
    """Liste toutes les séries disponibles, enrichies par les métadonnées BDD."""
    if moteur is None:
        return jsonify({"error": "Le moteur de recherche n'a pas pu être initialisé."}), 503
        
    try:
        bdd_map = preparer_mapping_bdd()
        
        series_enrichies = []
        # Joindre les données BDD (via cache) avec les slugs du moteur
        for serie_slug in moteur.series:
            if serie_slug in bdd_map:
                serie_data = bdd_map[serie_slug]
                series_enrichies.append({
                    **serie_data,
                    "note_moyenne": None  
                })
        
        return jsonify({
            "nombre_series": len(series_enrichies),
            "series": series_enrichies
        })
    
    except Exception as e:
        print(f"Erreur inattendue dans /api/series: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/series/<int:serie_id>', methods=['GET'])
def details_serie(serie_id):
    """Obtient les détails d'une série spécifique par ID et calcule sa note moyenne."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Récupération des détails de base
        cur.execute("""
            SELECT id, nom, resume, affiche_url, langue_originale
            FROM series WHERE id = %s
        """, (serie_id,))
        serie = cur.fetchone()
        
        if not serie:
            cur.close()
            release_db_connection(conn)
            return jsonify({"error": "Série introuvable"}), 404
        
        # 2. Calcul de la note moyenne (agrégation)
        cur.execute("""
            SELECT AVG(note) as note_moyenne, COUNT(*) as nb_notes
            FROM recommandations WHERE id_series = %s
        """, (serie_id,))
        
        notes_info = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        
        return jsonify({
            **serie,
            "note_moyenne": round(float(notes_info['note_moyenne']), 1) if notes_info['note_moyenne'] else None,
            "nb_notes": notes_info['nb_notes']
        })
    
    except Exception as e:
        print(f"Erreur inattendue dans /api/series/<id>: {e}", file=sys.stderr)
        return jsonify({"error": f"Erreur interne du serveur."}), 500

@app.route('/api/utilisateur/<int:user_id>/noter', methods=['POST'])
def noter_serie(user_id):
    """Insère ou met à jour une note utilisateur (Upsert) pour une série donnée."""
    try:
        data = request.get_json()
        
        if not data or 'serie_id' not in data or 'note' not in data:
            return jsonify({"error": "Champs 'serie_id' et 'note' requis"}), 400
        
        serie_id = data['serie_id']
        note = data['note']
        commentaire = data.get('commentaire', '')
        
        # Validation de la plage de note (1-5)
        if not (1 <= note <= 5):
            return jsonify({"error": "La note doit être entre 1 et 5"}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Commande INSERT ON CONFLICT pour gérer la mise à jour si la note existe déjà
            cur.execute("""
                INSERT INTO recommandations (id_utilisateur, id_series, note, commentaire)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_utilisateur, id_series) 
                DO UPDATE SET 
                    note = EXCLUDED.note,
                    commentaire = EXCLUDED.commentaire,
                    date_notation = NOW()
            """, (user_id, serie_id, note, commentaire))
            
            message = "Note enregistrée ou mise à jour avec succès"
            conn.commit()

        except psycopg2.IntegrityError as e:
            conn.rollback() # Annuler si l'utilisateur ou la série n'existe pas (FK error)
            return jsonify({"error": "Erreur critique: La série ou l'utilisateur n'existe pas dans la BDD."}), 409
        finally:
            cur.close()
            release_db_connection(conn)
        
        return jsonify({
            "message": message,
            "user_id": user_id,
            "serie_id": serie_id,
            "note": note
        })
    
    except Exception as e:
        print(f"Erreur fatale dans noter_serie: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/utilisateur/<int:user_id>/series', methods=['GET'])
def series_utilisateur(user_id):
    """Obtient la liste des séries notées par un utilisateur spécifique, triées par note."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT s.id, s.nom, s.affiche_url, s.resume, s.langue_originale, r.note, r.commentaire
            FROM recommandations r
            JOIN series s ON r.id_series = s.id
            WHERE r.id_utilisateur = %s
            ORDER BY r.note DESC
        """, (user_id,))
        
        series_notees = cur.fetchall()
        cur.close()
        release_db_connection(conn)
        
        return jsonify({
            "user_id": user_id,
            "nombre_series": len(series_notees),
            "series": series_notees
        })
    
    except Exception as e:
        print(f"Erreur dans series_utilisateur: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/utilisateur/<int:user_id>/series/<int:serie_id>/note', methods=['DELETE'])
def supprimer_note(user_id, serie_id):
    """Supprime la note d'un utilisateur pour une série donnée."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            DELETE FROM recommandations
            WHERE id_utilisateur = %s AND id_series = %s
        """, (user_id, serie_id))
        
        deleted_rows = cur.rowcount
        conn.commit()
        cur.close()
        release_db_connection(conn)

        if deleted_rows > 0:
            return jsonify({"message": "Note supprimée avec succès"}), 200
        else:
            return jsonify({"error": "Note non trouvée ou ID invalide."}), 404
    
    except Exception as e:
        print(f"Erreur suppression note: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/api/utilisateur/<int:user_id>/series/<int:serie_id>/note', methods=['GET'])
def get_user_note(user_id, serie_id):
    """Récupère la note d'un utilisateur spécifique pour une série donnée (utile pour l'affichage en modale)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT note
            FROM recommandations
            WHERE id_utilisateur = %s AND id_series = %s
        """, (user_id, serie_id))
        
        note_info = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        
        # Retourne 0 si aucune note n'est trouvée
        note = note_info['note'] if note_info else 0
        
        return jsonify({"note": note}), 200
        
    except Exception as e:
        print(f"Erreur dans get_user_note: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


#  9. LANCEMENT DE L'API


if __name__ == '__main__':
    # Initialisation des systèmes critiques
    init_connection_pool()
    initialiser_moteur()
    preparer_mapping_bdd()  # Précharge le cache BDD

    # Démarrage conditionnel si le moteur a bien été initialisé
    if moteur is None:
        print("L'API ne peut pas démarrer sans une initialisation correcte du moteur.")
    else:
        print("\n" + "="*50)
        print(" API OK")
        print("="*50)
        print(" API sur http://localhost:5001")
        print(" Connection pooling OK")
        print(" Cache BDD en mémoire OK")
        print("="*50 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5001)