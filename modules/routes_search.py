import sys
import time
from flask import Blueprint, request, jsonify
from modules.database import preparer_mapping_bdd
from modules.engine import get_moteur, get_systeme_reco

search_bp = Blueprint('search', __name__)

@search_bp.route('/recherche', methods=['GET'])
def rechercher_series():
    moteur = get_moteur()
    if moteur is None:
        return jsonify({"error": "Le moteur de recherche n'a pas pu être initialisé."}), 503

    try:
        requete = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 20))
        
        if not requete:
            return jsonify({"error": "Paramètre 'q' requis"}), 400
        
        start_search_time = time.time()
        resultats = moteur.rechercher(requete, top_k=limit)
        search_time = time.time() - start_search_time

        bdd_map = preparer_mapping_bdd()
        series_enrichies = []
        
        for serie_slug, score, details in resultats:
            if serie_slug in bdd_map:
                serie_info = bdd_map[serie_slug]
                series_enrichies.append({
                    "id": serie_info['id'],
                    "nom": serie_info['nom'],
                    "resume": serie_info['resume'],
                    "affiche_url": serie_info['affiche_url'],
                    "langue_originale": serie_info['langue_originale'],
                    "score": round(score, 4),
                    "details_score": {
                        "tfidf": round(details.get('tfidf', 0), 4),
                        "couverture": round(details.get('couverture', 0), 4)
                    }
                })
        
        return jsonify({
            "requete": requete,
            "temps_recherche_ms": round(search_time * 1000, 2),
            "nombre_resultats": len(series_enrichies),
            "resultats": series_enrichies
        })
    except Exception as e:
        print(f"Erreur recherche: {e}", file=sys.stderr)
        return jsonify({"error": "Erreur interne du serveur."}), 500

@search_bp.route('/recommandations/similarite', methods=['GET'])
def recommandations_similarite():
    systeme_reco = get_systeme_reco()
    if systeme_reco is None:
        return jsonify({"error": "Le moteur de recommandation non initialisé."}), 503

    try:
        serie_nom_slug = request.args.get('serie', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not serie_nom_slug: return jsonify({"error": "Paramètre 'serie' requis"}), 400
        if serie_nom_slug not in systeme_reco.series: return jsonify({"error": "Série introuvable"}), 404
        
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
            "resultats": series_recommandees
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@search_bp.route('/recommandations/profil', methods=['POST'])
def recommandations_profil():
    systeme_reco = get_systeme_reco()
    if systeme_reco is None: return jsonify({"error": "Moteur non initialisé."}), 503
        
    try:
        data = request.get_json()
        series_aimees = data.get('series_aimees', [])
        limit = data.get('limit', 5)
        
        recommandations = systeme_reco.recommander_par_profil(series_aimees, top_k=limit)
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
                    "score_profil": round(score, 4)
                })
        
        return jsonify({"recommandations": series_recommandees})
    except Exception as e:
        return jsonify({"error": str(e)}), 500