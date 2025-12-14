import sys
import psycopg2
from flask import Blueprint, request, jsonify
from modules.database import get_db_connection, release_db_connection, preparer_mapping_bdd
from modules.engine import get_moteur

series_bp = Blueprint('series', __name__)

@series_bp.route('/series', methods=['GET'])
def lister_series():
    moteur = get_moteur()
    if moteur is None: return jsonify({"error": "Moteur non initialisé."}), 503
    try:
        bdd_map = preparer_mapping_bdd()
        series_enrichies = []
        for serie_slug in moteur.series:
            if serie_slug in bdd_map:
                series_enrichies.append({**bdd_map[serie_slug], "note_moyenne": None})
        return jsonify({"nombre_series": len(series_enrichies), "series": series_enrichies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@series_bp.route('/series/<int:serie_id>', methods=['GET'])
def details_serie(serie_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM series WHERE id = %s", (serie_id,))
        serie = cur.fetchone()
        
        if not serie:
            cur.close(); release_db_connection(conn)
            return jsonify({"error": "Série introuvable"}), 404
        
        cur.execute("SELECT AVG(note) as note_moyenne, COUNT(*) as nb_notes FROM recommandations WHERE id_series = %s", (serie_id,))
        notes_info = cur.fetchone()
        cur.close(); release_db_connection(conn)
        
        return jsonify({
            **serie,
            "note_moyenne": round(float(notes_info['note_moyenne']), 1) if notes_info['note_moyenne'] else None,
            "nb_notes": notes_info['nb_notes']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@series_bp.route('/utilisateur/<int:user_id>/noter', methods=['POST'])
def noter_serie(user_id):
    try:
        data = request.get_json()
        serie_id, note = data.get('serie_id'), data.get('note')
        
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO recommandations (id_utilisateur, id_series, note, commentaire)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_utilisateur, id_series) 
                DO UPDATE SET note = EXCLUDED.note, date_notation = NOW()
            """, (user_id, serie_id, note, data.get('commentaire', '')))
            conn.commit()
            return jsonify({"message": "Note enregistrée"}), 200
        except psycopg2.IntegrityError:
            conn.rollback()
            return jsonify({"error": "Erreur BDD"}), 409
        finally:
            cur.close(); release_db_connection(conn)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@series_bp.route('/utilisateur/<int:user_id>/series', methods=['GET'])
def series_utilisateur(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT s.*, r.note 
            FROM recommandations r JOIN series s ON r.id_series = s.id 
            WHERE r.id_utilisateur = %s ORDER BY r.note DESC
        """, (user_id,))
        series = cur.fetchall()
        cur.close(); release_db_connection(conn)
        return jsonify({"series": series, "nombre_series": len(series)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@series_bp.route('/utilisateur/<int:user_id>/series/<int:serie_id>/note', methods=['DELETE', 'GET'])
def gestion_note(user_id, serie_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if request.method == 'DELETE':
            cur.execute("DELETE FROM recommandations WHERE id_utilisateur = %s AND id_series = %s", (user_id, serie_id))
            deleted = cur.rowcount
            conn.commit()
            cur.close()
            release_db_connection(conn)
            
            if deleted:
                return jsonify({"message": "Supprimé"}), 200 
            else:
                return jsonify({"error": "Pas trouvé"}), 404 
        else: 
            cur.execute("SELECT note FROM recommandations WHERE id_utilisateur = %s AND id_series = %s", (user_id, serie_id))
            note = cur.fetchone()
            cur.close(); release_db_connection(conn)
            return jsonify({"note": note['note'] if note else 0}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500