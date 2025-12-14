import sys
import bcrypt
import psycopg2
from flask import Blueprint, request, jsonify
from modules.database import get_db_connection, release_db_connection

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/utilisateur/inscription', methods=['POST'])
def inscription():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        pseudo = data.get('pseudo', email.split('@')[0])
        
        if not email or not password:
            return jsonify({"error": "Email et mot de passe requis"}), 400

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
            conn.commit()
            return jsonify({"message": "Inscription réussie", "user_id": user_id, "pseudo": pseudo}), 201
            
        except psycopg2.IntegrityError:
            conn.rollback()
            return jsonify({"error": "Cet email est déjà utilisé."}), 409
        finally:
            cur.close()
            release_db_connection(conn)
            
    except Exception as e:
        print(f"Erreur inscription: {e}", file=sys.stderr)
        return jsonify({"error": "Erreur interne lors de l'inscription."}), 500

@auth_bp.route('/utilisateur/connexion', methods=['POST'])
def connexion():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email et mot de passe requis"}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, mdp_hash, pseudo FROM utilisateurs WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        
        if user and isinstance(user['mdp_hash'], str) and bcrypt.checkpw(password.encode('utf-8'), user['mdp_hash'].encode('utf-8')):
            return jsonify({"message": "Connexion réussie", "user_id": user['id'], "pseudo": user['pseudo']}), 200
        else:
            return jsonify({"error": "Email ou mot de passe incorrect."}), 401
            
    except Exception as e:
        print(f"Erreur connexion fatale: {e}", file=sys.stderr)
        return jsonify({"error": "Erreur interne lors de la connexion."}), 500