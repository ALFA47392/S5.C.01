import sys
import time
import re
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from functools import lru_cache
from typing import Dict, Any

from modules.config import DB_CONFIG

# Variables globales pour la BDD
connection_pool = None
bdd_mapping_cache = None
bdd_mapping_timestamp = 0
BDD_CACHE_TTL = 300

def init_connection_pool():
    """Initialise le pool de connexions PostgreSQL."""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            cursor_factory=RealDictCursor,
            **DB_CONFIG
        )
        print(" Pool de connexions PostgreSQL initialisé.")
    except psycopg2.OperationalError as e:
        print(f"FATAL: Impossible de créer le pool de connexions. Détails: {e}", file=sys.stderr)
        sys.exit(1)

def get_db_connection():
    """Récupère une connexion disponible depuis le pool."""
    if connection_pool:
        return connection_pool.getconn()
    raise ConnectionError("Pool de connexions non initialisé.")

def release_db_connection(conn):
    """Libère la connexion vers le pool."""
    if connection_pool:
        connection_pool.putconn(conn)

@lru_cache(maxsize=1)
def aligner_nom_bdd(nom_serie: str) -> str:
    """Convertit un nom de série en slug."""
    if not nom_serie:
        return ""
    nom = nom_serie.lower().replace(' ', '')
    nom = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ]", "", nom) 
    return nom

def preparer_mapping_bdd(force_refresh=False) -> Dict[str, Any]:
    """Récupère le mapping {slug: données_série} depuis la BDD ou cache."""
    global bdd_mapping_cache, bdd_mapping_timestamp
    
    current_time = time.time()
    if not force_refresh and bdd_mapping_cache and (current_time - bdd_mapping_timestamp) < BDD_CACHE_TTL:
        return bdd_mapping_cache
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, nom, resume, affiche_url, langue_originale FROM series")
    all_series_bdd = cur.fetchall()
    cur.close()
    release_db_connection(conn)
    
    bdd_map = {}
    for serie in all_series_bdd:
        slug = aligner_nom_bdd(serie['nom'])
        bdd_map[slug] = serie
    
    bdd_mapping_cache = bdd_map
    bdd_mapping_timestamp = current_time
    return bdd_map