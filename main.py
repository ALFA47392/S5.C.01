from flask import Flask
from flask_cors import CORS

# Imports de nos modules
from modules.database import init_connection_pool, preparer_mapping_bdd
from modules.engine import initialiser_moteur, get_moteur

# Imports des blueprints
from modules.routes_auth import auth_bp
from modules.routes_search import search_bp
from modules.routes_series import series_bp

app = Flask(__name__, static_folder='static')
CORS(app)

# Enregistrement des routes
app.register_blueprint(auth_bp, url_prefix='/api')
app.register_blueprint(search_bp, url_prefix='/api')
app.register_blueprint(series_bp, url_prefix='/api')

if __name__ == '__main__':
    # Initialisation
    init_connection_pool()
    initialiser_moteur()
    preparer_mapping_bdd()

    if get_moteur() is None:
        print("L'API ne peut pas démarrer sans le moteur.")
    else:
        print("\n" + "="*50)
        print(" API MODULAIRE LANCEE")
        print("="*50)
        print(" Accès: http://localhost:5001")
        print("="*50 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5001)