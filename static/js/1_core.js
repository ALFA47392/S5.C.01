// --- 1_core.js : Configuration et Outils de base ---

// Configuration de l'URL de l'API Flask (backend)
const API_URL = 'http://localhost:5001/api';

// État global de l'application (Variables partagées)
let currentTab = 'toutes';
let selectedRating = 0;
let selectedSerieId = null;
let loggedInUserId = null; 
let loggedInPseudo = null;

// --- Fonctions d'Alignement ---
function alignerNomBdd(nomSerie) {
    if (!nomSerie) return "";
    let nom = nomSerie.toLowerCase().replace(/ /g, '');
    nom = nom.replace(/[^a-z0-9àâçéèêëîïôûùüÿñæœ]/g, ''); 
    return nom;
}

// --- Wrapper Fetch pour l'API ---
async function fetchAPI(endpoint, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        ...options
    };
    
    try {
        const response = await fetch(`${API_URL}/${endpoint}`, defaultOptions);
        
        // Gestion de la réponse JSON ou Texte
        const isJson = response.headers.get('content-type')?.includes('application/json');
        const data = isJson ? await response.json() : await response.text();

        if (!response.ok) {
            const message = isJson && data.error ? data.error : `Erreur HTTP ${response.status} sur ${endpoint}`;
            throw new Error(message);
        }

        return data;
    } catch (error) {
        if (endpoint === 'series') {
             // Message spécifique pour l'échec de chargement initial
             afficherErreur(`ÉCHEC DE LA CONNEXION À L'API: Vérifiez que l'API Flask est lancée sur ${API_URL}`);
        } else {
             afficherErreur('API: ' + error.message);
        }
        throw error;
    }
}

// --- Getters Utiles ---
function getUserId() {
    return loggedInUserId;
}