// --- 2_ui.js : Interface Utilisateur et Rendu ---

// Met à jour la barre de statistiques
function updateStats(nbResultats = 0, tempsMs = '-', nbSeries = null) {
    document.getElementById('nbResultats').textContent = nbResultats;
    document.getElementById('tempsRecherche').textContent = tempsMs;
    if (nbSeries !== null) {
        document.getElementById('nbSeries').textContent = nbSeries;
    }
}

// Gère la classe active sur les boutons d'onglets
function setActiveButton(tab) {
    document.querySelectorAll('.tab-button').forEach(t => {
        t.classList.remove('active');
        if (t.dataset.tab === tab) {
            t.classList.add('active');
        }
    });
}

// Indicateur de chargement
function afficherChargement(afficher) {
    document.getElementById('loading').classList.toggle('hidden', !afficher);
    if (afficher) {
        document.getElementById('results').innerHTML = ''; 
        document.getElementById('initialMessage').classList.add('hidden');
    }
}

// Gestion des erreurs
function afficherErreur(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
    document.getElementById('results').innerHTML = '';
    document.getElementById('initialMessage').classList.add('hidden');
    updateStats(0, '-');
}

function cacherErreur() {
    document.getElementById('errorMessage').classList.add('hidden');
}

// Gestion de la touche Entrée pour la recherche
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        if (currentTab === 'recherche') rechercher();
    }
}

// --- Logique principale de changement d'onglet ---
function switchTab(tab, forceReload = false) {
    currentTab = tab;
    setActiveButton(tab);
    
    // Masquer tout le contenu des onglets
    document.querySelectorAll('.tab-content').forEach(element => {
        element.classList.add('hidden'); 
    });
    
    document.getElementById('results').innerHTML = '';
    cacherErreur();
    
    const initialMsg = document.getElementById('initialMessage');
    if (initialMsg) {
        initialMsg.classList.add('hidden');
    }

    // Afficher le contenu de l'onglet sélectionné
    const targetInputId = tab + '-input';
    const targetElement = document.getElementById(targetInputId);
    if (targetElement) {
        targetElement.classList.remove('hidden');
    }
    
    // Logique de chargement des données
    if (tab === 'toutes') {
        chargerToutesSeries(); 
    } else if (tab === 'vues') {
        if (!loggedInUserId) {
             afficherErreur("Veuillez vous connecter pour voir vos séries notées.");
        } else {
             chargerSeriesVues();
        }
    } else if (tab === 'profil') {
         if (!loggedInUserId) {
             afficherErreur("Veuillez vous connecter pour obtenir des recommandations.");
         }
    } else if (tab === 'recherche') {
         // Info stats seulement
         fetchAPI(`series`)
            .then(data => updateStats(0, '-', data.nombre_series))
            .catch(() => updateStats(0, '-', 'Erreur'));
    }
    
    if (!forceReload) {
        updateStats(0, '-');
    }
}

// --- Rendu des cartes de séries ---
function afficherResultats(series, scoreKey, isSearch, isVues = false) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; 
    document.getElementById('initialMessage').classList.add('hidden'); 

    if (series.length === 0) {
        resultsDiv.innerHTML = '<p class="text-gray-400 col-span-full text-center p-10">Aucun résultat trouvé.</p>';
        return;
    }

    series.forEach(serie => {
        const card = document.createElement('div');
        card.className = 'card-bg rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition duration-300 transform hover:scale-[1.02] relative border border-gray-700 cursor-pointer';
        card.onclick = () => openDetailsModal(serie); 

        const scoreValue = serie[scoreKey] !== undefined ? serie[scoreKey] : null;
        const displayName = serie.nom || 'Nom Indisponible'; 
        const firstChar = (displayName && displayName !== 'Nom Indisponible') ? displayName.substring(0, 1) : '?';

        let indicatorHTML = '';
        if (isVues || serie.note) {
            indicatorHTML = `<span class="score-badge bg-yellow-500 text-black absolute top-3 right-3 text-sm">⭐ ${serie.note || 'N/A'}</span>`;
        }
        
        const scoreHTML = scoreValue !== null && scoreKey !== 'N/A' && scoreKey !== 'note'
            ? `<span class="score-badge">${scoreKey.toUpperCase().replace('_', ' ')}: ${scoreValue.toFixed(4)}</span>`
            : (isSearch ? `<span class="score-badge bg-gray-600">Score Total: ${serie.score ? serie.score.toFixed(4) : 'N/A'}</span>` : ``);

        const posterHTML = serie.affiche_url 
            ? `<img src="${serie.affiche_url}" alt="${displayName}" class="serie-card-poster object-cover object-center" onerror="this.onerror=null;this.src='https://placehold.co/400x250/21262d/a855f7?text=${displayName.substring(0, 15).replace(/\s/g, '+')}';">`
            : `<div class="serie-card-poster flex items-center justify-center bg-gray-600 text-5xl font-bold text-gray-300">${firstChar}</div>`;

        card.innerHTML += `
            ${posterHTML}
            ${indicatorHTML}
            <div class="p-4">
                <div class="text-xl font-bold text-white truncate">${displayName}</div>
                <div class="text-sm text-gray-400 mt-2 flex justify-between items-center">
                    ${scoreHTML}
                    <span class="text-xs text-gray-500">${serie.langue_originale || 'Langue Inconnue'}</span>
                </div>
                <p class="text-gray-500 text-sm mt-3 line-clamp-2">${serie.resume || 'Aucun résumé disponible.'}</p>
            </div>
        `;

        resultsDiv.appendChild(card);
    });
}