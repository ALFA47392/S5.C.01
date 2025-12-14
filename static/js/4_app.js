// --- 4_app.js : Logique Métier (Search, Reco, Détails) ---

// --- Recherche et Listing ---
async function rechercher() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) {
        afficherErreur('Veuillez entrer un terme de recherche');
        return;
    }
    afficherChargement(true);
    try {
        const data = await fetchAPI(`recherche?q=${encodeURIComponent(query)}&limit=50`); 
        updateStats(data.nombre_resultats, data.temps_recherche_ms);
        afficherResultats(data.resultats, 'score', true); 
    } catch (error) {
    } finally {
        afficherChargement(false);
    }
}

async function recommanderProfil() {
    const userId = getUserId();
    if (!userId) {
         afficherErreur("Veuillez vous connecter pour utiliser la recommandation par profil.");
         return;
    }
    afficherChargement(true);
    try {
        const vuesData = await fetchAPI(`utilisateur/${userId}/series`);
        const seriesAimeesSlugs = vuesData.series.map(s => alignerNomBdd(s.nom));

        if (seriesAimeesSlugs.length === 0) {
             afficherErreur("Vous devez noter au moins une série pour obtenir des recommandations par profil.");
             return;
        }

        const data = await fetchAPI(`recommandations/profil`, {
            method: 'POST',
            body: JSON.stringify({ series_aimees: seriesAimeesSlugs, limit: 10 })
        });
        
        updateStats(data.nombre_recommandations, data.temps_reco_ms);
        afficherResultats(data.recommandations, 'score_profil', false);
    } catch (error) {
    } finally {
        afficherChargement(false);
    }
}

async function chargerSeriesVues() {
    const userId = getUserId();
    if (!userId) return; 
    afficherChargement(true);
    try {
        const data = await fetchAPI(`utilisateur/${userId}/series`);
        updateStats(data.nombre_series, '-');
        afficherResultats(data.series, 'note', false, true); 
    } catch (error) {
    } finally {
        afficherChargement(false);
    }
}

async function chargerToutesSeries() {
    afficherChargement(true);
    try {
        const data = await fetchAPI(`series`);
        updateStats(data.nombre_series, '-');
        afficherResultats(data.series, 'N/A', false); 
    } catch (error) {
    } finally {
        afficherChargement(false);
    }
}

// --- Modale Détails et Notation ---

async function openDetailsModal(serie) {
    selectedSerieId = serie.id;
    
    document.getElementById('detailTitle').textContent = serie.nom || 'Série Inconnue';
    document.getElementById('detailSummary').textContent = serie.resume || 'Aucun résumé disponible.';
    
    const posterUrl = serie.affiche_url || `https://placehold.co/400x250/21262d/a855f7?text=Affiche+Manquante`;
    document.getElementById('detailPoster').src = posterUrl;
    document.getElementById('detailPoster').alt = `Affiche de ${serie.nom}`;

    let scoreText = `Langue Originale: ${serie.langue_originale || 'N/A'}`;
    if (serie.score_similarite) {
        scoreText += `<br>Score Similarité: ${serie.score_similarite.toFixed(4)}`;
    } else if (serie.score_profil) {
        scoreText += `<br>Score Profil: ${serie.score_profil.toFixed(4)}`;
    } else if (serie.score) {
         scoreText += `<br>Score Recherche: ${serie.score.toFixed(4)}`;
    } else if (serie.note) {
         scoreText += `<br>Votre Note: ⭐ ${serie.note} / 5`;
    } else if (serie.note_moyenne) {
         scoreText += `<br>Note Moyenne: ⭐ ${serie.note_moyenne.toFixed(1)} / 5 (${serie.nb_notes} notes)`;
    }
    document.getElementById('detailScore').innerHTML = scoreText;

    const ratingSection = document.getElementById('ratingSection');
    const authPrompt = document.getElementById('authPrompt');
    const currentRatingInfo = document.getElementById('currentRatingInfo');
    const deleteButton = document.getElementById('deleteRatingButton');
    const starContainer = document.getElementById('detailStarContainer');
    
    starContainer.innerHTML = '';
    deleteButton.classList.add('hidden');
    currentRatingInfo.classList.add('hidden');
    
    if (loggedInUserId) {
        ratingSection.classList.remove('hidden');
        authPrompt.classList.add('hidden');
        
        let userNote = serie.note || 0; 
        
        if (userNote === 0 && currentTab !== 'vues') {
            try {
                const noteData = await fetchAPI(`utilisateur/${loggedInUserId}/series/${selectedSerieId}/note`);
                userNote = noteData.note || 0;
            } catch (e) {
                console.error("Impossible de récupérer la note de l'utilisateur:", e.message);
            }
        }
        
        if (userNote > 0) {
            currentRatingInfo.classList.remove('hidden');
            document.getElementById('currentNoteValue').textContent = `${userNote} / 5`;
            deleteButton.classList.remove('hidden');
        }

        selectedRating = userNote;
        for (let i = 1; i <= 5; i++) {
            const star = document.createElement('span');
            star.textContent = '★';
            star.className = `cursor-pointer transition duration-150 ${i <= userNote ? 'text-yellow-400' : 'text-gray-500'}`;
            star.onclick = () => selectDetailRating(i);
            starContainer.appendChild(star);
        }

    } else {
        ratingSection.classList.add('hidden');
        authPrompt.classList.remove('hidden');
    }

    document.getElementById('detailsModal').classList.remove('hidden');
}

function selectDetailRating(rating) {
    selectedRating = rating;
    const stars = document.getElementById('detailStarContainer').children;
    for (let i = 0; i < 5; i++) {
        stars[i].classList.toggle('text-yellow-400', i < rating);
        stars[i].classList.toggle('text-gray-500', i >= rating);
    }
}

function closeDetailsModal() {
    document.getElementById('detailsModal').classList.add('hidden');
    selectedSerieId = null;
    selectedRating = 0;
}

async function deleteRating() {
     const serieIdToDelete = selectedSerieId;
     if (!loggedInUserId || !serieIdToDelete) {
         alert('Erreur critique : ID de série ou utilisateur manquant pour la suppression.');
         return;
     }

     closeDetailsModal();
     afficherChargement(true);

     try {
        await fetchAPI(`utilisateur/${loggedInUserId}/series/${serieIdToDelete}/note`, {
            method: 'DELETE'
        });
     } catch (error) {
         const message = error.message.includes('404') ? "Erreur: Note non trouvée." : error.message;
         afficherErreur('API: ' + message);
     } finally {
         afficherChargement(false);
         switchTab(currentTab, true); 
     }
}

async function submitRating() {
    if (!selectedSerieId || selectedRating === 0 || !loggedInUserId) {
        alert('Veuillez sélectionner une note de 1 à 5 étoiles.');
        return;
    }

    const userId = loggedInUserId;
    const dataToSend = { serie_id: selectedSerieId, note: selectedRating }; 
    
    closeDetailsModal();
    afficherChargement(true);
    
    try {
        await fetchAPI(`utilisateur/${userId}/noter`, { 
            method: 'POST',
            body: JSON.stringify(dataToSend) 
        });
    } catch (error) {
    } finally {
        afficherChargement(false);
        switchTab(currentTab, true);
    }
}