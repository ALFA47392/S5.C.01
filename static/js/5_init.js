// --- 5_init.js : Démarrage de l'Application ---

document.addEventListener('DOMContentLoaded', () => {
    // Vérification de l'état initial
    updateAuthStatus(null); 
    
    // Attache les événements aux boutons d'onglets
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (event) => switchTab(button.dataset.tab));
    });
    
    // Charge l'onglet par défaut (Toutes les séries)
    switchTab('toutes'); 
});