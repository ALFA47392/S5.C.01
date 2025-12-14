// --- 3_auth.js : Gestion Authentification ---

function showLoginForm() {
    document.getElementById('authTitle').textContent = 'Connexion';
    document.getElementById('loginForm').classList.remove('hidden');
    document.getElementById('registerForm').classList.add('hidden');
    document.getElementById('loginError').classList.add('hidden');
    document.getElementById('registerError').classList.add('hidden');
}

function showRegisterForm() {
    document.getElementById('authTitle').textContent = 'Inscription';
    document.getElementById('loginForm').classList.add('hidden');
    document.getElementById('registerForm').classList.remove('hidden');
    document.getElementById('loginError').classList.add('hidden');
    document.getElementById('registerError').classList.add('hidden');
}

function updateAuthStatus(userId, pseudo = null) {
    const loginButton = document.getElementById('loginButton');
    const recoButton = document.getElementById('recoProfilButton');
    const profilMessage = document.getElementById('profilMessage');

    loggedInPseudo = pseudo;
    loggedInUserId = userId;

    if (userId) {
        loginButton.textContent = `${pseudo || 'Utilisateur'} (Déconnexion)`;
        loginButton.classList.remove('bg-indigo-600');
        loginButton.classList.add('bg-gray-700');
        loginButton.onclick = performLogout;
        
        profilMessage.textContent = 'Cliquez pour obtenir vos recommandations personnalisées.';
        profilMessage.classList.remove('text-red-400');
        recoButton.classList.remove('hidden');
    } else {
        loginButton.textContent = 'Se connecter';
        loginButton.classList.add('bg-indigo-600');
        loginButton.classList.remove('bg-gray-700');
        loginButton.onclick = openLoginModal;
        
        profilMessage.textContent = 'Veuillez vous connecter pour obtenir des recommandations par profil.';
        profilMessage.classList.add('text-red-400');
        recoButton.classList.add('hidden');
    }
    switchTab(currentTab, true);
}

function openLoginModal() {
    showLoginForm();
    document.getElementById('loginModal').classList.remove('hidden');
}

function closeLoginModal() {
    document.getElementById('loginModal').classList.add('hidden');
}

async function performLogin() {
    const email = document.getElementById('loginEmailInput').value.trim();
    const password = document.getElementById('loginPasswordInput').value.trim();
    const errorElement = document.getElementById('loginError');
    errorElement.classList.add('hidden');

    if (!email || !password) {
        errorElement.textContent = "Veuillez remplir tous les champs.";
        errorElement.classList.remove('hidden');
        return;
    }

    try {
        const data = await fetchAPI('utilisateur/connexion', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });
        updateAuthStatus(data.user_id, data.pseudo);
        closeLoginModal();
    } catch (error) {
        const message = error.message.replace('API: Error: ', '').replace('API: ', '');
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
    }
}

async function performRegister() {
    const pseudo = document.getElementById('registerPseudoInput').value.trim();
    const email = document.getElementById('registerEmailInput').value.trim();
    const password = document.getElementById('registerPasswordInput').value.trim();
    const errorElement = document.getElementById('registerError');
    errorElement.classList.add('hidden');

    if (!email || !password || password.length < 6) {
        errorElement.textContent = "Email et mot de passe (min 6 caractères) sont requis.";
        errorElement.classList.remove('hidden');
        return;
    }

    try {
        await fetchAPI('utilisateur/inscription', {
            method: 'POST',
            body: JSON.stringify({ pseudo, email, password })
        });
        showLoginForm();
    } catch (error) {
        const message = error.message.replace('API: Error: ', '').replace('API: ', '');
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
    }
}

function performLogout() {
    updateAuthStatus(null);
    switchTab('toutes');
}