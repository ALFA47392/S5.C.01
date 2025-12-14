-- 1. Création de la table 'series'
-- Stocke les informations générales sur chaque série.
CREATE TABLE IF NOT EXISTS series (
    id SERIAL PRIMARY KEY,
    nom VARCHAR(255) NOT NULL UNIQUE,
    resume TEXT,
    affiche_url TEXT,
    langue_originale VARCHAR(50)
);

-- 2. Création de la table 'episodes'
-- Lie chaque épisode à une série via une clé étrangère.
CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    id_series INTEGER NOT NULL,
    saison INTEGER NOT NULL,
    numero INTEGER NOT NULL,
    FOREIGN KEY (id_series) REFERENCES series(id) ON DELETE CASCADE
);

-- 3. Création de la table 'sous_titres'
-- Stocke le contenu textuel des sous-titres pour chaque épisode.
CREATE TABLE IF NOT EXISTS sous_titres (
    id SERIAL PRIMARY KEY,
    id_episode INTEGER NOT NULL,
    langue VARCHAR(10),
    contenu TEXT,
    FOREIGN KEY (id_episode) REFERENCES episodes(id) ON DELETE CASCADE
);

-- 4. Création de la table 'utilisateurs'
-- Gère les comptes utilisateurs (pseudo, email, mot de passe haché).
CREATE TABLE IF NOT EXISTS utilisateurs (
    id SERIAL PRIMARY KEY,
    pseudo VARCHAR(100),
    email VARCHAR(255) UNIQUE NOT NULL,
    mdp_hash TEXT NOT NULL
);

-- 5. Création de la table 'recommandations' (Notation)
-- Table de liaison pour les notes et commentaires des utilisateurs sur les séries.
CREATE TABLE IF NOT EXISTS recommandations (
    id SERIAL PRIMARY KEY,
    id_utilisateur INTEGER NOT NULL,
    id_series INTEGER NOT NULL,
    note INTEGER CHECK (note >= 1 AND note <= 5),
    commentaire TEXT,
    date_notation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(id_utilisateur, id_series), -- Un utilisateur ne note qu'une fois une série
    FOREIGN KEY (id_utilisateur) REFERENCES utilisateurs(id) ON DELETE CASCADE,
    FOREIGN KEY (id_series) REFERENCES series(id) ON DELETE CASCADE
);