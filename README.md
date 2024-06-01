# Federated Learning Server

Ce projet est un serveur d'apprentissage fédéré qui permet de mettre à jour et de partager un modèle de machine learning entre plusieurs clients sans partager les données privées des utilisateurs. Le serveur est construit avec Node.js et TensorFlow.js.

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- [Node.js](https://nodejs.org/en/) (version 14 ou supérieure)
- [npm](https://www.npmjs.com/get-npm) (gestionnaire de paquets de Node.js)

## Installation

1. Clonez ce dépôt sur votre machine locale :

    ```sh
    git clone https://github.com/votre-utilisateur/votre-repo.git
    ```

2. Accédez au répertoire du projet :

    ```sh
    cd federated-learning-server
    ```

3. Installez les dépendances nécessaires :

    ```sh
    npm install
    ```

## Utilisation

1. Démarrez le serveur :

    ```sh
    node server.js
    ```

    Le serveur écoutera par défaut sur le port `3001`.

## Fonctionnalités

### Initialiser le modèle global

Le serveur initialise le modèle global soit en chargeant un modèle existant depuis le disque, soit en créant un nouveau modèle si aucun modèle n'existe.

### Mettre à jour le modèle global

Le serveur peut recevoir de nouveaux poids de modèle de la part des clients et les appliquer au modèle global. Ces poids sont ensuite sauvegardés sur le disque.

- **Route**: `/update-model`
- **Méthode**: `POST`
- **Requête**: JSON contenant les poids du modèle
- **Réponse**: Message de succès ou d'erreur

### Récupérer le modèle global

Les clients peuvent récupérer la dernière version du modèle global depuis le serveur.

- **Route**: `/global-model`
- **Méthode**: `GET`
- **Réponse**: JSON contenant la topologie du modèle, les spécifications des poids et les données de poids

});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
