const express = require('express'); // Importer le module Express pour créer le serveur web
const cors = require('cors'); // Importer le module CORS pour gérer les requêtes entre différentes origines
const tf = require('@tensorflow/tfjs-node'); // Importer TensorFlow.js pour Node.js pour manipuler les modèles de machine learning
const fs = require('fs'); // Importer le module File System pour lire et écrire des fichiers
const app = express(); // Créer une application Express
const port = process.env.PORT || 3001; // Définir le port sur lequel le serveur va écouter, par défaut 3001

app.use(cors()); // Utiliser CORS pour permettre les requêtes cross-origin
app.use(express.json()); // Utiliser express.json() pour parser les requêtes JSON

let globalModel = null; // Variable pour stocker le modèle global

// Fonction pour créer un nouveau modèle
const createModel = () => {
    const model = tf.sequential(); // Créer un modèle séquentiel
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [1] })); // Ajouter une couche dense avec 10 unités
    model.add(tf.layers.dense({ units: 1 })); // Ajouter une couche dense avec 1 unité (sortie)
    model.compile({
        optimizer: tf.train.adam(), // Utiliser l'optimiseur Adam
        loss: 'meanSquaredError' // Utiliser l'erreur quadratique moyenne comme fonction de perte
    });
    return model; // Retourner le modèle créé
};

// Fonction pour initialiser le modèle global
const initializeGlobalModel = async () => {
    if (!globalModel) { // Si le modèle global n'existe pas encore
        if (fs.existsSync('./model/model.json')) { // Vérifier si le fichier du modèle existe sur le disque
            globalModel = await tf.loadLayersModel('file://./model/model.json'); // Charger le modèle à partir du fichier
            console.log('Loaded global model from disk.'); // Afficher un message de confirmation
        } else {
            globalModel = createModel(); // Créer un nouveau modèle si aucun modèle n'existe
            console.log('Initialized new global model.'); // Afficher un message de confirmation
        }
    }
};

initializeGlobalModel(); // Appeler la fonction pour initialiser le modèle global

// Route pour mettre à jour le modèle avec de nouveaux poids
app.post('/update-model', async (req, res) => {
    try {
        const weightsData = req.body.weights; // Récupérer les poids envoyés dans la requête
        console.log('Received weights data:', weightsData);

        if (!globalModel) { // Si le modèle global n'existe pas encore
            globalModel = createModel(); // Créer un nouveau modèle
        }

        // Convertir les poids reçus en tenseurs TensorFlow
        const weightTensors = weightsData.map(w => tf.tensor(w.data, w.shape, w.dtype));
        globalModel.setWeights(weightTensors); // Appliquer les nouveaux poids au modèle global
        console.log('Applied new weights to the global model.');

        // Journaliser les nouveaux poids du modèle
        const newWeights = globalModel.getWeights();
        newWeights.forEach((weight, index) => {
            weight.array().then(array => {
                console.log(`Weight ${index}:`, array);
            });
        });

        await globalModel.save('file://./model'); // Sauvegarder le modèle global sur le disque
        console.log('Global model saved to disk.');

        res.json({ message: 'Model updated successfully' }); // Répondre avec un message de succès
    } catch (error) {
        console.error('Error updating the model:', error); // Afficher un message d'erreur en cas de problème
        res.status(500).json({ message: 'Error updating the model', error: error.message }); // Répondre avec un message d'erreur
    }
});

// Route pour récupérer le modèle global
app.get('/global-model', async (req, res) => {
    if (globalModel) { // Si le modèle global existe
        try {
            await globalModel.save('file://./model'); // Sauvegarder le modèle global sur le disque

            // Lire le fichier JSON du modèle et les données de poids
            const modelJSON = JSON.parse(fs.readFileSync('./model/model.json', 'utf8'));
            const weightData = fs.readFileSync('./model/weights.bin');

            // Préparer la réponse avec la topologie du modèle, les spécifications des poids et les données de poids
            const response = {
                modelTopology: modelJSON.modelTopology,
                weightSpecs: modelJSON.weightsManifest[0].weights,
                weightData: Array.from(new Uint8Array(weightData)),
            };

            console.log('Serialized model JSON:', response);
            res.json(response); // Envoyer la réponse au client
            console.log('Global model sent to client.');
        } catch (error) {
            res.status(500).json({ message: 'Error serializing model', error: error.message }); // Répondre avec un message d'erreur en cas de problème
        }
    } else {
        res.status(404).json({ message: 'Global model not found' }); // Répondre avec un message d'erreur si le modèle global n'est pas trouvé
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`); // Démarrer le serveur et afficher le port sur lequel il écoute
});
