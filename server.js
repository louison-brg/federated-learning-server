const express = require('express');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

let globalModel = null;

// Function to create a simple model
const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError'
    });
    return model;
};

// Initialize the global model at startup
const initializeGlobalModel = () => {
    if (!globalModel) {
        globalModel = createModel();
        console.log('Initialized global model.');
    }
};

initializeGlobalModel();

const averageWeights = (models) => {
    const modelWeights = models.map(model => model.getWeights());
    const averageWeights = modelWeights[0].map((_, weightIndex) => {
        const sumWeights = modelWeights.reduce((sum, weights) => {
            const weight = weights[weightIndex];
            return sum.add(weight);
        }, tf.zerosLike(modelWeights[0][weightIndex]));
        return sumWeights.div(tf.scalar(models.length));
    });
    return averageWeights;
};

app.post('/update-model', async (req, res) => {
    const weightsData = req.body.weights;

    if (!globalModel) {
        globalModel = createModel();
    }

    const weights = weightsData.map(w => tf.tensor(w, w.shape, w.dtype));
    globalModel.setWeights(weights);

    console.log('Received weights:', weightsData);
    console.log('Global model weights updated.');

    res.json({ message: 'Model updated successfully' });
});

app.get('/global-model', async (req, res) => {
    if (globalModel) {
        try {
            // Properly serialize the model
            const modelJSON = globalModel.toJSON();
            res.json(modelJSON);
            console.log('Global model sent to client.');
        } catch (error) {
            res.status(500).json({ message: 'Error serializing model', error: error.message });
        }
    } else {
        res.status(404).json({ message: 'Global model not found' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
