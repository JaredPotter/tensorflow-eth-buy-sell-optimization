const tensorFlow = require('@tensorflow/tfjs-node');
const hpjs = require('hyperparameters');

const optimizers = {
  // sgd: tensorFlow.train.sgd,
  // adagrad: tensorFlow.train.adagrad,
  adam: tensorFlow.train.adam, // "A solid performer - Stefan, the magic man"
  // adamax: tensorFlow.train.adamax,
  // rmsprop: tensorFlow.train.rmsprop,
};

// https://medium.com/@martin_stoyanov/a-simple-example-with-hyperparametersjs-3b36edbe838f

// An optimization function. The parameters are optimizer and epochs and will
// use the loss returned by the fn to measure which parameters are "best"
// Input and output data are passed as second argument
async function optimizationFunction({ learningRate, optimizer }, { xs, ys }) {
  // Create a simple sequential model.
  const model = tensorFlow.sequential();

  // add a dense layer to the model and compile
  model.add(tensorFlow.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: optimizers[optimizer](learningRate),
  });

  // train model using defined data
  const h = await model.fit(xs, ys, { epochs: 250 });

  //printint out each optimizer and its loss
  console.log(optimizer);
  console.log(
    'learning rate: ',
    learningRate,
    'loss: ',
    h.history.loss[h.history.loss.length - 1]
  );

  return {
    loss: h.history.loss[h.history.loss.length - 1],
    status: hpjs.STATUS_OK,
  };
}

async function hyperTensorFlowJs() {
  // Generating some data for training (y = 2x - 1) in tf tensors and defining its shape
  const xs = tensorFlow.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tensorFlow.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // defining a search space we want to optimize. Using hpjs parameters here
  const space = {
    learningRate: hpjs.uniform(0.0001, 0.1),
    optimizer: hpjs.choice(['sgd', 'adagrad', 'adam', 'adamax', 'rmsprop']),
    // epochs: hpjs.
  };

  // finding the optimal hyperparameters using hpjs.fmin. Here, 6 is the # of times
  // the optimization function will be called (this can be changed)
  const trials = await hpjs.fmin(
    optimizationFunction,
    space,
    hpjs.search.randomSearch,
    6,
    { rng: new hpjs.RandomState(654321), xs, ys }
  );

  const opt = trials.argmin;

  //printing out data
  console.log('trials', trials);
  console.log('best optimizer:', opt.optimizer);
  console.log('best learning rate:', opt.learningRate);
}

console.log('GO TENSORFLOW!');
hyperTensorFlowJs();
