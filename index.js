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
async function optimizationFunction(
  { learningRate, optimizer, ewmaAlphaCoefficient, futureMinutes },
  { prices }
) {
  // Create a simple sequential model.
  const model = tensorFlow.sequential();

  // add a dense layer to the model and compile
  model.add(
    tensorFlow.layers.dense({
      units: 1 /* Output Dimensions */,
      inputShape: [1] /* EWMA - Price = Diff */,
    })
  );
  model.compile({
    loss: 'meanSquaredError', // standard
    optimizer: optimizers[optimizer](learningRate),
  });

  // Generating some data for training (y = 2x - 1) in tf tensors
  // and defining its shape
  // TODO: import prices, calculate EWMA, and calculate the difference.
  // const ewmaPriceVsPriceDifferenceSignal = tensorFlow.tensor2d([...(EWMA - price) / price], [prices.length, 1]);
  // const priceChange = tensorFlow.tensor2d([...(future price - now price) / (now price)], [prices.length, 1])

  // train model using defined data
  const h = await model.fit(ewmaPriceVsPriceDifferenceSignal, priceChange, {
    epochs: 5,
  });

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
  // defining a search space we want to optimize. Using hpjs parameters here
  const space = {
    learningRate: hpjs.uniform(0.0001, 0.1),
    optimizers: hpjs.choice(['adam']),
    // buyThreshold: hpjs.uniform(0.001, 0.1),
    ewmaAlphaCoefficient: hpjs.uniform(0.01, 0.1),
    futureMinutes: hpjs.uniform(1, 360),
    // optimizer: hpjs.choice(['sgd', 'adagrad', 'adam', 'adamax', 'rmsprop']),
  };

  // finding the optimal hyperparameters using hpjs.fmin.
  // We're predicting the price X time in the future
  const trials = await hpjs.fmin(
    optimizationFunction,
    space,
    hpjs.search.randomSearch,
    50,
    { rng: new hpjs.RandomState(654321), prices: [] }
  );

  const opt = trials.argmin;

  //printing out data
  console.log('trials', trials);
  console.log('best optimizer:', opt.optimizer);
  console.log('best learning rate:', opt.learningRate);
  console.log('best buyThreshold', opt.buyThreshold);
  console.log('best ewmaAlphaCoefficient', opt.ewmaAlphaCoefficient);
  console.log('best futureMinutes', opt.futureMinutes);
}

console.log('GO TENSORFLOW!');

hyperTensorFlowJs();
