const tensorFlow = require('@tensorflow/tfjs-node');
const hpjs = require('hyperparameters');

const ethPrices = require('./eth-prices-1620691200-to-1622992226.json');
const ethPriceTimestamps = Object.keys(ethPrices).map((timestampString) =>
  Number(timestampString)
);

const ewmaAlphaCoefficients = [0.1, 0.05, 0.01];

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
  { learningRate, optimizer, futureMinutes },
  { priceTimestamps }
) {
  // Create a simple sequential model.
  const model = tensorFlow.sequential();

  // add a dense layer to the model and compile
  model.add(
    tensorFlow.layers.dense({
      // Add a fully connected layer output = activation(dot(input, kernel) + bias)
      units: 1, // 1 neuron /* Output Dimensions */, 300, 500, 1000 blocks in the future
      inputShape: [4], // 3 different EWMAs, price // total of 4
      activation: 'sigmoid', // Activation function, sigmoid function: compress the output value to between 0-1
    })
  );
  model.compile({
    loss: 'meanSquaredError', // standard
    optimizer: optimizers['adam'](learningRate),
  });

  // Generating some data for training in tf tensors
  // and defining its shape
  // const ewmaPrices = [];
  const prices = [];
  const ewmaListOfList = [];

  for (let i = 0; i < ewmaAlphaCoefficients.length; i++) {
    ewmaListOfList.push([]);
  }

  const predictions = []; // 300 blocks in the future; 500, 1000

  // avoid the 1st index and the last 360 minutes (6 hours) as a buffer for lookup
  for (let index = 1; index < priceTimestamps.length - 361; index++) {
    // Calculate Inputs EWMA Price Difference
    const priceTimestamp = priceTimestamps[index];
    const price = ethPrices[priceTimestamp][priceTimestamp];
    const previousPriceTimestamp = priceTimestamps[index - 1];
    const previousPrice =
      ethPrices[previousPriceTimestamp][previousPriceTimestamp];

    for (let i = 0; i < ewmaAlphaCoefficients.length; i++) {
      const ewma =
        ewmaAlphaCoefficients[i] * price +
        (1 - ewmaAlphaCoefficients[i]) * previousPrice;

      ewmaListOfList[i].push(ewma);
    }

    prices.push(price);

    // Calculate Outputs
    const futureSeconds = futureMinutes * 60;
    const futurePriceTargetTimestamp = priceTimestamp + futureSeconds;
    const futurePriceTimestamp = binarySearchTimestamp(
      futurePriceTargetTimestamp,
      0,
      ethPriceTimestamps.length - 1
    );
    const futurePrice = ethPrices[futurePriceTimestamp][futurePriceTimestamp];
    const priceDifference = (futurePrice - price) / price;

    if (priceDifference >= 0.02) {
      predictions.push(1);
    } else {
      predictions.push(0);
    }
  }

  // const ewmaXs = tensorFlow.tensor2d(ewmaListOfList, [ewmas[0].length, 3]);
  // const ewmaXs = tensorFlow
  //
  const result = [...ewmaListOfList, prices];

  // const concat = tensorFlow.concat(result, (axis = 1)); // ([ewmas, prices], (axis = 1));
  // const xs = tensorFlow.tensor2d([stack], stack.shape);
  const xs = tensorFlow.tensor2d(result).transpose();

  // const priceDifferenceSignals = tensorFlow.tensor2d(priceChangeDifference, [
  //   priceChangeDifference.length,
  //   1,
  // ]);

  const ys = tensorFlow.tensor1d(predictions);

  // Xs / inputs
  // EWMA Prices; 0.1, 0.05, 0.01 alphas

  // Price
  // What other technical traders are looking at...

  // Ys / outputs / predictions
  // The binary 1 or 0 signal, buy or not.
  // debugger;
  // train model using defined data
  const h = await model.fit(xs, ys, {
    epochs: 3,
    batchSize: 128,
  });

  const res = await model.getWeights()[1].data();

  debugger;
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
    learningRate: 0.001,
    // learningRate: hpjs.uniform(0.0001, 0.1),
    optimizers: hpjs.choice(['adam']),
    // ewmaAlphaCoefficient: hpjs.uniform(0.01, 0.1),
    // ewmaAlphaCoefficients: [0.1, 0.05, 0.01],
    futureMinutes: hpjs.uniform(1, 360),
    // buyThreshold: hpjs.uniform(0.001, 0.1),
  };

  // finding the optimal hyperparameters using hpjs.fmin.
  // We're predicting the price X time in the future
  const trials = await hpjs.fmin(
    optimizationFunction,
    space,
    hpjs.search.randomSearch,
    50, // max_estimates
    { rng: new hpjs.RandomState(654321), priceTimestamps: ethPriceTimestamps }
  );

  const opt = trials.argmin;

  //printing out data
  console.log('trials', trials);
  console.log('best optimizer:', opt.optimizer);
  console.log('best learning rate:', opt.learningRate);
  // console.log('best buyThreshold', opt.buyThreshold);
  console.log('best ewmaAlphaCoefficient', opt.ewmaAlphaCoefficient);
  console.log('best futureMinutes', opt.futureMinutes);
}

function binarySearchTimestamp(targetTimestamp, startIndex, endIndex) {
  const midIndex = Math.floor((startIndex + endIndex) / 2);
  const previousTimestamp = ethPriceTimestamps[midIndex - 1];
  const nextTimestamp = ethPriceTimestamps[midIndex + 1];
  const midTimestamp = ethPriceTimestamps[midIndex];

  // Base Case
  if (
    previousTimestamp <= targetTimestamp &&
    targetTimestamp <= nextTimestamp
  ) {
    return midTimestamp;
  }

  // Recursive Case
  if (midTimestamp > targetTimestamp) {
    // look left
    return binarySearchTimestamp(targetTimestamp, startIndex, midIndex - 1);
  } else {
    // look right
    return binarySearchTimestamp(targetTimestamp, midIndex + 1, endIndex);
  }
}

console.log('GO TENSORFLOW!');
hyperTensorFlowJs();
