const tensorFlow = require('@tensorflow/tfjs-node');
const hpjs = require('hyperparameters');
const fs = require('fs-extra');

const EPOCH = 20;
const BATCH_SIZE = 128;

const ethPrices = require('./eth-prices-1620691200-to-1623296112.json');
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

  const futurePriceIncrease = []; // 300 blocks in the future; 500, 1000
  const priceDifferences = [];
  const timestamps = [];
  let lastPerviousTimestampBuySignal = false;

  // avoid the 1st index and the last 360 minutes (6 hours) as a buffer for lookup
  for (let index = 1; index < priceTimestamps.length - 361; index++) {
    // Calculate Inputs EWMA Price Difference
    const priceTimestamp = priceTimestamps[index];
    timestamps.push(priceTimestamp);
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
    const futureMinutesTemp = 60;
    const futureSeconds = futureMinutesTemp * 60;
    const futurePriceTargetTimestamp = priceTimestamp + futureSeconds;
    const futurePriceTimestamp = binarySearchTimestamp(
      futurePriceTargetTimestamp,
      0,
      ethPriceTimestamps.length - 1
    );
    const futurePrice = ethPrices[futurePriceTimestamp][futurePriceTimestamp];
    const priceDifference = (futurePrice - price) / price;

    priceDifferences.push(priceDifference);

    if (priceDifference >= 0.02) {
      futurePriceIncrease.push(1);
      // lastPerviousTimestampBuySignal = true;
    } else {
      // lastPerviousTimestampBuySignal = false;
      futurePriceIncrease.push(0);
    }
  }

  // const myTable = [
  //   ...ewmaListOfList,
  //   prices,
  //   futurePriceIncrease,
  //   priceDifferences,
  //   timestamps,
  // ];
  // debugger;
  // const outputFilename = 'my-table.csv';
  // try {
  //   fs.unlinkSync(outputFilename);
  // } catch (error) {}
  // fs.ensureFileSync(outputFilename);
  // fs.appendFileSync(
  //   outputFilename,
  //   `ewma_01,ewma_005,ewma_001,price,buy_signal,price_diff,timestamp\n`
  // );

  // for (let i = 0; i < prices.length; i++) {
  //   fs.appendFileSync(
  //     outputFilename,
  //     `${myTable[0][i]},${myTable[1][i]},${myTable[2][i]},${myTable[3][i]},${myTable[4][i]},${myTable[5][i]},${myTable[6][i]}\n`
  //   );
  // }

  // return;

  // const ewmaXs = tensorFlow.tensor2d(ewmaListOfList, [ewmas[0].length, 3]);
  // const ewmaXs = tensorFlow
  //
  // const result = [...ewmaListOfList, prices];
  const xsDataSet = [...ewmaListOfList, prices];
  const xsDateSetTraining = [
    xsDataSet[0].slice(0, Math.round(xsDataSet[0].length * 0.8)),
    xsDataSet[1].slice(0, Math.round(xsDataSet[1].length * 0.8)),
    xsDataSet[2].slice(0, Math.round(xsDataSet[2].length * 0.8)),
    xsDataSet[3].slice(0, Math.round(xsDataSet[3].length * 0.8)),
  ];
  const xsDateSetValidation = [
    xsDataSet[0].slice(Math.round(xsDataSet[0].length * 0.8) + 1),
    xsDataSet[1].slice(Math.round(xsDataSet[1].length * 0.8) + 1),
    xsDataSet[2].slice(Math.round(xsDataSet[2].length * 0.8) + 1),
    xsDataSet[3].slice(Math.round(xsDataSet[3].length * 0.8) + 1),
  ];

  // const concat = tensorFlow.concat(result, (axis = 1)); // ([ewmas, prices], (axis = 1));
  // const xs = tensorFlow.tensor2d([stack], stack.shape);
  const xs = tensorFlow.tensor2d(xsDateSetTraining).transpose();

  const ysDataSet = futurePriceIncrease;
  const ysDataSetTraining = ysDataSet.slice(
    0,
    Math.round(ysDataSet.length * 0.8)
  );
  const ysDataSetValidation = ysDataSet.slice(
    Math.round(ysDataSet.length * 0.8) + 1
  );

  const ys = tensorFlow.tensor1d(ysDataSetTraining);

  // Xs / inputs
  // EWMA Prices; 0.1, 0.05, 0.01 alphas

  // Price
  // What other technical traders are looking at...

  // Ys / outputs / futurePriceIncrease
  // The binary 1 or 0 signal, buy or not.
  // debugger;
  // train model using defined data
  // TODO: randomize our training.
  const h = await model.fit(xs, ys, {
    epochs: EPOCH,
    batchSize: BATCH_SIZE,
  });

  const weightsObject = model.getWeights();
  const res1 = await weightsObject[0].data(); // kernel
  const res2 = await weightsObject[1].data(); // bias
  debugger;
  console.log(res1);

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
    1, // max_estimates
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
