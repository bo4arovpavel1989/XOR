var R = require('ramda');
void 0; 
var data=require('./data.js');
var nn = require('./nn.js');
var applyTrainUpdate=require('./train.js');


var calculateResults = () =>
    R.mean(data.map(({input: [i1, i2], output: y}) => Math.pow(y - nn(i1, i2), 2)));

var outputResults = () => 
    data.forEach(({input: [i1, i2], output: y}) => 
                 console.log(`${i1} XOR ${i2} => ${nn(i1, i2)} (expected ${y})`));


R.times(() => applyTrainUpdate(), 100000)
				 
calculateResults();

outputResults();				 