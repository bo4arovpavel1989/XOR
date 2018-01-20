
var activation_sigmoid = require('./sigmoid.js').activation_sigmoid;
var derivative_sigmoid = require('./sigmoid.js').derivative_sigmoid;
var weights = require('./weights.js');

function nn(i1, i2) {
    var h1_input =
        weights.i1_h1 * i1 +
        weights.i2_h1 * i2 +
        weights.bias_h1;
    var h1 = activation_sigmoid(h1_input);

    var h2_input =
        weights.i1_h2 * i1 +
        weights.i2_h2 * i2 +
        weights.bias_h2;
    var h2 = activation_sigmoid(h2_input);


    var o1_input =
        weights.h1_o1 * h1 +
        weights.h2_o1 * h2 +
        weights.bias_o1;

    var o1 = activation_sigmoid(o1_input);
    
    return o1;
}

module.exports = nn;