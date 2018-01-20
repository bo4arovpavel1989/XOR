var activation_sigmoid = x => 1 / (1 + Math.exp(-x));
var derivative_sigmoid = x => {
    const fx = activation_sigmoid(x);
    return fx * (1 - fx);
};

module.exports.activation_sigmoid = activation_sigmoid;
module.exports.derivative_sigmoid = derivative_sigmoid;