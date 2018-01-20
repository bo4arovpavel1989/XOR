var random = require('seed-random')(1337);
var weights = {
    i1_h1: random(),
    i2_h1: random(),
    bias_h1: random(),
    i1_h2: random(),
    i2_h2: random(),
    bias_h2: random(),
    h1_o1: random(),
    h2_o1: random(),
    bias_o1: random(),
};

module.exports=weights;