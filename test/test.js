var computeMesh = require("../mesh.js")
var ndarray = require("ndarray")
var fill = require("ndarray-fill")
var voxel = require("voxel")

var ndseg = require("ndarray-segment")

var x = ndarray(new Int32Array(33*33*33), [33,33,33])

fill(x, function(i,j,k) {
  var a = i-16
  var b = i-16
  var c = i-16
  return (a*a + b*b + c*c) ? 1<<15 : 0
})

for(var j=0; j<10; ++j) {
  computeMesh(x)
}

setTimeout(function(){
  var start = Date.now()
  for(var i=0; i<1000; ++i) {
    computeMesh(x)
    //voxel.meshers.greedy(x.data, x.shape)
  }
  console.log(Date.now() - start)
}, 10)