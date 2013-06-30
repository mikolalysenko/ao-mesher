var computeMesh = require("../mesh.js")
var ndarray = require("ndarray")
var ops = require("ndarray-ops")

var voxel = require("voxel")

var x = ndarray(new Int32Array(33*33*33), [33,33,33])

ops.assigns(x.hi(8,8,8).lo(4,4,4), 1)

var r = computeMesh(x)

var start = Date.now()
for(var i=0; i<1000; ++i) {
  computeMesh(x)
}

console.log(Date.now() - start)