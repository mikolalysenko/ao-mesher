"use strict"

var ndarray = require("ndarray")
var compileStencil = require("ndarray-stencil")
var compileMesher = require("greedy-mesher")
var pool = require("typedarray-pool")

var OPAQUE_BIT      = (1<<15)
var VOXEL_MASK      = (1<<16)-1
var AO_SHIFT        = 16
var AO_BITS         = 2
var AO_MASK         = (1<<AO_BITS)-1
var FLIP_BIT        = (1<<(AO_SHIFT+4*AO_BITS))
var TEXTURE_SHIFT   = 4
var TEXTURE_MASK    = (1<<TEXTURE_SHIFT)-1
var VERTEX_SIZE     = 8

//
// Vertex format:
//
//  x, y, z, ambient occlusion, nx, ny, nz, tex_id
//
//
// Voxel format:
//
//  * Max 16 bits per voxel
//  * Bit 15 is opacity flag  (set to 1 for voxel to be solid, otherwise rendererd transparent)
//  * Texture index is calculated by masking out lower order bits
//
//
// This stuff can be changed over time.  -Mik
//

//Retrieves the texture for a voxel
function voxelTexture(voxel, side) {
  return voxel&0xff
}

//Calculates ambient occlusion level for a vertex
function vertexAO(s1, s2, c) {
  if(s1 && s2) {
    return 1
  }
  return 3 - (s1 + s2 + c)
}

//Calculates the ambient occlusion bit mask for a facet
function facetAO(a00, a01, a02,
                 a10,      a12,
                 a20, a21, a22) {
  var s00 = (a00&OPAQUE_BIT) ? 1 : 0
    , s01 = (a01&OPAQUE_BIT) ? 1 : 0
    , s02 = (a02&OPAQUE_BIT) ? 1 : 0
    , s10 = (a10&OPAQUE_BIT) ? 1 : 0
    , s12 = (a12&OPAQUE_BIT) ? 1 : 0
    , s20 = (a20&OPAQUE_BIT) ? 1 : 0
    , s21 = (a21&OPAQUE_BIT) ? 1 : 0
    , s22 = (a22&OPAQUE_BIT) ? 1 : 0
  return (vertexAO(s10, s01, s00)<< AO_SHIFT) +
         (vertexAO(s01, s12, s02)<<(AO_SHIFT+AO_BITS)) +
         (vertexAO(s12, s21, s22)<<(AO_SHIFT+2*AO_BITS)) +
         (vertexAO(s21, s10, s20)<<(AO_SHIFT+3*AO_BITS))
}

//Generates a surface voxel, complete with ambient occlusion type
function generateSurfaceVoxel(
  v000, v001, v002,
  v010, v011, v012,
  v020, v021, v022,
  v100, v101, v102,
  v110, v111, v112,
  v120, v121, v122) {
  if(v111 && !v011) {
    return v011 | FLIP_BIT | facetAO(v000, v001, v002,
                                     v010,       v012,
                                     v020, v021, v022)
  } else if(v011 && !v111) {
    return v111 | facetAO(v100, v101, v102,
                          v110,       v112,
                          v120, v121, v122)
  }
}



//Surface stencil operation
var surfaceStencil = compileStencil([
  [ 0,-1,-1], [ 0,-1, 0], [ 0,-1, 1],
  [ 0, 0,-1], [ 0, 0, 0], [ 0, 0, 1],
  [ 0, 1,-1], [ 0, 1, 0], [ 0, 1, 1],
  [ 1,-1,-1], [ 1,-1, 0], [ 1,-1, 1],
  [ 1, 0,-1], [ 1, 0, 0], [ 1, 0, 1],
  [ 1, 1,-1], [ 1, 1, 0], [ 1, 1, 1]], generateSurfaceVoxel)


function MeshBuilder() {
  this.buffer = pool.mallocUint8(1024)
  this.ptr = 0
  this.z = 0
  this.u = 0
  this.v = 0
  this.d = 0
}

MeshBuilder.prototype.append = function(lo_x, lo_y, hi_x, hi_y, val) {
  var buffer = this.buffer
  var ptr = this.ptr>>>0
  var z = this.z|0
  var u = this.u|0
  var v = this.v|0
  var d = this.d|0

  //Grow buffer if we exceed capacity
  if(ptr + 6*VERTEX_SIZE > buffer.length) {
    var tmp = pool.mallocUint8(2*buffer.length);
    tmp.set(buffer)
    pool.freeUint8(buffer)
    buffer = tmp
    this.buffer = buffer
  }

  var flip = !!(val & FLIP_BIT)
  var nv = flip ? 127 : 129
  var nx = 0, ny = 0, nz = 0
  if(d === 0) {
    nx = nv
    ny = nz = 128
  } else if(d === 1) {
    ny = nv
    nx = nz = 128
  } else {
    nz = nv
    nx = ny = 128
  }
  
  var a00 = 1+((val>>>AO_SHIFT)&AO_MASK)
  var a10 = 1+((val>>>(AO_SHIFT+AO_BITS))&AO_MASK)
  var a11 = 1+((val>>>(AO_SHIFT+2*AO_BITS))&AO_MASK)
  var a01 = 1+((val>>>(AO_SHIFT+3*AO_BITS))&AO_MASK)
  
  var tex_id = voxelTexture(val&VOXEL_MASK, d + flip ? 3 : 0)
  
  var flip_diag = (a00 + a11 < a10 + a01)

  if(flip_diag) {
    if(flip) {
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8

      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8

      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
    } else {
    
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8

      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
    }
  } else {
    //Check if flipped
    if(!flip) {
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8

      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
    } else {
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a01
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
      
      buffer[ptr+u] = hi_x
      buffer[ptr+v] = hi_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a11
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8

      buffer[ptr+u] = hi_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a10
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id

      ptr += 8
      
      buffer[ptr+u] = lo_x
      buffer[ptr+v] = lo_y
      buffer[ptr+d] = z
      buffer[ptr+3] = a00
      buffer[ptr+4] = nx
      buffer[ptr+5] = ny
      buffer[ptr+6] = nz
      buffer[ptr+7] = tex_id
      
      ptr += 8
    }
  }
  
  this.ptr = ptr
}

var meshBuilder = new MeshBuilder()

//Compile mesher
var meshSlice = compileMesher({
  order: [1, 0],
  append: MeshBuilder.prototype.append.bind(meshBuilder)
})

//Compute a mesh
function computeMesh(array) {

  var scratch = pool.mallocInt32(array.size)
  var st0     = ndarray(scratch,array.shape.slice(0))
  
  meshBuilder.ptr = 0
  for(var d=0; d<3; ++d) {
    var u = (d+1)%3
    var v = (d+2)%3
    
    //Create slice
    var st = st0.transpose(d, u, v)
    var slice = st.pick(0)

    meshBuilder.d = d
    meshBuilder.u = u
    meshBuilder.v = v
    
    //Compute surface stencil for this side
    surfaceStencil(st, array.transpose(d, u, v))
    
    //Generate slices
    var nx = st.shape[0]|0
    for(var i=0; i<nx-1; ++i) {
      meshBuilder.z = i+1
      meshSlice(slice)
      slice.offset += st.stride[0]
    }
  }
  pool.freeInt32(scratch)
  
  //Release uint8 array if no vertices were allocated
  if(meshBuilder.ptr === 0) {
    return null
  }
  
  //Slice out buffer
  var rbuffer = meshBuilder.buffer
  var rptr = meshBuilder.ptr
  meshBuilder.buffer = pool.mallocUint8(1024)
  meshBuilder.ptr = 0
  return { buffer: rbuffer, length: rptr }
}

module.exports = computeMesh