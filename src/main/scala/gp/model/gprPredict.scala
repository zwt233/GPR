package gp.model

import breeze.linalg.{DenseMatrix, DenseVector, diag,inv}


import scala.math._

object gprPredict {

  def apply(z: DenseMatrix[Double], model: GprModel): DenseMatrix[Double] = {

    val meanX = model.meanFunc(model.x)

    val kXX = model.calcKXX()
//    println(kXX)
    val kXXInv = model.calcKXXInv(kXX)
//    val kXXInv = inv(kXX)
//    println(kXX)
//
//    println(kXXInv*kXX)

    val kXZ = model.covFunc.cov(model.x, z, model.covFuncParams)


    val kZZ = model.covFunc.cov(z, z, model.covFuncParams)


    val meanZ = model.meanFunc(z)

    val predMean = meanZ + kXZ.t * (kXXInv * (model.y - meanX))
    val predVar = kZZ - kXZ.t * kXXInv * kXZ

    DenseVector.horzcat(predMean, diag(predVar))

  }

}