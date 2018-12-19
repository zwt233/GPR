package gp.model

import breeze.linalg.{DenseMatrix, DenseVector, cholesky}
import gp.cov.CovFunc
import gp.math.invchol

import scala.math._

//@TODO remove x and y from the model, keep covFunc and covFuncParams only
case class GprModel(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseStdDev: Double, meanFunc: (DenseMatrix[Double]) => DenseVector[Double]) {

  def calcKXX(): DenseMatrix[Double] = {
    val kXX = covFunc.cov(x, x, covFuncParams) + pow(noiseStdDev,2) * DenseMatrix.eye[Double](x.rows) + DenseMatrix.eye[Double](x.rows) * 1e-7
    kXX
  }

  def calcKXXInv(kXX: DenseMatrix[Double]): DenseMatrix[Double] = {
//    println(kXX)
//    println(invchol(cholesky(kXX).t))
    val kXXInv = invchol(cholesky(kXX).t)
//    println(kXXInv)
    kXXInv
  }
}

object GprModel {

  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseStdDev: Double, mean: Double = 0d): GprModel = {

    val meanFunc = (x: DenseMatrix[Double]) => DenseVector.zeros[Double](x.rows) + mean
    new GprModel(x, y, covFunc, covFuncParams, noiseStdDev, meanFunc)
  }

}