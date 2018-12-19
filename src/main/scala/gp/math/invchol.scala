package gp.math

import breeze.linalg.{DenseMatrix, inv}

object invchol {

  /**
    * @param R, where R'*R= A, cholesky decomposition
    */
  def apply(R:DenseMatrix[Double]):DenseMatrix[Double] = {
    val Rinv = inv(R)
    Rinv*Rinv.t

  }
}