package gp.cov

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import gp.math.sqDist

/**
 * Implementation based 'http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html'
 *
 *  Squared Exponential covariance function with isotropic distance measure. The
 * covariance function is parameterized as:
 *
 * k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
 *
 * where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
 * variance.
 *
 * Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
 *
 * @param sf - log of signal standard deviation
 * @param ell - log of length scale standard deviation
 */

case class Cov_SEiso() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    require(covFuncParams.size == 2, "Wrong number of hyper parameters. Is=%d, expected=2".format(covFuncParams.size))

    val Sf = covFuncParams(0)
    val Ell = covFuncParams(1)

    val sqDistMatrix = sqDist(x1, x2)
    val covMatrix = pow(Sf,2) * exp(-0.5 * sqDistMatrix/pow(Ell,2))
    covMatrix

  }



  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val Sf = covFuncParams(0)
    val Ell = covFuncParams(1)

    val sqDistMatrix = sqDist(x1, x2)

    val expSqDistMatrix = exp(-0.5 * sqDistMatrix/pow(Ell,2))

    val covMatrixDSf = 2 * Sf * expSqDistMatrix

    val covMatrixDEll = pow(Sf,2)* expSqDistMatrix :* sqDistMatrix/pow(Ell,3)

    Array(covMatrixDSf, covMatrixDEll)

  }

}