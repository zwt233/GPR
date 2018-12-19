package gp.cov

import breeze.linalg.{DenseMatrix, DenseVector}
import gp.math.sqDist
import breeze.numerics._

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

case class Cov_Matern() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    require(covFuncParams.size == 2, "Wrong number of hyper parameters. Is=%d, expected=1".format(covFuncParams.size))

    val l = covFuncParams(0)
    val theta = covFuncParams(1)

    val sqDistMatrix = sqDist(x1, x2)

    val cov_left = (sqrt(5) * sqrt(sqDistMatrix)) / l +  (sqDistMatrix / pow(l,2) *5.0/3.0) + 1.0
    val cov_right = exp(-sqrt(5) * sqrt(sqDistMatrix) / l)
    val covMatrix = pow(theta,2)*cov_left *:* cov_right
    //    println(covMatrix)
    covMatrix
  }


  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val l = covFuncParams(0)
    val theta = covFuncParams(1)

    val sqDistMatrix = sqDist(x1, x2)
    val cov_left = (sqrt(5) * sqrt(sqDistMatrix)) / l +  5.0/3.0 * sqDistMatrix / pow(l,2) + 1.0
    val cov_right = exp(-sqrt(5) * sqrt(sqDistMatrix) / l)

    val cov_left_grad = -(sqrt(5) * sqrt(sqDistMatrix) / pow(l,2) + 10.0*sqDistMatrix / (3.0 * pow(l,3))) *:* cov_right*pow(theta,2)
    val cov_right_grad = cov_left*:*cov_right*:*(sqrt(5) * sqrt(sqDistMatrix) / pow(l,2))*pow(theta,2)

    val cov_l_grad = cov_left_grad + cov_right_grad
    val cov_theta_grad = cov_left*:*cov_right*2.0*theta
    //    println(cov_l_grad)
    Array(cov_l_grad,cov_theta_grad)

  }
}