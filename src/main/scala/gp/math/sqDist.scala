package gp.math

import breeze.generic.UFunc
import breeze.linalg.{DenseMatrix, _}

/**
 * Computes matrix of pair-wise square distances between matrices x1 and x2.
 * Implementation based on sq_dist function from gpml library for Gaussian processes
 *
 * @param x1 [D x N]
 * @param x1 [D x M]
 * @return matrix of square distances [N x M]
 */
object sqDist extends UFunc {

  implicit object implDMDM extends Impl2[DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]] {
    def apply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double] = {
      
      val t1 = -2.0 * (x1 * x2.t)

      val t2 = t1(*, ::) + sum(x2.t :* x2.t, Axis._0).t

      t2(::, *) + sum(x1.t :* x1.t, Axis._0).t
      
 
    }
  }
}
  
  