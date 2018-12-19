package gp.model

import java.io.File

import breeze.linalg._
import breeze.numerics._
import org.junit.Assert._
import org.junit._
import gp.cov.Cov_Matern
import gp.cov.Cov_SEiso

class gprTest {

  //[x,y]
//  val data = csvread(new File("src/test/resources/gp/gpml_regression_data.csv"), skipLines = 1)

//  val x = data(::, 0 to 0)
//  val y = data(::, 1)
  val x = DenseMatrix((1.0, 2.0, 3.0,4.0,5.0)).t
  val y = DenseVector(2.0,4.0,6.0,8.0,10.0)
//  val y = sin(DenseVector(1.0, 2.0, 3.0,4.0,5.0))

  val covFunc = Cov_Matern()
  val covFuncParams = DenseVector(1.0,1.0)
  val noiseStdDev = 0.1

  @Test def test = {

    val gpModel = gpr(x, y, covFunc, covFuncParams, noiseStdDev)
    println(gpModel.covFuncParams)
    println(gpModel.noiseStdDev)

//    assertEquals(0.68594, gpModel.covFuncParams(0), 0.0001)
//    assertEquals(-0.99340, gpModel.covFuncParams(1), 0.0001)
//    assertEquals(-1.9025, gpModel.noiseStdDev, 0.0001)

  }
}