package gp.math

import breeze.linalg.{DenseMatrix, DenseVector}
import org.junit.Assert._
import org.junit._

class sqDistTest {

  @Test def test_xx_1D = {

    val x = DenseVector(1.0, 2.0, 3.0).toDenseMatrix.t

    val expected = DenseMatrix((0.0, 1.0, 4.0), (1.0, 0.0, 1.0), (4.0, 1.0, 0.0))
//    println(sqDist(x, x))
    assertEquals(expected, sqDist(x, x))

  }

  @Test def test_xx_2D = {

    val x = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)).t
    val expected = DenseMatrix((0.0, 2.0, 8.0), (2.0, 0.0, 2.0), (8.0, 2.0, 0.0))
    assertEquals(expected, sqDist(x, x))

  }

  @Test def test_x1_x2_1D = {

    val x1 = DenseVector(1.0, 2.0, 3.0).toDenseMatrix
    val x2 = DenseVector(4.0, 5.0).toDenseMatrix

    val expected = DenseMatrix((9.0, 16.0), (4.0, 9.0), (1.0, 4.0))
    assertEquals(expected, sqDist(x1, x2))

  }

  @Test def test_x1_x2_2D = {

    val x1 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val x2 = DenseMatrix((7.0, 8.0), (9.0, 10.0))

    val expected = DenseMatrix((61.0, 85.0), (41.0, 61.0), (25.0, 41.0))
    assertEquals(expected, sqDist(x1, x2))

  }
}