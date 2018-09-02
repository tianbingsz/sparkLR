package baidu.infra.cg

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector

object Utils {

  /**
   * Load data in the {+1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023}
   * format into an RDD[Example]. y in {-1, +1} , x Seq{idx : val}
   */
  def loadData(sc: SparkContext, path: String): RDD[Example] = {
    loadData(sc, path, sc.defaultMinSplits)
  }

  /**
   * Load data into an RDD[Example].
   */
  def loadData(sc: SparkContext, path: String, numPartitions: Int): RDD[Example] = {
    val parsed = sc.textFile(path, numPartitions).map(_.trim).filter(!_.isEmpty)
    parsed.map(line => {
      val tokens = line.split(" |\t|\n")
      val d = tokens.size - 1
      val y = tokens.head.toDouble
      val x = new Array[Int](d + 1)
      x(0) = 0 // idx:bias (0,1)
      for (i <- 1 to d) {
        // idx:val, idx in [0...], val = 1.0 by default
        val pair = tokens(i).split(":")
        x(i) = pair(0).toInt
      }
      new Example(x, y)
    })
  }

  /**
   * write a vector into file
   */
  def saveVec2File(v: Array[Double], name: String) = {
    val file = new File(name)
    val bw = new BufferedWriter(new FileWriter(file))
    // e.g. loglik : 0.5, 0.4, 0.3
    bw.write(v.mkString(", "))
    bw.close()
  }

  /**
   * sigmoid(z) = 1 / (1 + exp(-z))
   */
  def sigmoid(z: Double): Double = {
    var res = 0.5
    if (Math.abs(z) == 0.0) return res
    if (z > 0) {
      res = 1.0 / (1.0 + Math.exp(-z))
    } else {
      res = 1.0 - 1.0 / (1.0 + Math.exp(z))
    }
    res
  }

  def norm2(v: Vector): Double = {
    return math.sqrt(v dot v)
  }

  /**
   * Load input sparse features in adfea format
   * Load data into an RDD[(Array[Int], Double)].
   * RDD[Example(feature, label)]
   */
  def LoadAdfeaFeatures(sc: SparkContext, path: String, dictPath: String): RDD[Example] = {
    // Load and broadcast the global dimensionality reduction dictionary
    println(s"Loading DR dictionary...")
    val fMap = new FeatureMapping()
    fMap.loadFeatureMap(dictPath)
    val DRDict = fMap.featureMap
    println(s"Loaded DR dictionary of size: ${DRDict.size}")
    val gDict = sc.broadcast(DRDict.toMap)

    val origInput = sc.textFile(path)
    val sparseRDD = origInput.flatMap(line => {
      // Segments of each record are separated by tabs
      val segs = line.split("\t")

      // Extract the show, click and feature list for the first segment
      val showClickFea = segs(0).split(" ")
      //val show1 = java.lang.Integer.parseInt(showClickFea(0))
      //val click1 = java.lang.Integer.parseInt(showClickFea(1))
      val show1 = showClickFea(0).toInt
      val click1 = showClickFea(1).toInt
      val fea1 = showClickFea.drop(2)

      // If there is only one segment, then itself will generate multiple samples, otherwise,
      // each of the remaining segments will combine with the first segment to produce multiple samples, while
      // the first segment itself won't be processed.
      if (segs.size == 1) {
        genSamples(show1, click1, fea1, gDict.value)
      } else {
        (1 until segs.size).flatMap(segIndex => {
          val curShowClickFea = segs(segIndex).split(" ")
          //val curShow = java.lang.Integer.parseInt(curShowClickFea(0))
          //val curClick = java.lang.Integer.parseInt(curShowClickFea(1))
          val curShow = curShowClickFea(0).toInt
          val curClick = curShowClickFea(1).toInt
          val curFea = curShowClickFea.drop(2)
          val combinedFea = curFea ++ fea1
          genSamples(curShow, curClick, combinedFea, gDict.value)
        })
      }
    })

    // The dimension of feature space should be the size of dict, no need to compute through other ways
    val dim = DRDict.size + 1
    println("largest possible dim : " + dim)

    sparseRDD.map {
      case (indices, y) => new Example(indices, y)
    }
  }

  /**
   * Each original adfea features are transformed into #show samples among which #click of them are positive samples.
   * The dimension of original adfea feasign is also reduced based on a dimension reduction dictionary
   *
   * @param features in adfea feature format,which feasign:slot
   * @param dict for dimensionality reduction
   * @return Array of feature samples in the forward of (x_indices[], x_values[], y)
   */
  def genSamples(show: Int, click: Int, features: Array[String], dict: Map[Long, Int]): Array[(Array[Int], Double)] = {
    //    val featureDim = features.size
    val result = new Array[(Array[Int], Double)](show)
    //    val indices = new Array[Int](featureDim)
    val origIndices = features.map(f => {
      val pair = f.split(":")
      // Note some low-frequency features might not exist inside dict
      if (dict contains java.lang.Long.parseUnsignedLong(pair(0))) {
        dict(java.lang.Long.parseUnsignedLong(pair(0)))
      } else {
        -1
      }
    }).filter(_ != -1).sorted

    // dimension 0 with value 1 is bias adjustment item
    val indices = 0 +: origIndices

    // value is 1.0 by default
    //val values = new Array[Double](indices.size).map(_ => 1.0)
    var posCount = click
    for (i <- 0 until show) {
      var yValue: Double = -1
      if (posCount > 0) {
        yValue = 1
      }
      posCount -= 1
      //result(i) = (indices, values, yValue)
      result(i) = (indices, yValue)
    }
    result
  }

}
