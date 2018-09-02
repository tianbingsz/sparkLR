package baidu.infra.cg

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector

class Model(val param: Parameter) extends Serializable {
  var w: Vector = null
  // <idx, weigth> from w
  var wMap = scala.collection.mutable.Map[Long, Double]()

  def testAccuracy(examples: RDD[Example]): Double = {
    val predLabels = examples.map { example =>
      val pred = predict(example)
      (example.y, pred)
    }
    val accuracy = predLabels.filter(e => e._1 == e._2).count.toDouble / examples.count
    accuracy
  }

  /**
   * Predict a label given a Example.
   */
  def predict(example: Example): Double = {
    var label = 1
    if (innerProduct(example.x) < 0) label = -1
    label
  }

  // <pred weight, x> for given x (array[feature])
  private def innerProduct(features: Array[Int]): Double = {
    var innerProd = 0.0 // <w,x>
    val lastIndex = w.length
    for (fIdx <- features) {
      if (fIdx <= lastIndex) {
        innerProd += w(fIdx)
      }
    }
    innerProd
  }

  def AUCROC(examples: RDD[Example]): Double = {
    val scoreLabels: RDD[(Double, Double)] = examples.map {
      example =>
        val score = predictProbability(example)
        (score, (1 + example.y) / 2) // {-1, +1} => {0,1}
    }
    val metrics = new BinaryClassificationMetrics(scoreLabels)
    val auc = metrics.areaUnderROC()
    auc
  }

  /**
   * Predict probabilities given a Example
   * P(y | x) = 1 / (1 + exp(- w'x y)), y in {-1, +1}
   */
  def predictProbability(example: Example): Double = {
    // p(y = 1 | x) = 1/(1 + exp(-w'x))
    val predProb = Utils.sigmoid(innerProduct(example.x))
    predProb
  }

  /**
   * save w into disk
   */
  def saveModel(name: String) {
    println("save model to " + name)
    val file = new File(name)
    val bw = new BufferedWriter(new FileWriter(file))
    // e.g. weight(0.3)
    bw.write(w.elements.mkString("\n"))
    bw.close()
  }

  /**
   * save w into disk, with reverse featureMap
   */
  def saveModel(name: String, rfMap: scala.collection.mutable.Map[Int, Long]) {
    println("save model map to " + name)
    val file = new File(name)
    val bw = new BufferedWriter(new FileWriter(file))
    genWMap(rfMap) // map model with fmap
    bw.write(wMap.map{pair => pair._1 + " " + pair._2}.mkString("\n"))
    bw.close()
  }

  // <idx, weigth> from w
  private def genWMap(rfMap: scala.collection.mutable.Map[Int, Long]) {
    wMap += 0.toLong -> w(0) // bias
    var idx = 1
    while (idx < w.length) {
      if (rfMap contains idx) {
        wMap += rfMap(idx) -> w(idx)
      }
      idx += 1
    }
  }

}

