package baidu.infra.cg

import org.apache.spark.{SparkConf, SparkContext}

object miniBatchSuite {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("cg-lr")
    val sc = new SparkContext(conf)
    TestShuffleExamples(sc)
  }

  def TestShuffleExamples(sc: SparkContext) {
    val param = new Parameter()
    param.batchSize = 10
    println("loading training dataset ................")
    val startLoad = System.currentTimeMillis()
    val examples = Utils.loadData(sc, param.train)
    val dataInfo = new DataInfo()
    dataInfo.getStats(examples.cache())
    println("Loading time: " + (System.currentTimeMillis() - startLoad) / 1000.0 + " seconds")

    val exampleArray = new ConjugateGradient(new LRCGT).shuffleExamples(examples, param, dataInfo.n)
    var i = 0
    for (mExamples <- exampleArray) {
      println("batch : " + i)
      mExamples.collect().foreach(e => e.printExample)
      i += 1
    }
  }

}

