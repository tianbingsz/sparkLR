package baidu.infra.cg

import org.apache.spark.{SparkConf, SparkContext}

object featureMappingSuite {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("cg-lr")
    val sc = new SparkContext(conf)
    TestFeatureMapping()
  }

  def TestFeatureMapping() {
    val param = new Parameter()
    val fMap = new FeatureMapping()
    fMap.loadFeatureMap(param.dict)
    fMap.printFeatureMap()
  }

}

