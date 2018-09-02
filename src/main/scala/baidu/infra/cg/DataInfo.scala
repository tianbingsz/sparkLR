package baidu.infra.cg

import org.apache.spark.rdd.RDD

class DataInfo() extends Serializable {
  var d: Int = 0
  // feature dim (including bias at idx 0)
  var n: Long = 0 // num of examples

  def getStats(examples: RDD[Example]): this.type = {
    this.n = examples.count()
    this.d = examples.map(e => e.getMaxIndex()).reduce(math.max(_, _)) + 1
    this
  }
}
