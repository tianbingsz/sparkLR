package baidu.infra.cg

class Parameter() extends Serializable {
  var eps: Double = 1e-6
  var lambda: Double = 0.1
  var nIter: Int = 100
  var batchSize: Int = 1000
  // how many epoches for each minibatch
  var miter: Int = 10
  //  usually take n (num of examples)
  var nScaling = 1 
  var dir: String = "/tmp/t"
  var train: String = "/Users/tianbingxu/work/cg_dense_1/data/test/test"

  var test: String = "/Users/tianbingxu/work/cg/data/iris/iris"
  var dict: String = "/Users/tianbingxu/work/cg_dense_1/data/dict/DRdict"
}
