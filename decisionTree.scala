import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.DecisionTree //树模型
import org.apache.spark.mllib.tree.model.DecisionTreeModel

object decision_tree{
    def main(args: Array[String]){
        val conf = new SparkConf().setAppName("decisonTree")
        val sc = new SparkContext(conf)
        // 加载数据
        val data = MLUtils.loadLibSVMFile(sc,"./data/sample_libsvm_data.txt")
        // 数据划分
        val splits = data.randomSplit(Array(0.7,0.3))
        val (train, test) = (splits(0).cache(),splits(1))
        // 模型训练
        val numClasses = 2 // 类别数
        val categoricalFeaturesInfo = Map[Int, Int]()
        val impurity = "gini"
        val maxDepth = 5 // 最大深度
        val maxBins = 32  // 叶子最大样本数
        val model = DecisionTree.trainClassifier(train,numClasses,categoricalFeaturesInfo,impurity,maxDepth,maxBins)
        // 预测
        val label_pred = test.map{
                            piont =>
                            val pred = model.predict(piont.features)
                            (piont.label, pred)
                        }
        // 测试集误差率
        val pred_error = label_pred.filter(t => t._1 != t._2).count.toDouble / test.count.toDouble
        println("The error on test is "+ pred_error)
    }

}