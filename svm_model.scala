import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.classification.{ SVMModel, SVMWithSGD } // svm类
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics // 二分类度量
import org.apache.spark.mllib.util.MLUtils

object svm{

    def main(args: Array[String]){
        val conf = new SparkConf().setAppName("svm")
        val sc = new SparkContext(conf)
        // 加载数据
        val data = MLUtils.loadLibSVMFile(sc, "./data/sample_libsvm_data.txt")
        // 数据划分
        val splits = data.randomSplit(Array(0.6,0.4))
        val (train, test) = (splits(0).cache(),splits(1))
        // 模型训练，设置100次迭代
        val model = SVMWithSGD.train(input = train,numIterations = 100)
        // 测试集预测
        val label_pred = test.map{
                            point=>
                            val pred = model.predict(point.features)
                            (point.label, pred)
                        }

        // 度量
        val metric = new BinaryClassificationMetrics(label_pred)
        val auc = metric.areaUnderROC
        println("The auc score on test is "+auc)
    }

}


