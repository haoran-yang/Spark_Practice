import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS} // 逻辑回归类
import org.apache.spark.mllib.util.MLUtils 
import org.apache.spark.mllib.evaluation.MulticlassMetrics // 多分类度量

object LogisticRegressionModel{

    def main(args: Array[String]){

        val conf = new SparkConf().setAppName("LogisticRegression")
        val sc = new SparkContext(conf)
        // 读取数据
        val data = MLUtils.loadLibSVMFile(sc, "./data/sample_libsvm_data.txt")
        // 训练集和测试集划分
        val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
        val train = splits(0).cache()
        val test = splits(1)
        // 模型训练，设置类别为2
        val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(train)

        // 测试集预测
        val label_pred = test.map{
            case LabeledPoint(label,features) =>
            val pred = model.predict(features)
            (label, pred)
        }
        // 打印前10个
        val label_pred10 = label_pred.take(10)
        println("true_label"+"\t"+"pred_label")
        for(i <- 0 to 9){
            println(label_pred10(i)._1+"\t"+label_pred10(i)._2)
        }
        // 准确率
        val metric = new MulticlassMetrics(label_pred)
        println("The accuracy on test is "+metric.accuracy)
        //保存和加载
        model.save(sc,"./LogisticRegression")
        val model2 = LogisticRegressionModel.load(sc, "./LogisticRegression")
    }

}