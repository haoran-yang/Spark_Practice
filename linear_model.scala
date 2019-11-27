import org.apache.spark.mllib.regression.{LinearRegressionModel,LinearRegressionWithSGD} //线性回归库
import org.apache.spark.{ SparkContext, SparkConf} //环境配置库
import org.apache.spark.mllib.linalg.Vectors //向量
import org.apache.spark.mllib.regression.LabeledPoint //标注点
import org.apache.spark.mllib.util.MLUtils //用于加载和解析数据
