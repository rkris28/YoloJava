import java.awt.image.BufferedImage

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;



println "Starting Yolo PyTorch Java example"
String url = "https://avatars.githubusercontent.com/u/1254726?v=4";
BufferedImage img = BufferedImageUtils.fromUrl(url);

def availibleModels=ModelZoo.listModels();
println availibleModels



ZooModel<BufferedImage, DetectedObjects> model = PredictorFactory.imageContentsFactory(ImagePredictorType.yolov5);

Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()
DetectedObjects detection = predictor.predict(img);
System.out.println(detection);