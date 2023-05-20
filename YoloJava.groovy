import java.awt.image.BufferedImage

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.BufferedImageUtils;
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


Criteria<BufferedImage, DetectedObjects> criteria =
		Criteria.builder()
		.optApplication(Application.CV.OBJECT_DETECTION)
		.setTypes(BufferedImage.class, DetectedObjects.class)
		.optFilter("backbone", "resnet50")
		.optProgress(new ProgressBar())
		.build();


ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)

Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()
DetectedObjects detection = predictor.predict(img);
System.out.println(detection);