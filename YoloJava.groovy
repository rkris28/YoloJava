@Grab(group='net.java.dev.jna', module='jna', version='5.7.0')
@Grab(group='com.alphacephei', module='vosk', version='0.3.45')
@Grab(group='org.openpnp', module='opencv', version='4.7.0-0')

@Grab(group='ai.djl', module='api', version='0.4.0')
@Grab(group='ai.djl', module='repository', version='0.4.0')
@Grab(group='ai.djl.pytorch', module='pytorch-model-zoo', version='0.4.0')
//@Grab(group='ai.djl.pytorch', module='pytorch-native-auto', version='0.4.0')

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


String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
BufferedImage img = BufferedImageUtils.fromUrl(url);

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

println "Starting Yolo PyTorch Java example"