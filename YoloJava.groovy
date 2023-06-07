import java.awt.image.BufferedImage

import javax.imageio.ImageIO

import com.neuronrobotics.bowlerkernel.djl.ImagePredictorType
import com.neuronrobotics.bowlerkernel.djl.PredictorFactory

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
HttpURLConnection connection = null;
connection = (HttpURLConnection) new URL(url).openConnection();
connection.connect();
BufferedImage img = ImageIO.read(connection.getInputStream());
connection.disconnect();

def availibleModels=ModelZoo.listModels();
println availibleModels



ZooModel<BufferedImage, DetectedObjects> model = PredictorFactory.imageContentsFactory(ImagePredictorType.yolov5);

Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()
DetectedObjects detection = predictor.predict(img);
System.out.println(detection);