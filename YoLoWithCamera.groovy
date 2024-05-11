import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.neuronrobotics.bowlerkernel.djl.ImagePredictorType;
import com.neuronrobotics.bowlerkernel.djl.PredictorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;

void saveBoundingBoxImage(Image img, DetectedObjects detection, String type) throws Exception {
	Path outputDir = Paths.get("build/output");
	Files.createDirectories(outputDir);

	img.drawBoundingBoxes(detection);

	Path imagePath = outputDir.resolve(type + ".png").toAbsolutePath();
	img.save(Files.newOutputStream(imagePath), "png");
	System.out.println("Face detection result image has been saved in: {} " + imagePath);
}

System.err.println(Thread.currentThread().getStackTrace()[1].getMethodName());

Predictor<Image, DetectedObjects> predictor = PredictorFactory.imageContentsFactory(ImagePredictorType.yolov5);
for (int i = 0; i < 3; i++) {
	Image input = ImageFactory.getInstance()
			.fromUrl("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");

	DetectedObjects objects = predictor.predict(input);
	saveBoundingBoxImage(input, objects, "yolov5");
}
	