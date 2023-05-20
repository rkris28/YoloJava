import ai.djl.modality.Classifications.Classification

@Grab(group='org.openpnp', module='opencv', version='4.7.0-0')



@Grab(group='ai.djl', module='api', version='0.4.0')
@Grab(group='ai.djl', module='repository', version='0.4.0')
@Grab(group='ai.djl.pytorch', module='pytorch-model-zoo', version='0.4.0')

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import java.io.FileNotFoundException;
import java.io.IOException;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Tab
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;

import org.opencv.videoio.VideoCapture;

import com.neuronrobotics.bowlerkernel.djl.FaceDetectionTranslator
import com.neuronrobotics.bowlerkernel.djl.ImagePredictorType
import com.neuronrobotics.bowlerkernel.djl.PredictorFactory
import com.neuronrobotics.bowlerstudio.BowlerStudio
import com.neuronrobotics.bowlerstudio.BowlerStudioController
import com.neuronrobotics.bowlerstudio.scripting.ScriptingEngine

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.output.BoundingBox;

import com.neuronrobotics.bowlerkernel.djl.ImagePredictorType;
import com.neuronrobotics.bowlerkernel.djl.PredictorFactory;

// For proper execution of native libraries
// Core.NATIVE_LIBRARY_NAME must be loaded before
// calling any of the opencv methods
try {
	nu.pattern.OpenCV.loadLocally()
}catch(Throwable t) {
	BowlerStudio.printStackTrace(t)
	return
}
Mat matrix =new Mat();
VideoCapture capture = new VideoCapture(0);
capture.open(0)
WritableImage img = null;
CascadeClassifier faceCascade = new CascadeClassifier();
//File fileFromGit = ScriptingEngine.fileFromGit("https://github.com/CommonWealthRobotics/harr-cascade-archive.git", "resources/haarcascades/haarcascade_frontalcatface_extended.xml")
File fileFromGit = ScriptingEngine.fileFromGit("https://github.com/CommonWealthRobotics/harr-cascade-archive.git", "resources/haarcascades/haarcascade_frontalface_default.xml")

faceCascade.load(fileFromGit.getAbsolutePath());
int absoluteFaceSize=0;
Tab t =new Tab()
boolean run = true

ZooModel<Image, DetectedObjects> model  = PredictorFactory.imageContentsFactory(ImagePredictorType.yolov5);
Predictor<Image, DetectedObjects> predictor =model.newPredictor()
factory=ImageFactory.getInstance()
while(!Thread.interrupted() && run) {
	//Thread.sleep(16)
	try {
		// If camera is opened
		if( capture.isOpened()) {
			//println "Camera Open"
			// If there is next video frame
			if (capture.read(matrix)) {
				MatOfRect faces = new MatOfRect();
				Mat grayFrame = new Mat();
				// face cascade classifier

				// convert the frame in gray scale
				Imgproc.cvtColor(matrix, grayFrame, Imgproc.COLOR_BGR2GRAY);
				// equalize the frame histogram to improve the result
				Imgproc.equalizeHist(grayFrame, grayFrame);

				// compute minimum face size (20% of the frame height, in our case)
				if (absoluteFaceSize == 0)
				{
					int height = grayFrame.rows();
					if (Math.round(height * 0.2f) > 0)
					{
						absoluteFaceSize = Math.round(height * 0.2f);
					}
				}
				// each rectangle in faces is a face: draw them!

				//println "Capture success"
				// Creating BuffredImage from the matrix
				BufferedImage image = new BufferedImage(matrix.width(),
						matrix.height(), BufferedImage.TYPE_3BYTE_BGR);

				WritableRaster raster = image.getRaster();
				DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
				byte[] data = dataBuffer.getData();
				matrix.get(0, 0, data);
				
				DetectedObjects detection = predictor.predict(factory.fromImage(image));
				List<DetectedObject> items = detection.items();
				Rect[] facesArray = new Rect[items.size()];
				for (int i = 0; i < items.size(); i++) {
					DetectedObject c = items.get(i);
					BoundingBox cGetBoundingBox = c.getBoundingBox();
					def topLeft = cGetBoundingBox.getPoint();
					def rect = cGetBoundingBox.getBounds();
					facesArray[i]=new Rect(topLeft.getX()*matrix.width(),topLeft.getY()*matrix.height(),rect.getWidth()*matrix.width() ,rect.getHeight()*matrix.height())
					System.out.println(c);
					System.out.println("Name: "+c.getClassName() +" probability "+c.getProbability()+" center x "+topLeft.getX()+" center y "+topLeft.getY()+" rect h"+rect.getHeight()+" rect w"+rect.getWidth() );
					Imgproc.rectangle(matrix, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
					Imgproc.putText(matrix, c.getClassName(), new Point(topLeft.getX()*matrix.width(),topLeft.getY()*matrix.height()-5), 3,1,  new Scalar(0, 255, 0));
				}
				matrix.get(0, 0, data);
				//println detection

				// Creating the Writable Image
				if(img==null) {
					img = SwingFXUtils.toFXImage(image, null);

					t=new Tab("Imace capture ");
					t.setContent(new ImageView(img))
					BowlerStudioController.addObject(t, null);
				}else{
					SwingFXUtils.toFXImage(image, img);
				}
			}
		}
	}catch(Error tr) {
		BowlerStudio.printStackTrace(tr)
		break;
	}

}
BowlerStudioController.removeObject(t)
capture.release()
println "clean exit and closed camera"

