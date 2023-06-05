import ai.djl.inference.Predictor
import ai.djl.modality.Classifications.Classification
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.output.BoundingBox
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject
import ai.djl.pytorch.jni.JniUtils
import ai.djl.modality.cv.output.Landmark
import ai.djl.repository.zoo.ZooModel

//@Grab(group='org.tensorflow', module='tensorflow', version='1.15.0')

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
import java.awt.image.TileObserver
import java.awt.image.WritableRaster;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files
import java.nio.file.Path
import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Label
import javafx.scene.control.Tab
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox

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
import com.neuronrobotics.bowlerstudio.opencv.OpenCVManager;
import com.neuronrobotics.bowlerkernel.djl.UniquePersonFactory;
import com.neuronrobotics.bowlerkernel.djl.UniquePerson;

UniquePersonFactory   upf = UniquePersonFactory.get();

VideoCapture capture = OpenCVManager.get(0).getCapture()

Mat matrix =new Mat();
WritableImage img = null;
CascadeClassifier faceCascade = new CascadeClassifier();

int absoluteFaceSize=0;
Tab t =new Tab()
boolean run = true
VBox workingMemory = upf.getWorkingMemory();
Predictor<Image, DetectedObjects> predictor =PredictorFactory.imageContentsFactory(ImagePredictorType.ultranet)
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
				BufferedImage imageFull = new BufferedImage(matrix.width(),
						matrix.height(), BufferedImage.TYPE_3BYTE_BGR);

				WritableRaster r = imageFull.getRaster();
				DataBufferByte db = (DataBufferByte) r.getDataBuffer();
				byte[] dataFull = db.getData();
				matrix.get(0, 0, dataFull);
				ai.djl.modality.cv.Image tmp= factory.fromImage(imageFull)


				DetectedObjects detection = predictor.predict(tmp);
				List<DetectedObject> items = detection.items();
				Rect[] facesArray = new Rect[items.size()];
				HashMap<BufferedImage,org.opencv.core.Point> facePlaces = new HashMap<>()

				for (int detectionIndex = 0; detectionIndex < items.size(); detectionIndex++) {

					DetectedObject c = items.get(detectionIndex);
					BoundingBox cGetBoundingBox = c.getBoundingBox();
					ai.djl.modality.cv.output.Point topLeft = cGetBoundingBox.getPoint();
					ai.djl.modality.cv.output.Rectangle rect = cGetBoundingBox.getBounds();
					facesArray[detectionIndex]=new Rect(topLeft.getX()*matrix.width(),topLeft.getY()*matrix.height(),rect.getWidth()*matrix.width() ,rect.getHeight()*matrix.height())

					Rect crop =facesArray[detectionIndex]
					try {
						Mat image_roi = new Mat(matrix,crop);
						BufferedImage image = new BufferedImage((int)rect.getWidth()*matrix.width() ,(int)rect.getHeight()*matrix.height(), BufferedImage.TYPE_3BYTE_BGR);
						WritableRaster raster = image.getRaster();
						DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
						byte[] data = dataBuffer.getData();
						image_roi.get(0,0, data);


						nameLoc = new Point(topLeft.getX()*matrix.width(),topLeft.getY()*matrix.height()-5)
						facePlaces.put(image, nameLoc)
					}catch(Throwable tr) {
						BowlerStudio.printStackTrace(tr)
						continue;
					}
					// draw the face in the corner
					//matrix.get(0, 0, data);


					Iterator<ai.djl.modality.cv.output.Point> path = cGetBoundingBox.getPath().iterator();
					ArrayList<ai.djl.modality.cv.output.Point> list = new ArrayList<>();
					// sort into an ordered list
					for(ai.djl.modality.cv.output.Point p:path) {
						boolean added=false;
						for(int j=0;j<list.size();j++) {
							if(p.getY()<list.get(j).getY()) {
								list.add(j, p);
								added=true;
								break;
							}
						}
						if(!added)
							list.add(p)
					}
					if(list.size()>=5) {
						double tiltAngle = 0
						def left = list.get(0)
						def right=list.get(1)

						if(left.getY()!=right.getY()) {
							double y=left.getY()-right.getY()
							double x=left.getX()-right.getX()
							tiltAngle=Math.toDegrees(Math.atan2(y, x))
							if(tiltAngle<-90) {
								tiltAngle+=180
							}
							//println "Tilt angle = "+tiltAngle
						}else {
							// angle is 0, they are the same
						}
					}

					//lm.get
					//System.out.println(c);
					//System.out.println("Name: "+c.getClassName() +" probability "+c.getProbability()+" center x "+topLeft.getX()+" center y "+topLeft.getY()+" rect h"+rect.getHeight()+" rect w"+rect.getWidth() );
					Imgproc.rectangle(matrix, facesArray[detectionIndex].tl(), facesArray[detectionIndex].br(), new Scalar(0, 255, 0), 3);

					if(list.size()>3) {
						for(int j=0;j<2;j++) {
							ai.djl.modality.cv.output.Point p= list.get(j)
							Imgproc.circle(matrix, new Point(p.getX(),p.getY()), 3, new Scalar(255, 0, 0))
						}
						ai.djl.modality.cv.output.Point n= list.get(2)
						Imgproc.circle(matrix, new Point(n.getX(),n.getY()), 5, new Scalar(0, 0, 255))
						for(int j=list.size()-2;j<list.size();j++) {
							ai.djl.modality.cv.output.Point p= list.get(j)
							Imgproc.circle(matrix, new Point(p.getX(),p.getY()), 3, new Scalar(255, 0, 255))
						}
					}
				}
				if( items.size()==0) {
					factoryFromImage=null

				}else {

					upf.setFactoryFromImage(facePlaces)
				}

				HashMap<UniquePerson,org.opencv.core.Point> lhm =  upf.getCurrentPersons()
				if(lhm!=null) {
					for(UniquePerson currentPerson:lhm.keySet()) {
						def p = lhm.get(currentPerson)
						Imgproc.putText(matrix, currentPerson.name,p , 3,1,  new Scalar(0, 255, 0));
					}
					lhm.clear()
					lhm=null
				}
				// Write Matrix wiht image, and detections ovelaid, onto the matrix
				matrix.get(0, 0, dataFull);
				//println detection

				// Creating the Writable Image
				if(img==null) {
					img = SwingFXUtils.toFXImage(imageFull, null);

					t=new Tab("Imace capture ");
					HBox content = new HBox()
					content.getChildren().add(new ImageView(img))
					content.getChildren().add(workingMemory)
					t.setContent(content)
					BowlerStudioController.addObject(t, null);
				}else{
					SwingFXUtils.toFXImage(imageFull, img);
				}
			}
		}else {
			println "Camera failed to open!"

			throw new RuntimeException("Camera failed!");
		}
	}catch(Throwable tr) {
		BowlerStudio.printStackTrace(tr)
		run=false;
		break;
	}

}
run=false;

BowlerStudioController.removeObject(t)

println "clean exit and closed camera"

