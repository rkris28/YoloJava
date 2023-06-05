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

double calculateDistance(float[] embedding1, float[] embedding2) {
	double sum = 0.0;
	for (int i = 0; i < embedding1.length; i++) {
		sum += Math.pow(embedding1[i] - embedding2[i], 2);
	}
	return Math.sqrt(sum);
}

public static float calculSimilarFaceFeature(float[] feature1, ArrayList<float[]> people) {
	float ret = 0.0f;
	float mod1 = 0.0f;
	float mod2 = 0.0f;
	int length = feature1.length;
	for(int j=0;j<people.size();j++) {
		float[] feature2 = people.get(j);
		for (int i = 0; i < length; ++i) {
			ret += feature1[i] * feature2[i];
			mod1 += feature1[i] * feature1[i];
			mod2 += feature2[i] * feature2[i];
		}
	}
	return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
}
VideoCapture capture = OpenCVManager.get(0).getCapture()

Mat matrix =new Mat();
WritableImage img = null;
CascadeClassifier faceCascade = new CascadeClassifier();

int absoluteFaceSize=0;
Tab t =new Tab()
boolean run = true
VBox workingMemory = new VBox()
Predictor<Image, DetectedObjects> predictor =PredictorFactory.imageContentsFactory(ImagePredictorType.ultranet)
factory=ImageFactory.getInstance()


class UniquePerson{
	String name=""
	ArrayList<float[]> features=[];
	String referenceImageLocation;
	Image memory;
	int timesSeen = 1;
	long time=System.currentTimeMillis()
	HBox box
	Label percent
	double confidenceTarget = 0.7
}
HashMap<BufferedImage,org.opencv.core.Point> factoryFromImage=null
HashMap<UniquePerson,org.opencv.core.Point> currentPersons=null

new Thread({

	ArrayList<UniquePerson> knownPeople =[]
	Predictor<Image, float[]> features = PredictorFactory.faceFeatureFactory()
	JniUtils.setGraphExecutorOptimize(false);
	float confidence=0.88
	long timeout = 30000
	long countPeople=1
	int numberOfTrainingHashes =50
	while(!Thread.interrupted() && run) {
		try {
			if(factoryFromImage==null) {
				Thread.sleep(16)
				continue;
			}
			HashMap<BufferedImage,org.opencv.core.Point> local = new HashMap<>()
			local.putAll(factoryFromImage)
			HashMap<UniquePerson,org.opencv.core.Point> tmpPersons = new HashMap<>()
			for(BufferedImage imgBuff:local.keySet()) {
				ai.djl.modality.cv.Image cmp= factory.fromImage(imgBuff)
				def point = local.get(imgBuff);
				//println "Processing new image "
				float[] id = features.predict(cmp);
				boolean found=false;
				def duplicates =[]

				for(UniquePerson pp:knownPeople) {
					UniquePerson p=pp
					int count= 0;
					//for(int i=0;i<p.features.size();i++) {
					//float[] featureFloats =p.features.get(i);
					float result = calculSimilarFaceFeature(id, p.features)
					println "Difference from "+p.name+" is "+result
					if (result>p.confidenceTarget) {
						if(found) {
							duplicates.add(p)
						}else {
							count++;

							p.timesSeen++
							found=true;
							if(p.timesSeen>2)
								tmpPersons.put(p, point)
							if(p.timesSeen==3) {
								//on the third seen, display
								WritableImage tmpImg = SwingFXUtils.toFXImage(imgBuff, null);
								p.box.getChildren().addAll(new ImageView(tmpImg))
								p.box.getChildren().addAll(new Label(p.name))
								p.percent=new Label()
								p.box.getChildren().addAll(p.percent)
								BowlerStudio.runLater({workingMemory.getChildren().add(p.box)})
							}
							p.time=System.currentTimeMillis()
							//if(result<(confidence+0.01))
							if(p.features.size()<numberOfTrainingHashes) {
								p.features.add(id)
								int percent=(int)(((double)p.features.size())/((double)numberOfTrainingHashes)*100)
								println "Trained "+percent
								BowlerStudio.runLater({p.percent.setText(" : Trained "+percent+"%")})
								p.confidenceTarget = confidence;
								if(p.features.size()==numberOfTrainingHashes) {
									println " Trained "+p.name
									BowlerStudio.runLater({p.box.getChildren().addAll(new Label(" Done! "))})
	
								}
							}
						}
					}
				}
				for(int i=0;i<knownPeople.size();i++) {
					UniquePerson p = knownPeople.get(i)
					if((System.currentTimeMillis()-p.time)>timeout&& p.timesSeen<numberOfTrainingHashes) {
						duplicates.add(p)
					}
				}
				for(UniquePerson p:duplicates) {
					knownPeople.remove(p)
					BowlerStudio.runLater({workingMemory.getChildren().remove(p.box)})
					println "Removing "+p.name
				}

				if(found==false) {
					UniquePerson p = new UniquePerson();
					p.features.add(id);
					p.name="Person "+(countPeople++)
					String tmpDirsLocation = System.getProperty("java.io.tmpdir")+"/idFiles/"+p.name+".jpeg";
					p.box= new HBox()
					p.referenceImageLocation=tmpDirsLocation
					println "New person found! "+tmpDirsLocation
					knownPeople.add(p)
				}
			}
			currentPersons=tmpPersons
		}catch(Throwable tr) {
			BowlerStudio.printStackTrace(tr)
			//run=false;
		}
	}

}).start()

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
					factoryFromImage=facePlaces
				}
				if(currentPersons!=null) {
					HashMap<UniquePerson,org.opencv.core.Point> lhm = new HashMap<>()
					lhm.putAll(currentPersons)
					for(UniquePerson currentPerson:lhm.keySet()) {
						def p = lhm.get(currentPerson)
						Imgproc.putText(matrix, currentPerson.name,p , 3,1,  new Scalar(0, 255, 0));
					}
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

