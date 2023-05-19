@Grab(group='ai.djl', module='api', version='0.4.0')
@Grab(group='ai.djl', module='repository', version='0.4.0')
@Grab(group='ai.djl.pytorch', module='pytorch-model-zoo', version='0.4.0')
@Grab(group='ai.djl.pytorch', module='pytorch-native-auto', version='0.4.0')





    String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
    BufferedImage img = BufferedImageUtils.fromUrl(url);

    Criteria<BufferedImage, DetectedObjects> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(BufferedImage.class, DetectedObjects.class)
                    .optFilter("backbone", "resnet50")
                    .optProgress(new ProgressBar())
                    .build();

    try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
        try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(img);
            System.out.println(detection);
        }
    }
println "Starting Yolo PyTorch Java example"