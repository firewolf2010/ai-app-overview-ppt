// MinimalDJLExample.java
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.repository.zoo.ModelZoo;

public class MinimalDJLExample {
    public static void main(String[] args) throws Exception {
        Criteria<String, Classifications> criteria = Criteria.builder()
            .setTypes(String.class, Classifications.class)
            .optEngine("PyTorch") // or "TensorFlow", or use ONNX Runtime
            .optModelUrls("https://djl-ai.s3.amazonaws.com/resources/test/model/sentiment.pth") // 示例 URL
            .build();

        try (ZooModel<String, Classifications> model = ModelZoo.loadModel(criteria);
             Predictor<String, Classifications> predictor = model.newPredictor()) {

            Classifications result = predictor.predict("I love this product!");
            System.out.println(result);
        } catch (TranslateException e) {
            e.printStackTrace();
        }
    }
}