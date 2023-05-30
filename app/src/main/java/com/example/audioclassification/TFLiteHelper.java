package com.example.audioclassification;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;



public class TFLiteHelper {

    private List<String> labels;
    private Interpreter tflite;

    private MappedByteBuffer tfliteModel;
    private TensorBuffer inputBuffer;
    private TensorBuffer outputBuffer;

    private Activity context;

    TFLiteHelper(Activity context) {
        this.context = context;
    }

    // ---- Inisialisasi TensorFlow Lite Interpreter ----

    void init() {
        try {
            Interpreter.Options opt = new Interpreter.Options();
            tflite = new Interpreter(loadModelFile(context), opt);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    // ----------------------------------------------------

    // ---- Preprocessing audio ----
    private float[] preprocessAudio(float[] audioData) {
        // Convert audioData to 2D array (mfccs)
        float[][] mfccs = new float[30][150];

        // Reshape audioData to match mfccs shape
        int index = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                if (index < audioData.length) {
                    mfccs[i][j] = audioData[index];
                    index++;
                } else {
                    mfccs[i][j] = 0.0f;  // Pad with zeros if audioData is shorter than 30x150
                }
            }
        }

        // Flatten mfccs to obtain the preprocessed audio data
        float[] preprocessedData = new float[30 * 150];
        index = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                preprocessedData[index] = mfccs[i][j];
                index++;
            }
        }

        return preprocessedData;
    }
    // ----------------------------------------------------

    // ---- Load model file ----
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        String MODEL_NAME = "audio_classification_model.tflite";
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    // ----------------------------------------------------

    // ---- Audio classification ----
    public String classifyAudio(float[] audioData) {
        // Preprocess the audio data
        float[] preprocessedData = preprocessAudio(audioData);

        // Prepare input and output buffers
        int inputTensorIndex = 0;
        int[] inputShape = tflite.getInputTensor(inputTensorIndex).shape();
        DataType inputDataType = tflite.getInputTensor(inputTensorIndex).dataType();

        int outputTensorIndex = 0;
        int[] outputShape = tflite.getOutputTensor(outputTensorIndex).shape();
        DataType outputDataType = tflite.getOutputTensor(outputTensorIndex).dataType();

        inputBuffer = TensorBuffer.createFixedSize(inputShape, inputDataType);
        outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType);

        // Set input data
        inputBuffer.loadArray(preprocessedData);

        // Run inference
        tflite.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());

        // Postprocess the output
        return postprocessOutput();
    }
    // ----------------------------------------------------

    // ---- Postprocessing output ----
    private String postprocessOutput() {
        try {
            labels = FileUtil.loadLabels(context, "vegs.txt");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        float[] probabilities = outputBuffer.getFloatArray();

        int maxIndex = 0;
        float maxValue = probabilities[0];

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxValue) {
                maxIndex = i;
                maxValue = probabilities[i];
            }
        }

        return labels.get(maxIndex);
    }
    // ----------------------------------------------------
}