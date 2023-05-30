package com.example.audioclassification;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.res.AssetFileDescriptor;
import android.net.Uri;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;


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
    private float[] preprocessAudio(Uri audioUri) {
        float[] preprocessedData = new float[30 * 150 * 1];  // Initialize preprocessedData array

        try {
            ContentResolver contentResolver = context.getContentResolver();
            AssetFileDescriptor fileDescriptor = contentResolver.openAssetFileDescriptor(audioUri, "r");
            FileInputStream inputStream = fileDescriptor.createInputStream();
            byte[] audioBytes = new byte[inputStream.available()];
            inputStream.read(audioBytes);

            // Convert audioBytes to short array (audioSamples)
            short[] audioSamples = new short[audioBytes.length / 2];  // Assuming audioBytes are 16-bit PCM samples (2 bytes per sample)
            ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(audioSamples);

            // Resize audioSamples to match (30, 150, 1) shape
            float[][][] mfccs = resizeArray(audioSamples);

            // Calculate mean and std from mfccs
            float mean = calculateMean(mfccs);
            float std = calculateStd(mfccs);

            // Standardize mfccs
            standardize(mfccs, mean, std);

            // Flatten mfccs to obtain the preprocessed audio data
            int index = 0;
            for (int i = 0; i < 30; i++) {
                for (int j = 0; j < 150; j++) {
                    preprocessedData[index] = mfccs[i][j][0];
                    index++;
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return preprocessedData;
    }

    // Helper method to resize audioSamples to (30, 150, 1) shape
    private float[][][] resizeArray(short[] audioSamples) {
        float[][][] mfccs = new float[30][150][1];

        int index = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                if (index < audioSamples.length) {
                    mfccs[i][j][0] = audioSamples[index] / 32768.0f;  // Normalize the audio sample to the range of [-1.0, 1.0]
                    index++;
                } else {
                    mfccs[i][j][0] = 0.0f;  // Pad with zeros if audioSamples is shorter than 30x150x1
                }
            }
        }

        return mfccs;
    }

    // Helper method to calculate mean from mfccs
    private float calculateMean(float[][][] mfccs) {
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                sum += mfccs[i][j][0];
                count++;
            }
        }
        return sum / count;
    }

    // Helper method to calculate standard deviation from mfccs
    private float calculateStd(float[][][] mfccs) {
        float mean = calculateMean(mfccs);
        float sum = 0.0f;
        int count = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                float diff = mfccs[i][j][0] - mean;
                sum += diff * diff;
                count++;
            }
        }
        float variance = sum / count;
        return (float) Math.sqrt(variance);
    }

    // Helper method to standardize mfccs using given mean and std
    private void standardize(float[][][] mfccs, float mean, float std) {
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 150; j++) {
                mfccs[i][j][0] = (mfccs[i][j][0] - mean) / std;
            }
        }
    }
    // ----------------------------------------------------

    // ---- Load model file ----
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        String MODEL_NAME = "audio.tflite";
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    // ----------------------------------------------------

    // ---- Audio classification ----
    public String classifyAudio(Uri audioUri) {
        // Preprocess the audio data
        float[] preprocessedData = preprocessAudio(audioUri);

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