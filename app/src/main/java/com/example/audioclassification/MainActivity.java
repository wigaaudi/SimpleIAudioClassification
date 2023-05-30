package com.example.audioclassification;

import android.content.Intent;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    Button playButton;
    Uri audioUri;
    Button classifyButton;
    TextView classifyText;

    TFLiteHelper tfLiteHelper;
    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        playButton = findViewById(R.id.playButton);
        classifyButton = findViewById(R.id.classify);
        classifyText = findViewById(R.id.classifytext);

        tfLiteHelper = new TFLiteHelper(this);
        tfLiteHelper.init();

        playButton.setOnClickListener(playAudioListener);
        classifyButton.setOnClickListener(classifyAudioListener);
    }

    View.OnClickListener playAudioListener = new View.OnClickListener() {
        @Override
        public void onClick(View view) {
            String SELECT_TYPE = "audio/*";
            String SELECT_AUDIO = "Select Audio";

            Intent intent = new Intent();
            intent.setType(SELECT_TYPE);
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, SELECT_AUDIO), 123);
        }
    };

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 123 && resultCode == RESULT_OK && data != null) {
            audioUri = data.getData();
            playAudioFromUri(audioUri);
        }
    }

    private void playAudioFromUri(Uri audioUri) {
        try {
            MediaPlayer mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(getApplicationContext(), audioUri);
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    View.OnClickListener classifyAudioListener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (audioUri != null) {
                float[] audioData = extractAudioDataFromUri(audioUri);
                String label = tfLiteHelper.classifyAudio(audioData);
                setLabel(label);
            }
        }
    };
    private float[] extractAudioDataFromUri(Uri audioUri) {
        float[] audioData = null;
        return audioData;
    }
    void setLabel(String label) {
        classifyText.setText(label);
    }
}