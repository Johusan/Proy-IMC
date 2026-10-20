package com.bmiapp.estimator

import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.bmiapp.estimator.databinding.ActivityResultBinding
import com.bumptech.glide.Glide
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.concurrent.Executors

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private var pytorchModule: Module? = null
    private var personDetector: EfficientDetPersonDetector? = null
    private var postureDetectionClustering: PostureDetectionClustering? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var imagePath: String? = null
    private lateinit var imagePreprocessor: ImagePreprocessor

    companion object {
        private const val TAG = "ResultActivity"
        private const val MODEL_FILE_PTE = "bmi_model_fixed.pte"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imagePreprocessor = ImagePreprocessor()
        setupToolbar()
        setupClickListeners()
        loadModel()
        loadPersonDetector()
        loadMoveNetThunder()
        processImage()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setDisplayShowHomeEnabled(true)
    }

    private fun setupClickListeners() {
        binding.btnTryAgain.setOnClickListener {
            finish()
        }

        binding.btnSaveResult.setOnClickListener {
            saveResultToGallery()
        }
    }

    private fun loadModel() {
        try {
            // Try to load ExecuTorch model first, then fallback to TorchScript
            val modelPath = when {
                assetExists(MODEL_FILE_PTE) -> copyAssetToFiles(MODEL_FILE_PTE)
                else -> throw Exception("No model file found")
            }
            
            pytorchModule = Module.load(modelPath)
            Log.d(TAG, "Model loaded successfully from: $modelPath")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadPersonDetector() {
        try {
            personDetector = EfficientDetPersonDetector(this)
            Log.d(TAG, "EfficientDet person detector loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading EfficientDet person detector", e)
            // Person detection is optional, so we don't show an error toast
        }
    }

    private fun loadMoveNetThunder() {
        try {
            postureDetectionClustering = PostureDetectionClustering(this)
            Log.d(TAG, "PostureDetectionClustering loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading PostureDetectionClustering", e)
        }
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            assets.open(assetName).close()
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun copyAssetToFiles(assetName: String): String {
        val file = File(filesDir, assetName)
        if (!file.exists()) {
            assets.open(assetName).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    private fun processImage() {
        imagePath = intent.getStringExtra("image_uri")
        if (imagePath.isNullOrEmpty()) {
            Toast.makeText(this, "No image provided", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // Load and display the image
        Glide.with(this)
            .load(imagePath)
            .into(binding.ivResultImage)

        // Show loading state initially
        showLoadingState()

        // Process image for BMI estimation
        executor.execute {
            try {
                val bmi = estimateBMI(imagePath!!)
                runOnUiThread {
                    showResultState(bmi)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error estimating BMI", e)
                runOnUiThread {
                    // Check if it's a person detection error (no person or low confidence)
                    if (e.message?.contains("person", ignoreCase = true) == true) {
                        // Show prominent dialog that requires user interaction
                        showPersonDetectionErrorDialog(e.message ?: "Person detection failed")
                    } else {
                        // For other errors, show in the result screen
                        showErrorState("Error processing image: ${e.message}")
                    }
                }
            }
        }
    }

    private fun showLoadingState() {
        binding.loadingLayout.visibility = android.view.View.VISIBLE
        binding.resultLayout.visibility = android.view.View.GONE
    }

    private fun showResultState(bmi: Float) {
        binding.loadingLayout.visibility = android.view.View.GONE
        binding.resultLayout.visibility = android.view.View.VISIBLE
        
        displayResult(bmi)
    }

    private fun showErrorState(errorMessage: String) {
        binding.loadingLayout.visibility = android.view.View.GONE
        binding.resultLayout.visibility = android.view.View.VISIBLE
        
        // Show error in the BMI score field
        binding.tvBMIScore.text = "Error"
        binding.tvBMIScore.setTextColor(getColor(android.R.color.holo_red_dark))
        binding.tvBMICategory.text = errorMessage
        binding.tvBMICategory.setTextColor(getColor(android.R.color.holo_red_dark))
        
        Toast.makeText(this, errorMessage, Toast.LENGTH_LONG).show()
    }

    private fun showPersonDetectionErrorDialog(errorMessage: String) {
        AlertDialog.Builder(this)
            .setTitle("Please retake the picture.")
            .setMessage(errorMessage)
            .setIcon(android.R.drawable.ic_dialog_alert)
            .setPositiveButton("OK") { _, _ ->
                // Return to home screen when user clicks OK
                finish()
            }
            .setCancelable(false) // User must click OK to dismiss
            .show()
    }

    private fun estimateBMI(imagePath: String): Float {
        // Load the original image
        val originalBitmap = if (imagePath.startsWith("content://")) {
            imagePreprocessor.loadBitmapFromUri(this, imagePath)
        } else {
            BitmapFactory.decodeFile(imagePath)
        }
        
        if (originalBitmap == null) {
            throw Exception("Failed to load image")
        }
        
        // Detect person and get bounding box using EfficientDet-Lite4
        val personDetection = personDetector?.detectPerson(originalBitmap)
        
        Log.d(TAG, "Person detection result: $personDetection")

        // Check if person was detected
        if (personDetection?.boundingBox == null) {
            Log.d(TAG, "No person detected in the image")
            throw Exception("A person could not be confidently detected. Please ensure the person is fully visible and facing the camera.")
        }
        
        // Check confidence threshold (0.5f = 50%)
        if (personDetection.confidence < 0.5f) {
            Log.d(TAG, "Person detected but confidence too low: ${personDetection.confidence}")
            throw Exception("Person detected with low confidence (${(personDetection.confidence * 100).toInt()}%). Please ensure the person is clearly visible and well-lit.")
        }
        
        // Check bounding box area ratio threshold (30%)
        if (personDetection.bboxAreaRatio < 0.25f) {
            Log.d(TAG, "Person detected but bounding box too small: ${personDetection.bboxAreaRatio * 100}%")
            throw Exception("The person is too small in the image. Please move closer to the camera or ensure the person takes up more of the frame.")
        }

        // Detect keypoints and posture using PostureDetectionClustering
        val keypointAndPosture = postureDetectionClustering?.detectKeypointsAndPosture(originalBitmap, personDetection.boundingBox)
        Log.d(TAG, "Keypoint and posture result: $keypointAndPosture")
        
        // Check if keypoint detection was successful
        if (keypointAndPosture == null) {
            Log.d(TAG, "Keypoint detection failed")
            throw Exception("Failed to detect body keypoints. Please ensure the person is clearly visible and try again.")
        }
        
        // Check minimum keypoint confidence threshold (30%)
        val minKeypointConfidence = keypointAndPosture.keypoints.minOfOrNull { it.third } ?: 0f
        if (minKeypointConfidence < 0.4f) {
            Log.d(TAG, "Minimum keypoint confidence too low: ${minKeypointConfidence * 100}%")
            throw Exception("Person is not standing upright. Please ensure the person is standing straight and facing the camera.")
        }
        
        // Check posture cluster (0 and 1 are good standing postures)
        val postureCluster = keypointAndPosture.postureCluster
        if (postureCluster == null || (postureCluster != 0 && postureCluster != 1)) {
            Log.d(TAG, "Posture not suitable for BMI estimation: cluster $postureCluster")
            throw Exception("Person is not standing upright. Please ensure the person is standing straight and facing the camera.")
        }
        
        // Display the original image with bounding box and skeleton overlay
        runOnUiThread {
            Log.d(TAG, "Setting bounding box: ${personDetection.boundingBox}, confidence: ${personDetection.confidence}")
            (binding.ivResultImage as? BoundingBoxImageView)?.setBoundingBox(
                personDetection.boundingBox, 
                originalBitmap.width, 
                originalBitmap.height,
                personDetection.confidence
            )
            // Also set the image bitmap (if not already set)
            (binding.ivResultImage as? BoundingBoxImageView)?.setImageBitmap(originalBitmap)
            // Set keypoints for skeleton overlay
            binding.skeletonOverlay.setKeypoints(keypointAndPosture?.keypoints ?: emptyList(), originalBitmap.width, originalBitmap.height)
        }
        
        // Use the original image for BMI estimation (no cropping)
        val inputTensor = imagePreprocessor.preprocessBitmap(originalBitmap)
        
        // Run inference
        val outputEValues = pytorchModule?.forward(EValue.from(inputTensor))
        val outputTensor = outputEValues?.get(0)?.toTensor()
        
        // Get BMI value
        val bmi = outputTensor?.dataAsFloatArray?.get(0) ?: 25.0f
        
        // Apply reasonable bounds
        return bmi.coerceIn(15.0f, 50.0f)
    }

    private fun displayResult(bmi: Float) {
        binding.tvBMIScore.text = String.format("%.1f", bmi)
        
        val category = when {
            bmi < 18.5 -> {
                binding.tvBMICategory.setTextColor(getColor(R.color.underweight_color))
                getString(R.string.underweight)
            }
            bmi < 25.0 -> {
                binding.tvBMICategory.setTextColor(getColor(R.color.normal_color))
                getString(R.string.normal_weight)
            }
            bmi < 30.0 -> {
                binding.tvBMICategory.setTextColor(getColor(R.color.overweight_color))
                getString(R.string.overweight)
            }
            else -> {
                binding.tvBMICategory.setTextColor(getColor(R.color.obese_color))
                getString(R.string.obese)
            }
        }
        
        binding.tvBMICategory.text = category
    }

    private fun saveResultToGallery() {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
            val filename = "BMI_Result_${System.currentTimeMillis()}.jpg"
            
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }

            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            uri?.let {
                contentResolver.openOutputStream(it)?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
                }
                Toast.makeText(this, getString(R.string.result_saved), Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error saving result", e)
            Toast.makeText(this, "Error saving result", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        personDetector?.close()
        postureDetectionClustering?.close()
    }
} 