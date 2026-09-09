package com.bmiapp.estimator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import kotlin.math.min
import kotlin.math.max
import java.io.IOException

data class PersonDetection(
    val boundingBox: RectF,
    val confidence: Float,
    val label: String = "person",
    val bboxAreaRatio: Float = 0f
)

data class Quintuple<A, B, C, D, E>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
    val fifth: E
)

class EfficientDetPersonDetector(private val context: Context) {
    
    private var objectDetector: ObjectDetector? = null
    // We'll create the image processor dynamically to preserve aspect ratio
    
    companion object {
        private const val PERSON_CLASS_ID = 0  // COCO dataset person class
        private const val MIN_CONFIDENCE = 0.5f
        private const val MAX_RESULTS = 10
    }
    
    init {
        setupDetector()
    }
    
    /**
     * Preprocess image to exactly 640x640 with padding, similar to bounding_box_detection.py
     * Returns: (processedImage, scaleX, scaleY, offsetX, offsetY)
     */
    private fun preprocessImagePreserveAspectRatio(bitmap: Bitmap): Quintuple<TensorImage, Float, Float, Float, Float> {
        val origWidth = bitmap.width
        val origHeight = bitmap.height
        
        // Define target size (EfficientDet-Lite4 expects 640x640)
        val targetSize = 640
        
        // Compute scaling factor to fit within 640x640
        val scale = min(targetSize.toFloat() / origWidth, targetSize.toFloat() / origHeight)
        
        // Calculate new dimensions (will be <= 640x640)
        val newWidth = (origWidth * scale).toInt()
        val newHeight = (origHeight * scale).toInt()
        
        android.util.Log.d("EfficientDet", "Resizing from ${origWidth}x${origHeight} to ${newWidth}x${newHeight}")
        
        // Resize bitmap while preserving aspect ratio
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        
        // Create 640x640 bitmap with padding
        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, resizedBitmap.config)
        val canvas = android.graphics.Canvas(paddedBitmap)
        
        // Calculate padding to center the image
        val offsetX = (targetSize - newWidth) / 2f
        val offsetY = (targetSize - newHeight) / 2f
        
        // Draw the resized image centered on the padded bitmap
        canvas.drawBitmap(resizedBitmap, offsetX, offsetY, null)
        
        android.util.Log.d("EfficientDet", "Added padding: offsetX=$offsetX, offsetY=$offsetY")
        
        // Convert to TensorImage
        val tensorImage = TensorImage.fromBitmap(paddedBitmap)
        
        // Compute scaling factors for mapping back to original coordinates
        val scaleX = newWidth.toFloat() / origWidth
        val scaleY = newHeight.toFloat() / origHeight
        
        return Quintuple(tensorImage, scaleX, scaleY, offsetX, offsetY)
    }
    
    private fun setupDetector() {
        try {
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(MAX_RESULTS)
                .setScoreThreshold(MIN_CONFIDENCE)
                .build()
            
            // Use the built-in EfficientDet-Lite4 model from TensorFlow Lite Task Library
            objectDetector = ObjectDetector.createFromFileAndOptions(context, "efficientdet_lite4.tflite", options)
            
        } catch (e: IOException) {
            e.printStackTrace()
            // Fallback: try to create with default options
            try {
                objectDetector = ObjectDetector.createFromFileAndOptions(context, "efficientdet_lite4.tflite", ObjectDetector.ObjectDetectorOptions.builder().build())
            } catch (e2: IOException) {
                e2.printStackTrace()
            }
        }
    }
    
    fun detectPerson(bitmap: Bitmap): PersonDetection? {
        if (objectDetector == null) {
            android.util.Log.w("EfficientDet", "Object detector is null")
            return null
        }
        
        try {
            android.util.Log.d("EfficientDet", "Starting person detection on bitmap: ${bitmap.width}x${bitmap.height}")
            
            // Preprocess image using the same approach as bounding_box_detection.py
            val (processedImage, scaleX, scaleY, offsetX, offsetY) = preprocessImagePreserveAspectRatio(bitmap)
            android.util.Log.d("EfficientDet", "Image processed with scale factors: scaleX=$scaleX, scaleY=$scaleY, offsets: offsetX=$offsetX, offsetY=$offsetY")
            
            // Run detection
            val results = objectDetector?.detect(processedImage)
            
            android.util.Log.d("EfficientDet", "Detection results: ${results?.size} detections")
            
            // Find the person with highest confidence
            var bestPerson: PersonDetection? = null
            var bestConfidence = 0f
            
            results?.forEach { detection ->
                if (detection.categories.isNotEmpty()) {
                    val category = detection.categories[0]
                    android.util.Log.d("EfficientDet", "Detection: class=${category.index}, label=${category.label}, confidence=${category.score}")
                    
                    // Check if it's a person (class 0) and has good confidence
                    if (category.index == PERSON_CLASS_ID && category.score > bestConfidence) {
                        bestConfidence = category.score
                        
                        // Scale bounding box back to original image coordinates (accounting for padding)
                        val originalBoundingBox = RectF(
                            (detection.boundingBox.left - offsetX) / scaleX,
                            (detection.boundingBox.top - offsetY) / scaleY,
                            (detection.boundingBox.right - offsetX) / scaleX,
                            (detection.boundingBox.bottom - offsetY) / scaleY
                        )
                        
                        // Calculate aspect ratios for comparison
                        val detectedWidth = detection.boundingBox.right - detection.boundingBox.left
                        val detectedHeight = detection.boundingBox.bottom - detection.boundingBox.top
                        val detectedAspectRatio = detectedWidth / detectedHeight
                        
                        val originalWidth = originalBoundingBox.right - originalBoundingBox.left
                        val originalHeight = originalBoundingBox.bottom - originalBoundingBox.top
                        val originalAspectRatio = originalWidth / originalHeight
                        
                        android.util.Log.d("EfficientDet", "=== ASPECT RATIO COMPARISON ===")
                        android.util.Log.d("EfficientDet", "Detected box (640x640): ${detection.boundingBox}")
                        android.util.Log.d("EfficientDet", "Detected dimensions: ${detectedWidth}x${detectedHeight}")
                        android.util.Log.d("EfficientDet", "Detected aspect ratio: $detectedAspectRatio")
                        android.util.Log.d("EfficientDet", "Original box: ${originalBoundingBox}")
                        android.util.Log.d("EfficientDet", "Original dimensions: ${originalWidth}x${originalHeight}")
                        android.util.Log.d("EfficientDet", "Original aspect ratio: $originalAspectRatio")
                        android.util.Log.d("EfficientDet", "Aspect ratio preserved: ${kotlin.math.abs(detectedAspectRatio - originalAspectRatio) < 0.01}")
                        android.util.Log.d("EfficientDet", "=== END COMPARISON ===")
                        
                        // Calculate bounding box area ratio relative to original image
                        val originalImageArea = bitmap.width * bitmap.height
                        val bboxArea = originalWidth * originalHeight
                        val bboxAreaRatio = bboxArea / originalImageArea
                        
                        android.util.Log.d("EfficientDet", "Bbox area ratio: $bboxAreaRatio (bbox: ${bboxArea}, image: ${originalImageArea})")
                        
                        bestPerson = PersonDetection(
                            boundingBox = originalBoundingBox,
                            confidence = category.score,
                            label = category.label,
                            bboxAreaRatio = bboxAreaRatio
                        )
                        android.util.Log.d("EfficientDet", "Found person: original=${originalBoundingBox}, confidence: ${category.score}")
                    }
                }
            }
            
            if (results.isNullOrEmpty()) {
                android.util.Log.w("EfficientDet", "No detections found at all")
            }
            
            android.util.Log.d("EfficientDet", "Best person detection: $bestPerson")
            return bestPerson
            
        } catch (e: Exception) {
            android.util.Log.e("EfficientDet", "Error in person detection", e)
            e.printStackTrace()
            return null
        }
    }
    
    fun detectAllPersons(bitmap: Bitmap): List<PersonDetection> {
        if (objectDetector == null) {
            return emptyList()
        }
        
        try {
            // Preprocess image using the same approach as bounding_box_detection.py
            val (processedImage, scaleX, scaleY, offsetX, offsetY) = preprocessImagePreserveAspectRatio(bitmap)
            
            // Run detection
            val results = objectDetector?.detect(processedImage)
            
            // Collect all persons
            val persons = mutableListOf<PersonDetection>()
            
            results?.forEach { detection ->
                if (detection.categories.isNotEmpty()) {
                    val category = detection.categories[0]
                    // Check if it's a person (class 0)
                    if (category.index == PERSON_CLASS_ID) {
                        // Scale bounding box back to original image coordinates (accounting for padding)
                        val originalBoundingBox = RectF(
                            (detection.boundingBox.left - offsetX) / scaleX,
                            (detection.boundingBox.top - offsetY) / scaleY,
                            (detection.boundingBox.right - offsetX) / scaleX,
                            (detection.boundingBox.bottom - offsetY) / scaleY
                        )
                        
                        // Calculate bounding box area ratio relative to original image
                        val originalImageArea = bitmap.width * bitmap.height
                        val bboxWidth = originalBoundingBox.right - originalBoundingBox.left
                        val bboxHeight = originalBoundingBox.bottom - originalBoundingBox.top
                        val bboxArea = bboxWidth * bboxHeight
                        val bboxAreaRatio = bboxArea / originalImageArea
                        
                        persons.add(PersonDetection(
                            boundingBox = originalBoundingBox,
                            confidence = category.score,
                            label = category.label,
                            bboxAreaRatio = bboxAreaRatio
                        ))
                    }
                }
            }
            
            // Sort by confidence (highest first)
            return persons.sortedByDescending { it.confidence }
            
        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        }
    }
    
    fun close() {
        objectDetector?.close()
        objectDetector = null
    }
} 