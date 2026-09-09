package com.bmiapp.estimator

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.channels.FileChannel
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.DataType
import com.google.gson.Gson
import java.io.InputStreamReader

// Data class for a single keypoint
// index: COCO keypoint index, x/y: pixel coordinates, score: confidence
data class KeypointDetection(
    val index: Int,
    val x: Float,
    val y: Float,
    val score: Float
)

data class KeypointAndPostureResult(
    val keypoints: List<Triple<Float, Float, Float>>,
    val postureCluster: Int?,
    val isGoodPosture: Boolean?,
    val postureConfidence: Double?
)

class PostureDetectionClustering(private val context: Context) {
    // Parameter data classes as inner classes
    data class ScalerParams(val mean: List<Double>, val scale: List<Double>)
    data class PCAParams(val components: List<List<Double>>, val mean: List<Double>)
    data class KMeansParams(val centers: List<List<Double>>)
    data class PostureModelParams(
        val scaler: ScalerParams,
        val pca: PCAParams,
        val kmeans: KMeansParams
    )
    private var interpreter: Interpreter? = null
    private val inputImageSize = 256
    private val numKeypoints = 17
    private var postureParams: PostureModelParams? = null

    // COCO keypoint order (MoveNet output order)
    // 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    // 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    // 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    // 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    // Expected order for scaler/PCA (your provided order)
    // 0: left_ankle-x, 1: left_ear-x, 2: left_elbow-x, 3: left_eye-x, 4: left_hip-x,
    // 5: left_knee-x, 6: left_shoulder-x, 7: left_wrist-x, 8: nose-x,
    // 9: right_ankle-x, 10: right_ear-x, 11: right_elbow-x, 12: right_eye-x,
    // 13: right_hip-x, 14: right_knee-x, 15: right_shoulder-x, 16: right_wrist-x,
    // 17: left_ankle-y, 18: left_ear-y, 19: left_elbow-y, 20: left_eye-y, 21: left_hip-y,
    // 22: left_knee-y, 23: left_shoulder-y, 24: left_wrist-y, 25: nose-y,
    // 26: right_ankle-y, 27: right_ear-y, 28: right_elbow-y, 29: right_eye-y,
    // 30: right_hip-y, 31: right_knee-y, 32: right_shoulder-y, 33: right_wrist-y

    // Mapping from MoveNet order to expected order
    private val keypointMapping = mapOf(
        // MoveNet index -> Expected index for x coordinates
        15 to 0,  // left_ankle -> left_ankle
        3 to 1,   // left_ear -> left_ear  
        7 to 2,   // left_elbow -> left_elbow
        1 to 3,   // left_eye -> left_eye
        11 to 4,  // left_hip -> left_hip
        13 to 5,  // left_knee -> left_knee
        5 to 6,   // left_shoulder -> left_shoulder
        9 to 7,   // left_wrist -> left_wrist
        0 to 8,   // nose -> nose
        16 to 9,  // right_ankle -> right_ankle
        4 to 10,  // right_ear -> right_ear
        8 to 11,  // right_elbow -> right_elbow
        2 to 12,  // right_eye -> right_eye
        12 to 13, // right_hip -> right_hip
        14 to 14, // right_knee -> right_knee
        6 to 15,  // right_shoulder -> right_shoulder
        10 to 16  // right_wrist -> right_wrist
    )

    init {
        setupInterpreter()
        loadPostureParams()
    }

    private fun setupInterpreter() {
        try {
            val assetFileDescriptor = context.assets.openFd("movenet_thunder_float16.tflite")
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            interpreter = Interpreter(modelBuffer)
            Log.d("PostureDetectionClustering", "Interpreter loaded successfully.")
        } catch (e: IOException) {
            Log.e("PostureDetectionClustering", "Failed to load model", e)
            interpreter = null
        }
    }

    private fun loadPostureParams() {
        try {
            val inputStream = context.assets.open("posture_model_params.json")
            val reader = InputStreamReader(inputStream)
            postureParams = Gson().fromJson(reader, PostureModelParams::class.java)
            reader.close()
        } catch (e: Exception) {
            Log.e("PostureDetectionClustering", "Failed to load posture model params", e)
            postureParams = null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, inputImageSize, inputImageSize, true)
    }

    /**
     * Reorder keypoints from MoveNet order to expected order for scaler/PCA
     */
    private fun reorderKeypointsForPosture(keypoints: List<Triple<Float, Float, Float>>): List<Pair<Float, Float>> {
        val reordered = MutableList(17) { Pair(0.0f, 0.0f) }
        
        // Log original MoveNet keypoint order for debugging
        Log.d("PostureDebug", "=== ORIGINAL MOVENET KEYPOINT ORDER ===")
        val movenetOrder = listOf(
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        )
        
        for ((i, keypoint) in keypoints.withIndex()) {
            val keypointName = movenetOrder.getOrElse(i) { "unknown_$i" }
            Log.d("PostureDebug", "$i: $keypointName -> (${keypoint.first}, ${keypoint.second})")
        }
        
        for ((movenetIndex, keypoint) in keypoints.withIndex()) {
            val expectedIndex = keypointMapping[movenetIndex]
            if (expectedIndex != null) {
                reordered[expectedIndex] = Pair(keypoint.first, keypoint.second)
            }
        }
        
        // Log reordered keypoints for debugging
        Log.d("PostureDebug", "=== REORDERED KEYPOINTS FOR POSTURE ===")
        val expectedOrder = listOf(
            "left_ankle", "left_ear", "left_elbow", "left_eye", "left_hip",
            "left_knee", "left_shoulder", "left_wrist", "nose",
            "right_ankle", "right_ear", "right_elbow", "right_eye",
            "right_hip", "right_knee", "right_shoulder", "right_wrist"
        )
        
        for ((i, keypoint) in reordered.withIndex()) {
            val keypointName = expectedOrder.getOrElse(i) { "unknown_$i" }
            Log.d("PostureDebug", "$i: $keypointName -> (${keypoint.first}, ${keypoint.second})")
        }
        
        return reordered
    }

    /**
     * Detect keypoints and immediately predict posture using the provided bounding box.
     * @param bitmap The input image
     * @param bbox The bounding box in original image coordinates
     * @return KeypointAndPostureResult or null if detection fails
     */
    fun detectKeypointsAndPosture(
        bitmap: Bitmap,
        bbox: android.graphics.RectF
    ): KeypointAndPostureResult? {
        val interpreter = interpreter ?: run {
            Log.w("PostureDetectionClustering", "Interpreter is null")
            return null
        }
        try {
            val origWidth = bitmap.width
            val origHeight = bitmap.height
            val scaledBitmap = preprocessImage(bitmap)
            // Use TensorImage for correct dtype and buffer
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(scaledBitmap)
            val inputBuffer = tensorImage.buffer
            // Output shape is [1, 1, 17, 3]
            val output = Array(1) { Array(1) { Array(numKeypoints) { FloatArray(3) } } }
            interpreter.run(inputBuffer, output)
            val keypoints = mutableListOf<Triple<Float, Float, Float>>()
            var minConfidence = Float.MAX_VALUE
            var minConfidenceKeypoint = -1
            
            for (i in 0 until numKeypoints) {
                // Output is [batch, person, keypoint, values]
                val y = output[0][0][i][0] * origHeight
                val x = output[0][0][i][1] * origWidth
                val score = output[0][0][i][2]
                keypoints.add(Triple(x, y, score))
                
                // Track minimum confidence
                if (score < minConfidence) {
                    minConfidence = score
                    minConfidenceKeypoint = i
                }
            }
            
            val keypointNames = listOf(
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            )
            
            Log.d("PostureDetectionClustering", "Detected ${keypoints.size} keypoints.")
            Log.d("PostureDetectionClustering", "Lowest keypoint confidence: ${minConfidence * 100}% (${keypointNames.getOrElse(minConfidenceKeypoint) { "unknown" }} at index $minConfidenceKeypoint)")
            
            // Reorder keypoints for posture prediction
            val reorderedKeypoints = reorderKeypointsForPosture(keypoints)
            val posture = predictPosture(reorderedKeypoints, bbox)
            
            return KeypointAndPostureResult(
                keypoints = keypoints,
                postureCluster = posture?.first,
                isGoodPosture = posture?.second,
                postureConfidence = posture?.third
            )
        } catch (e: Exception) {
            Log.e("PostureDetectionClustering", "Error in keypoint and posture detection", e)
            return null
        }
    }

    /**
     * Predict posture cluster and good/bad label from keypoints and bounding box
     * @param keypoints List of (x, y) keypoints in expected order for scaler/PCA
     * @param bbox Bounding box (original image coordinates)
     * @return Triple(cluster: Int, isGoodPosture: Boolean, confidence: Double)
     */
    fun predictPosture(
        keypoints: List<Pair<Float, Float>>,
        bbox: android.graphics.RectF
    ): Triple<Int, Boolean, Double>? {
        val params = postureParams ?: return null
        if (keypoints.size * 2 != params.scaler.mean.size) return null
        
        // Debug: Print input
        Log.d("PostureDebug", "=== POSTURE PREDICTION DEBUG ===")
        Log.d("PostureDebug", "Input keypoints (reordered): ${keypoints.take(15)}... (showing first 15)")
        Log.d("PostureDebug", "Bounding box: left=${bbox.left}, top=${bbox.top}, right=${bbox.right}, bottom=${bbox.bottom}")
        
        // 1. Normalize keypoints relative to bbox (build canonical features: all Xs then all Ys in expected order)
        val bboxWidth = (bbox.right - bbox.left).toDouble()
        val bboxHeight = (bbox.bottom - bbox.top).toDouble()
        val normX = MutableList(17) { 0.0 }
        val normY = MutableList(17) { 0.0 }
        for (i in 0 until 17) {
            val kp = keypoints[i]
            normX[i] = ((kp.first - bbox.left) / bboxWidth).toDouble()
            normY[i] = ((kp.second - bbox.top) / bboxHeight).toDouble()
        }
        val features = mutableListOf<Double>()
        features.addAll(normX)
        features.addAll(normY)

        // Optional: Log normalized values in MoveNet order for comparison
        Log.d("PostureDebug", "=== NORMALIZED FEATURES (Original MoveNet Order - display only) ===")
        val movenetOrder = listOf(
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        )
        val reverseMapping = keypointMapping.entries.associate { it.value to it.key }
        for (i in 0 until 17) {
            val movenetIndex = reverseMapping[i] ?: i
            val keypointName = movenetOrder.getOrElse(movenetIndex) { "unknown_$movenetIndex" }
            // For display, reconstruct interleaved view from canonical blocks
            val xDisp = normX[i]
            val yDisp = normY[i]
            Log.d("PostureDebug", "$movenetIndex: $keypointName -> x=$xDisp, y=$yDisp")
        }

        // 2. Scale (StandardScaler) aligned to canonical order [all Xs, then all Ys]
        val scaled = mutableListOf<Double>()
        Log.d("PostureDebug", "=== MEAN/SCALE USED (Expected Order, X block then Y block) ===")
        val expectedOrder = listOf(
            "left_ankle", "left_ear", "left_elbow", "left_eye", "left_hip",
            "left_knee", "left_shoulder", "left_wrist", "nose",
            "right_ankle", "right_ear", "right_elbow", "right_eye",
            "right_hip", "right_knee", "right_shoulder", "right_wrist"
        )
        // X block
        for (i in 0 until 17) {
            val meanX = params.scaler.mean[i]
            val scaleX = params.scaler.scale[i]
            val sX = (normX[i] - meanX) / scaleX
            scaled.add(sX)
            Log.d("PostureDebug", "$i: ${expectedOrder[i]}_x -> mean=$meanX, scale=$scaleX, scaled=$sX")
        }
        // Y block
        for (i in 0 until 17) {
            val meanY = params.scaler.mean[17 + i]
            val scaleY = params.scaler.scale[17 + i]
            val sY = (normY[i] - meanY) / scaleY
            scaled.add(sY)
            Log.d("PostureDebug", "${17 + i}: ${expectedOrder[i]}_y -> mean=$meanY, scale=$scaleY, scaled=$sY")
        }

        // Debug: Show PCA input vector shape and first 10
        Log.d("PostureDebug", "=== PCA INPUT (canonical: X[0..16], Y[0..16]) ===")
        Log.d("PostureDebug", "size=${scaled.size}, head10=${scaled.take(10)}")

        // 3. PCA (matrix multiplication over canonical order)
        val pca = params.pca.components.map { row -> row.mapIndexed { idx, w -> w * scaled[idx] }.sum() }

        // Debug: Print PCA output
        Log.d("PostureDebug", "PCA output: $pca")
        
        // 4. KMeans (find nearest center)
        val dists = params.kmeans.centers.map { center ->
            center.zip(pca).sumOf { (c, v) -> (c - v) * (c - v) }
        }
        
        // Debug: Print distances to each cluster
        Log.d("PostureDebug", "Distances to clusters: $dists")
        
        val cluster = dists.indices.minByOrNull { dists[it] } ?: -1
        val confidence = 1.0 / (1.0 + dists[cluster])
        val isGood = cluster == 2 || cluster == 3
        
        // Debug: Print final result
        Log.d("PostureDebug", "Final result: cluster=$cluster, isGood=$isGood, confidence=$confidence")
        
        return Triple(cluster, isGood, confidence)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
} 