package com.bmiapp.estimator

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.IOException

class ImagePreprocessor {
    companion object {
        private const val TARGET_SIZE = 224
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
        private const val TAG = "ImagePreprocessor"
    }

    /**
     * Preprocess an image file to the format expected by the BMI model
     * @param imagePath Path to the image file
     * @return Tensor ready for model inference
     */
    fun preprocessImage(imagePath: String): Tensor {
        val bitmap = BitmapFactory.decodeFile(imagePath)
        if (bitmap == null) {
            throw IllegalArgumentException("Failed to decode image from path: $imagePath")
        }
        return preprocessBitmap(bitmap)
    }

    /**
     * Preprocess an image from a content URI to the format expected by the BMI model
     * @param context Android context
     * @param imageUri Content URI of the image
     * @return Tensor ready for model inference
     */
    fun preprocessImageFromUri(context: Context, imageUri: String): Tensor {
        val bitmap = loadBitmapFromUri(context, imageUri)
        return preprocessBitmap(bitmap)
    }

    /**
     * Load a Bitmap from a content URI
     * @param context Android context
     * @param imageUri Content URI of the image
     * @return Bitmap or null if loading failed
     */
    fun loadBitmapFromUri(context: Context, imageUri: String): Bitmap {
        val uri = Uri.parse(imageUri)
        val bitmap = try {
            context.contentResolver.openInputStream(uri)?.use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error reading image from URI: $imageUri", e)
            null
        }
        
        if (bitmap == null) {
            throw IllegalArgumentException("Failed to decode image from URI: $imageUri")
        }
        
        return bitmap
    }

    /**
     * Preprocess a Bitmap to the format expected by the BMI model
     * @param bitmap Input bitmap
     * @return Tensor ready for model inference
     */
    fun preprocessBitmap(bitmap: Bitmap): Tensor {
        val (resizedBitmap, _, _) = resizeAndPad(bitmap, TARGET_SIZE)
        
        // Convert to float array with normalization
        val inputArray = FloatArray(1 * 3 * TARGET_SIZE * TARGET_SIZE)
        for (y in 0 until TARGET_SIZE) {
            for (x in 0 until TARGET_SIZE) {
                val pixel = resizedBitmap.getPixel(x, y)
                val r = (pixel shr 16 and 0xFF) / 255.0f
                val g = (pixel shr 8 and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                
                // Normalize using ImageNet statistics
                val normalizedR = (r - MEAN_R) / STD_R
                val normalizedG = (g - MEAN_G) / STD_G
                val normalizedB = (b - MEAN_B) / STD_B
                
                // CHW format
                val idx = y * TARGET_SIZE + x
                inputArray[idx] = normalizedR
                inputArray[TARGET_SIZE * TARGET_SIZE + idx] = normalizedG
                inputArray[2 * TARGET_SIZE * TARGET_SIZE + idx] = normalizedB
            }
        }
        
        return Tensor.fromBlob(inputArray, longArrayOf(1, 3, TARGET_SIZE.toLong(), TARGET_SIZE.toLong()))
    }

    /**
     * Resize image with aspect ratio preservation and pad to target size
     * @param bitmap Input bitmap
     * @param targetSize Target size (width and height)
     * @return Triple of (padded bitmap, left padding, top padding)
     */
    private fun resizeAndPad(bitmap: Bitmap, targetSize: Int): Triple<Bitmap, Int, Int> {
        val w = bitmap.width
        val h = bitmap.height
        val scale = targetSize.toFloat() / maxOf(w, h)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val padded = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        
        val padLeft = (targetSize - newW) / 2
        val padTop = (targetSize - newH) / 2
        
        val canvas = android.graphics.Canvas(padded)
        canvas.drawARGB(0, 0, 0, 0) // Fill with transparent black
        canvas.drawBitmap(resized, padLeft.toFloat(), padTop.toFloat(), null)
        
        return Triple(padded, padLeft, padTop)
    }
} 