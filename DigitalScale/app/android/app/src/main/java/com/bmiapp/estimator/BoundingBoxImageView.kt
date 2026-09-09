package com.bmiapp.estimator

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import androidx.appcompat.widget.AppCompatImageView

/**
 * Custom ImageView that displays an image with a bounding box overlay
 */
class BoundingBoxImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : AppCompatImageView(context, attrs, defStyleAttr) {

    private val boundingBoxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }

    private val confidencePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        textSize = 32f
        isAntiAlias = true
    }

    private var efficientDetBox: RectF? = null
    private var confidence: Float = 0f
    private var originalImageWidth: Int = 0
    private var originalImageHeight: Int = 0

    /**
     * Set the bounding box to display (EfficientDet RectF format)
     */
    fun setBoundingBox(box: RectF?, originalWidth: Int = 0, originalHeight: Int = 0, conf: Float = 0f) {
        efficientDetBox = box
        confidence = conf
        originalImageWidth = originalWidth
        originalImageHeight = originalHeight
        android.util.Log.d("BoundingBoxImageView", "Setting EfficientDet box: $box, confidence: $conf, original size: ${originalWidth}x${originalHeight}")
        invalidate() // Redraw the view
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Handle EfficientDet RectF format
        efficientDetBox?.let { box ->
            // Calculate scaling factors that preserve aspect ratio
            val imageAspectRatio = originalImageWidth.toFloat() / originalImageHeight
            val viewAspectRatio = width.toFloat() / height
            
            val scaleX: Float
            val scaleY: Float
            val offsetX: Float
            val offsetY: Float
            
            if (imageAspectRatio > viewAspectRatio) {
                // Image is wider than view - fit to width
                scaleX = width.toFloat() / originalImageWidth
                scaleY = scaleX // Same scale to preserve aspect ratio
                offsetX = 0f
                offsetY = (height - (originalImageHeight * scaleY)) / 2f
            } else {
                // Image is taller than view - fit to height
                scaleY = height.toFloat() / originalImageHeight
                scaleX = scaleY // Same scale to preserve aspect ratio
                offsetX = (width - (originalImageWidth * scaleX)) / 2f
                offsetY = 0f
            }
            
            android.util.Log.d("BoundingBoxImageView", "View size: ${width}x${height}, image: ${originalImageWidth}x${originalImageHeight}")
            android.util.Log.d("BoundingBoxImageView", "Image aspect: $imageAspectRatio, view aspect: $viewAspectRatio")
            android.util.Log.d("BoundingBoxImageView", "Scale: ${scaleX}x${scaleY}, offset: ${offsetX}x${offsetY}")
            
            // Scale bounding box coordinates to match displayed image
            val scaledX1 = box.left * scaleX + offsetX
            val scaledY1 = box.top * scaleY + offsetY
            val scaledX2 = box.right * scaleX + offsetX
            val scaledY2 = box.bottom * scaleY + offsetY
            
            // Calculate aspect ratios for comparison
            val originalWidth = box.right - box.left
            val originalHeight = box.bottom - box.top
            val originalAspectRatio = originalWidth / originalHeight
            
            val drawnWidth = scaledX2 - scaledX1
            val drawnHeight = scaledY2 - scaledY1
            val drawnAspectRatio = drawnWidth / drawnHeight
            
            android.util.Log.d("BoundingBoxImageView", "=== DRAWING ASPECT RATIO COMPARISON ===")
            android.util.Log.d("BoundingBoxImageView", "Original box: ${box}")
            android.util.Log.d("BoundingBoxImageView", "Original dimensions: ${originalWidth}x${originalHeight}")
            android.util.Log.d("BoundingBoxImageView", "Original aspect ratio: $originalAspectRatio")
            android.util.Log.d("BoundingBoxImageView", "Drawn box: (${scaledX1}, ${scaledY1}, ${scaledX2}, ${scaledY2})")
            android.util.Log.d("BoundingBoxImageView", "Drawn dimensions: ${drawnWidth}x${drawnHeight}")
            android.util.Log.d("BoundingBoxImageView", "Drawn aspect ratio: $drawnAspectRatio")
            android.util.Log.d("BoundingBoxImageView", "Aspect ratio preserved in drawing: ${kotlin.math.abs(originalAspectRatio - drawnAspectRatio) < 0.01}")
            android.util.Log.d("BoundingBoxImageView", "=== END DRAWING COMPARISON ===")
            
            // Draw bounding box
            val rect = RectF(scaledX1, scaledY1, scaledX2, scaledY2)
            canvas.drawRect(rect, boundingBoxPaint)
        }
    }
} 