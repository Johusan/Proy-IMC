package com.bmiapp.estimator

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class SkeletonOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private var keypoints: List<Triple<Float, Float, Float>>? = null
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    // COCO 17 keypoint skeleton connections
    private val skeleton = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4, // Nose -> REye -> REar, Nose -> LEye -> LEar
        0 to 5, 0 to 6, // Nose -> LShoulder, Nose -> RShoulder
        5 to 7, 7 to 9, // LShoulder -> LElbow -> LWrist
        6 to 8, 8 to 10, // RShoulder -> RElbow -> RWrist
        5 to 6, // LShoulder <-> RShoulder
        5 to 11, 6 to 12, // LShoulder -> LHip, RShoulder -> RHip
        11 to 12, // LHip <-> RHip
        11 to 13, 13 to 15, // LHip -> LKnee -> LAnkle
        12 to 14, 14 to 16 // RHip -> RKnee -> RAnkle
    )
    // COCO skeleton colors: head (yellow), arms (cyan/blue), torso (green), legs (magenta)
    private val skeletonColors = listOf(
        Color.YELLOW, // Nose -> REye
        Color.YELLOW, // REye -> REar
        Color.YELLOW, // Nose -> LEye
        Color.YELLOW, // LEye -> LEar
        Color.CYAN,   // Nose -> LShoulder
        Color.CYAN,   // Nose -> RShoulder
        Color.BLUE,   // LShoulder -> LElbow
        Color.BLUE,   // LElbow -> LWrist
        Color.BLUE,   // RShoulder -> RElbow
        Color.BLUE,   // RElbow -> RWrist
        Color.GREEN,  // LShoulder <-> RShoulder
        Color.GREEN,  // LShoulder -> LHip
        Color.GREEN,  // RShoulder -> RHip
        Color.GREEN,  // LHip <-> RHip
        Color.MAGENTA,// LHip -> LKnee
        Color.MAGENTA,// LKnee -> LAnkle
        Color.MAGENTA,// RHip -> RKnee
        Color.MAGENTA // RKnee -> RAnkle
    )
    private val keypointPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        strokeWidth = 8f
        isAntiAlias = true
    }
    private val linePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    fun setKeypoints(keypoints: List<Triple<Float, Float, Float>>?, imageWidth: Int, imageHeight: Int) {
        this.keypoints = keypoints
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    private fun mapPoint(x: Float, y: Float): Pair<Float, Float> {
        // Map (x, y) from original image size to displayed image rect in this view
        if (imageWidth == 0 || imageHeight == 0) return x to y
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val scale = minOf(viewWidth / imageWidth, viewHeight / imageHeight)
        val scaledImageWidth = imageWidth * scale
        val scaledImageHeight = imageHeight * scale
        val dx = (viewWidth - scaledImageWidth) / 2f
        val dy = (viewHeight - scaledImageHeight) / 2f
        val mappedX = x * scale + dx
        val mappedY = y * scale + dy
        return mappedX to mappedY
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val kp = keypoints ?: return
        // Draw lines
        for ((i, pair) in skeleton.withIndex()) {
            val (start, end) = pair
            if (start < kp.size && end < kp.size) {
                val (x1, y1, s1) = kp[start]
                val (x2, y2, s2) = kp[end]
                if (s1 > 0.2f && s2 > 0.2f) {
                    val (mx1, my1) = mapPoint(x1, y1)
                    val (mx2, my2) = mapPoint(x2, y2)
                    linePaint.color = skeletonColors.getOrElse(i) { Color.GREEN }
                    canvas.drawLine(mx1, my1, mx2, my2, linePaint)
                }
            }
        }
        // Draw keypoints
        for ((x, y, score) in kp) {
            if (score > 0.2f) {
                val (mx, my) = mapPoint(x, y)
                canvas.drawCircle(mx, my, 10f, keypointPaint)
            }
        }
    }
} 