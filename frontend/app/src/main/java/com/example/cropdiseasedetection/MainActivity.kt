package com.example.cropdiseasedetection

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.View
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody

import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var heatmapView: ImageView
    private lateinit var resultText: TextView
    private lateinit var btnGradCAM: MaterialButton
    private lateinit var btnLearnMore: MaterialButton
    private lateinit var checkboxUncertainty: CheckBox

    private var originalBitmap: Bitmap?  = null
    private var overlayBitmap: Bitmap?   = null
    private var isShowingHeatmap         = false

    // Store disease info for Learn More
    private var currentDiseaseInfo: DiseaseInfo? = null
    private var currentPrediction: String        = ""

    private val REQUEST_IMAGE  = 1
    private val REQUEST_CAMERA = 2

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView           = findViewById(R.id.imageView)
        heatmapView         = findViewById(R.id.heatmapView)
        resultText          = findViewById(R.id.resultText)
        btnGradCAM          = findViewById(R.id.btnGradCAM)
        btnLearnMore        = findViewById(R.id.btnInfo)
        checkboxUncertainty = findViewById(R.id.checkboxUncertainty)

        val btnSelect = findViewById<MaterialButton>(R.id.btnSelect)
        val btnCamera = findViewById<MaterialButton>(R.id.btnCamera)

        btnSelect.setOnClickListener  { openGallery()    }
        btnCamera.setOnClickListener  { openCamera()     }
        btnGradCAM.setOnClickListener { toggleHeatmap()  }

        btnLearnMore.setOnClickListener {
            if (currentDiseaseInfo != null) {
                openDiseaseInfo()
            }
        }

        // Initially disable both buttons
        btnLearnMore.isEnabled = false
        btnGradCAM.isEnabled   = false
    }

    // -------------------
    // OPEN DISEASE INFO
    // -------------------

    private fun openDiseaseInfo() {
        val intent = Intent(this, DiseaseInfoActivity::class.java)
        intent.putExtra("disease_name", currentPrediction)
        intent.putExtra("about",        currentDiseaseInfo?.about    ?: "")
        intent.putExtra("symptoms",     currentDiseaseInfo?.symptoms  ?: "")
        intent.putExtra("remedies",     currentDiseaseInfo?.remedies  ?: "")
        intent.putStringArrayListExtra(
            "precautions",
            ArrayList(currentDiseaseInfo?.precautions ?: emptyList())
        )
        startActivity(intent)
    }

    // -------------------
    // TOGGLE HEATMAP
    // -------------------

    private fun toggleHeatmap() {
        if (overlayBitmap == null) return

        if (isShowingHeatmap) {
            imageView.setImageBitmap(originalBitmap)
            btnGradCAM.text  = "Show Grad-CAM"
            isShowingHeatmap = false
        } else {
            imageView.setImageBitmap(overlayBitmap)
            btnGradCAM.text  = "Hide Grad-CAM"
            isShowingHeatmap = true
        }
    }

    // -------------------
    // OPEN GALLERY
    // -------------------

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, REQUEST_IMAGE)
    }

    // -------------------
    // OPEN CAMERA
    // -------------------

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, REQUEST_CAMERA)
    }

    // -------------------
    // HANDLE RESULTS
    // -------------------

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        super.onActivityResult(requestCode, resultCode, data)

        // Reset state
        isShowingHeatmap       = false
        overlayBitmap          = null
        originalBitmap         = null
        currentDiseaseInfo     = null
        currentPrediction      = ""
        btnGradCAM.isEnabled   = false
        btnGradCAM.text        = "Show Grad-CAM"
        btnLearnMore.isEnabled = false
        heatmapView.visibility = View.GONE

        // ---------- Gallery ----------
        if (requestCode == REQUEST_IMAGE && resultCode == Activity.RESULT_OK) {

            val uri: Uri? = data?.data

            val bitmap: Bitmap =
                if (Build.VERSION.SDK_INT >= 28) {
                    val source = ImageDecoder.createSource(contentResolver, uri!!)
                    ImageDecoder.decodeBitmap(source)
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(contentResolver, uri)
                }

            originalBitmap = bitmap
            imageView.setImageBitmap(bitmap)
            uploadImage(uri!!)
        }

        // ---------- Camera ----------
        if (requestCode == REQUEST_CAMERA && resultCode == Activity.RESULT_OK) {
            val bitmap     = data?.extras?.get("data") as Bitmap
            originalBitmap = bitmap
            imageView.setImageBitmap(bitmap)
            uploadBitmap(bitmap)
        }
    }

    // -------------------
    // UPLOAD FROM GALLERY
    // -------------------

    private fun uploadImage(uri: Uri) {
        uploadFile(uriToFile(uri))
    }

    // -------------------
    // UPLOAD FROM CAMERA
    // -------------------

    private fun uploadBitmap(bitmap: Bitmap) {
        val file   = File(cacheDir, "camera_image.jpg")
        val stream = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
        stream.flush()
        stream.close()
        uploadFile(file)
    }

    // -------------------
    // COMMON UPLOAD
    // -------------------

    private fun uploadFile(file: File) {

        val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
        val imagePart   = MultipartBody.Part.createFormData("image", file.name, requestFile)

        val useUncertainty  = checkboxUncertainty.isChecked
        val uncertaintyBody = useUncertainty.toString()
            .toRequestBody("text/plain".toMediaTypeOrNull())

        resultText.text = if (useUncertainty)
            "Processing with uncertainty estimation..."
        else
            "Analyzing image..."

        val call = RetrofitClient.instance.uploadImage(imagePart, uncertaintyBody)

        call.enqueue(object : Callback<PredictionResponse> {

            override fun onResponse(
                call: Call<PredictionResponse>,
                response: Response<PredictionResponse>
            ) {
                if (response.isSuccessful) {

                    val body = response.body()

                    // Handle error cases
                    if (body?.error != null) {
                        resultText.text        = "⚠️ ${body.message}"
                        btnGradCAM.isEnabled   = false
                        btnLearnMore.isEnabled = false
                        return
                    }

                    // Handle successful prediction
                    val prediction    = body?.prediction   ?: ""
                    val confidence    = body?.confidence
                    val heatmapBase64 = body?.heatmap
                    val uncertainty   = body?.uncertainty
                    val diseaseInfo   = body?.disease_info

                    currentPrediction  = prediction
                    currentDiseaseInfo = diseaseInfo

                    // Build result text
                    val resultBuilder = StringBuilder()
                    resultBuilder.append("Prediction: $prediction\n")
                    resultBuilder.append("Confidence: $confidence %")
                    if (uncertainty != null) {
                        resultBuilder.append("\nUncertainty: $uncertainty")
                    }
                    resultText.text = resultBuilder.toString()

                    // Enable Learn More if disease info available
                    if (diseaseInfo != null) {
                        btnLearnMore.isEnabled = true
                    }

                    // Decode overlay bitmap
                    if (!heatmapBase64.isNullOrEmpty()) {
                        try {
                            val decodedBytes = Base64.decode(heatmapBase64, Base64.DEFAULT)
                            overlayBitmap    = BitmapFactory.decodeByteArray(
                                decodedBytes, 0, decodedBytes.size
                            )
                            btnGradCAM.isEnabled = true
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }

                } else {
                    resultText.text = "Server Error: ${response.code()}"
                }
            }

            override fun onFailure(
                call: Call<PredictionResponse>,
                t: Throwable
            ) {
                resultText.text = "Failed: ${t.message}"
            }
        })
    }

    // -------------------
    // URI → FILE
    // -------------------

    private fun uriToFile(uri: Uri): File {
        val inputStream  = contentResolver.openInputStream(uri)
        val file         = File(cacheDir, "temp_image.jpg")
        val outputStream = FileOutputStream(file)
        inputStream!!.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
        return file
    }
}