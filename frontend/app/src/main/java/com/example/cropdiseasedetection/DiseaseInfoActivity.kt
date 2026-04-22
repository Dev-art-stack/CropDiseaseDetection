package com.example.cropdiseasedetection

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class DiseaseInfoActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_disease_info)

        // Retrieve data passed from MainActivity
        val diseaseName  = intent.getStringExtra("disease_name")  ?: "Unknown"
        val about        = intent.getStringExtra("about")         ?: "-"
        val symptoms     = intent.getStringExtra("symptoms")      ?: "-"
        val remedies     = intent.getStringExtra("remedies")      ?: "-"
        val precautions  = intent.getStringArrayListExtra("precautions") ?: arrayListOf()

        // Bind views
        val tvTitle        = findViewById<TextView>(R.id.tvDiseaseTitle)
        val tvAbout        = findViewById<TextView>(R.id.tvAbout)
        val tvSymptoms     = findViewById<TextView>(R.id.tvSymptoms)
        val tvRemedies     = findViewById<TextView>(R.id.tvRemedies)
        val tvPrecautions  = findViewById<TextView>(R.id.tvPrecautions)
        val btnBack        = findViewById<MaterialButton>(R.id.btnBack)

        // Format disease name nicely
        val formattedName = diseaseName
            .replace("___", " - ")
            .replace("__", " ")
            .replace("_", " ")

        tvTitle.text       = formattedName
        tvAbout.text       = about
        tvSymptoms.text    = symptoms
        tvRemedies.text    = remedies

        // Format precautions as numbered list
        val precautionText = precautions
            .mapIndexed { i, p -> "${i + 1}. $p" }
            .joinToString("\n")
        tvPrecautions.text = precautionText

        btnBack.setOnClickListener { finish() }
    }
}