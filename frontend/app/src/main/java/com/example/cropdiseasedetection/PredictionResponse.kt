package com.example.cropdiseasedetection

data class DiseaseInfo(
    val about:       String?       = null,
    val symptoms:    String?       = null,
    val remedies:    String?       = null,
    val precautions: List<String>? = null
)

data class PredictionResponse(
    val prediction:   String?      = null,
    val confidence:   Double?      = null,
    val heatmap:      String?      = null,
    val uncertainty:  Double?      = null,
    val disease_info: DiseaseInfo? = null,
    val error:        String?      = null,
    val message:      String?      = null
)