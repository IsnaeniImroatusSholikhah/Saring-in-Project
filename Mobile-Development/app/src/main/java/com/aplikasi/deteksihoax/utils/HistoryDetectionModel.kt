package com.aplikasi.deteksihoax.utils

import com.google.gson.annotations.SerializedName

data class HistoryDetectionModel(
    val detection_result: String? = "",
    val news_content: String? = "",
    val news_title: String? = "",
    val recommendation: ArrayList<String>? = arrayListOf(),
): java.io.Serializable