package com.aplikasi.deteksihoax.utils

data class DetectionModel(
    val berita: String? = "",
    val content: String? = "",
    val recommendation: ArrayList<String>? = arrayListOf(),
    val id: String? = ""
): java.io.Serializable
