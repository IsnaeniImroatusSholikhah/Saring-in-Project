package com.aplikasi.deteksihoax.utils

import com.google.gson.annotations.SerializedName

data class SOTModel(
    @SerializedName("news_source_of_truth" ) var news_source_of_truth : ArrayList<String>? = arrayListOf(),
)