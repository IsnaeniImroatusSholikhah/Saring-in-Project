package com.aplikasi.deteksihoax.utils

import com.google.gson.annotations.SerializedName

data class PredictModel(
    @SerializedName("predicition_result" ) var predicition_result : String? = "",
    @SerializedName("recomendations" ) var recomendations : ArrayList<String>? = arrayListOf(),
    @SerializedName("results"        ) var results        : ArrayList<String>? = arrayListOf()
)