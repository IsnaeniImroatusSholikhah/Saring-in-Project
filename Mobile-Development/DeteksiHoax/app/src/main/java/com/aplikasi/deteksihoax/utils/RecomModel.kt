package com.aplikasi.deteksihoax.utils

import com.google.gson.annotations.SerializedName

data class RecomModel(
    @SerializedName("recomendations" ) var recomendations : ArrayList<String>? = arrayListOf(),
    var predictModel: PredictModel?
)