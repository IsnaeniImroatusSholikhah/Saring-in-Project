package com.aplikasi.deteksihoax.network

import com.aplikasi.deteksihoax.utils.PredictModel
import com.aplikasi.deteksihoax.utils.RecomModel
import com.aplikasi.deteksihoax.utils.SOTModel
import retrofit2.Response
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.POST

interface ApiService {

    @FormUrlEncoded
    @POST("get-source-of-truth")
    suspend fun getSot(
        @Field("title") title: String,
    ): Response<SOTModel>


    @FormUrlEncoded
    @POST("predict")
    suspend fun predict(
        @Field("source_of_truth") source_of_truth: String,
        @Field("news_text") news_text: String
    ): Response<PredictModel>


    @FormUrlEncoded
    @POST("get-recomendation")
    suspend fun getRecommendation(
        @Field("news_text") news_text: String,
    ): Response<PredictModel>
}